import numpy as np
import torch as th
from tqdm.auto import tqdm
import logging
from utils.data.coco_dataset import coco_max_num_bboxes, coco_bb_count_pmf, coco_num_classes, coco_class_pmf
import pytorch_lightning as pl
from utils.data.BBox import BBox, create_bbox_like

logger = logging.getLogger('my_task')


class ReverseGaussianDiffusion(pl.LightningModule):
    """
    Utilities for reverse Gaussian Diffusion (i.e sampling from the model).

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    # TODO: look into rescale timesteps
    def __init__(self, beta_scheduler, model_mean_type, model_var_type,
                 rescale_timesteps=False, sampler='ddpm', eta=1.0, predict_class=False,
                 loss_type=None, train_cls_fmt='bits'):
        super().__init__()
        self.beta_scheduler = beta_scheduler
        self.rescale_timesteps = rescale_timesteps
        self.variance_transformer = ModelVarianceTransformer(beta_scheduler, model_var_type)

        if model_mean_type == 'prev_x':
            self.pred_xstart = self._predict_xstart_from_xprev
            self.extract_mean = self.identity

        elif model_mean_type == 'epsilon':
            self.pred_xstart = self._predict_xstart_from_eps
            self.extract_mean = self.beta_scheduler.q_posterior_mean

        elif model_mean_type == 'start_x':
            self.pred_xstart = self.identity
            self.extract_mean = self.beta_scheduler.q_posterior_mean

        else:
            raise NotImplementedError(model_mean_type)

        if sampler == 'ddpm' or sampler is None:
            self.p_sample_fn = self.p_sample_ddpm
        elif sampler == 'ddim':
            self.p_sample_fn = self.p_sample_ddim
        else:
            raise NotImplementedError(sampler)
        self.eta = eta
        self.predict_class = predict_class
        self.x_t_cls_fmt = train_cls_fmt if predict_class else None
        self.simple_regression = loss_type == 'simple_regression'

    def identity(self, model_mean, *args, **kwargs):
        return model_mean

    def _predict_xstart_from_eps(self, eps, x_t, t):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.beta_scheduler.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.beta_scheduler.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, xprev, x_t, t):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.beta_scheduler.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.beta_scheduler.posterior_mean_coef2 / self.beta_scheduler.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, pred_xstart, x_t, t):
        return (
            _extract_into_tensor(self.beta_scheduler.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.beta_scheduler.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.beta_scheduler.num_timesteps)
        return t

    def extract_p_mean_variance(self, model_mean, model_var_out, x_t, t, clip_denoised=True):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        model_log_var, model_var = self.variance_transformer.transform(
            model_var_out=model_var_out, t=t, x_shape=x_t.shape)
        pred_xstart = self.pred_xstart(model_mean, x_t, t)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        model_mean = self.extract_mean(
            x_start=pred_xstart, model_mean=model_mean, x_t=x_t, t=t)

        assert (
            model_mean.shape == model_log_var.shape == pred_xstart.shape == x_t.shape
        )
        res = {
            "mean": model_mean,
            "variance": model_var,
            "log_variance": model_log_var,
            "pred_xstart": pred_xstart,
        }

        return res

    @th.no_grad()
    def sample(self, model, batch, num_steps_to_ret=10, clip_denoised=True):
        logger.info('Generating samples')
        assert not hasattr(model, 'cached_bbone_res')

        if self.simple_regression:
            num_timesteps = 1
        else:
            num_timesteps = self.beta_scheduler.num_timesteps
        steps_to_return = np.linspace(
            0, num_timesteps,
            num=min(10, num_timesteps + 1)).astype(np.int32).tolist()

        model_out = self.p_sample_loop(
            model, batch, steps_to_return, clip_denoised=clip_denoised)

        model_out['steps_to_return'] = steps_to_return
        return model_out

        # return {'final_out': final_out,
                # 'all_x_t': all_x_t,
                # 'steps_to_return': steps_to_return,
                # 'bbone_preds': bbone_preds}

    def p_sample_loop(self, model, batch, steps_to_return, clip_denoised=True):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """

        if not self.simple_regression:
            indices = list(range(self.beta_scheduler.num_timesteps))[::-1]
        else:
            indices = [0]
        # indices = tqdm(indices)
        all_x_t = [batch['x_t']]
        bbone_preds = None
        for i in indices:
            batch['t'] = th.tensor([i], device=model.device).repeat(batch['x_t'].shape[0])

            result = self.p_sample_fn(
                model, batch, clip_denoised=clip_denoised, bbone_preds=bbone_preds)

            batch['x_t'] = result['sample'].detach()
            if not self.predict_class:
                x_t_bb = batch['x_t']
                x_t_cls = batch['x_start'].classes_in_train_fmt()
                x_t_mask = batch['x_start']['padding_mask']
            else:
                x_t_bb = batch['x_t'][..., :4]
                x_t_cls = batch['x_t'][..., 4:]
                x_t_mask = batch['x_start']['padding_mask']

            batch['x_t'] = create_bbox_like(x_t_bb, x_t_cls, x_t_mask, bbox_like=batch['x_start'], class_fmt=batch['x_start'].train_cls_fmt)
            bbone_preds = result['bbone_preds']

            if i in steps_to_return:
                all_x_t.append(batch['x_t'])
            # break

        batch['x_t']['bbox'] = batch['x_t']['bbox'].clamp(-1, 1)
        x_t_cls = batch['x_t'].classes_in_train_fmt().clamp(-1, 1)
        if batch['x_t'].train_cls_fmt == 'bits':
            batch['x_t'].classes_bits = x_t_cls
        elif batch['x_t'].train_cls_fmt == 'softmax':
            batch['x_t'].classes_softmax = x_t_cls
        else:
            raise Exception()

        return {'final_out': batch['x_t'],
                'all_x_t': all_x_t,
                'bbone_preds': bbone_preds}

    def p_sample_ddpm(self, model, batch, clip_denoised=True, bbone_preds=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out, bbone_preds = model(
            batch, clip_denoised=clip_denoised, bbone_preds=bbone_preds,
            inference=True)
        # for key, val in out.items():
        #     out[key] = val.flatten(start_dim=0, end_dim=1)

        x_t = batch['x_t'].get_features(cls_fmt=self.x_t_cls_fmt)
        # original_shape = x_t.shape
        # t = batch['t'].unsqueeze(-1).expand(x_t.shape[:-1]).flatten()
        # x_t = x_t.flatten(start_dim=0, end_dim=1)
        t = batch['t']

        noise = th.randn_like(x_t)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # return {"sample": sample.view(original_shape), "pred_xstart": out["pred_xstart"].view(original_shape),
        #         'bbone_preds': bbone_preds}
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'bbone_preds': bbone_preds}

    def p_sample_ddim(self, model, batch, clip_denoised=True, bbone_preds=None):
        out, bbone_preds = model(
            batch, clip_denoised=clip_denoised, bbone_preds=bbone_preds,
            inference=True)

        x_t = batch['x_t'].get_features(cls_fmt=self.x_t_cls_fmt)
        original_shape = x_t.shape
        t = batch['t'].unsqueeze(-1).expand(x_t.shape[:-1]).flatten()
        x_t = x_t.flatten(start_dim=0, end_dim=1)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(out["pred_xstart"], x_t, t)
        alpha_bar = _extract_into_tensor(self.beta_scheduler.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = _extract_into_tensor(self.beta_scheduler.alphas_cumprod_prev, t, x_t.shape)
        sigma = (
            self.eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x_t)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample.view(original_shape), "pred_xstart": out["pred_xstart"],
                'bbone_preds': bbone_preds}


class ModelVarianceTransformer(pl.LightningModule):
    def __init__(self, beta_scheduler, model_var_type):
        super().__init__()
        self.beta_scheduler = beta_scheduler

        if model_var_type == 'learned':
            self.transform = self.transform_var_learned
        elif model_var_type == 'learned_range':
            self.transform = self.transform_var_range

        elif model_var_type == 'fixed_small' or \
                model_var_type == 'fixed_large':
            if model_var_type == 'fixed_small':
                model_var = beta_scheduler.posterior_variance.clone()
                model_log_var = beta_scheduler.posterior_log_variance_clipped.clone()
                self.transform = self.model_fixed_var

            elif model_var_type == 'fixed_large':
                model_var = th.cat(
                    [self.beta_scheduler.posterior_variance[1].view(1),
                     self.beta_scheduler.betas[1:]]
                )
                model_log_var = th.log(model_var)
                self.transform = self.model_fixed_var
            self.register_buffer(
                'model_var', model_var
            )
            self.register_buffer(
                'model_log_var', model_log_var
            )
        else:
            raise Exception(model_var_type)

    def transform_var_learned(self, model_var_out, *args, **kwargs):
        return model_var_out, th.exp(model_var_out)

    def transform_var_range(self, model_var_out, t, x_shape):
        min_log = _extract_into_tensor(
            self.beta_scheduler.posterior_log_variance_clipped, t, x_shape
        )
        max_log = _extract_into_tensor(th.log(self.beta_scheduler.betas), t, x_shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_out + 1) / 2
        model_log_var = frac * max_log + (1 - frac) * min_log
        model_variance = th.exp(model_log_var)

        return model_log_var, model_variance

    def model_fixed_var(self, t, x_shape, **kwargs):
        model_variance = _extract_into_tensor(self.model_var, t, x_shape)
        model_log_variance = _extract_into_tensor(self.model_log_var, t, x_shape)
        return model_log_variance, model_variance


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # timesteps = timesteps.type(th.long)
    # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    # import ipdb; ipdb.set_trace()
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    # print(res.shape, broadcast_shape)
    return res.expand(broadcast_shape).type(th.float32)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return sorted(list(set(all_steps)))
