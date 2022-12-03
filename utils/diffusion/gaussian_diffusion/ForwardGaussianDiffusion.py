import math
from abc import ABC, abstractmethod
import numpy as np
import torch as th
import torch.distributed as dist
import logging
import pytorch_lightning as pl
import copy
from utils.data.BBox import BBox, create_bbox_like

logger = logging.getLogger('my_task')


# TODO: port all of the numpy to torch
class GaussianNoiser(pl.LightningModule):
    def __init__(self, beta_scheduler, sampler, loss_type=None):
        super().__init__()
        self.beta_scheduler = copy.deepcopy(beta_scheduler).to(self.device)
        self._create_sampler(sampler)
        self.simple_regression = loss_type == 'simple_regression'

    def _create_sampler(self, sampler):
        """
        Create a ScheduleSampler from a library of pre-defined samplers.

        :param name: the name of the sampler.
        :param diffusion: the diffusion object to sample for.
        """
        if sampler == "uniform":
            self.sampler = UniformSampler(self.beta_scheduler.num_timesteps)
        elif sampler == "loss-second-moment":
            self.sampler = LossSecondMomentResampler(self.beta_scheduler.num_timesteps)
        else:
            raise NotImplementedError(f"unknown schedule sampler: {sampler}")

    def sample_training_noise(self, batch):
        x_start = batch['x_start']
        noise_info = self.sample(x_start)
        batch.update(noise_info)
        return batch

    def sample(self, x_start, ts=None, weights=None, noise=None):
        ts, weights = self.get_ts_weights(x_start, ts, weights)
        if self.simple_regression:
            ts = th.zeros_like(ts) + self.beta_scheduler.num_timesteps - 1
        res = {'t': ts,
               'weights': weights}

        x_t, noise = self.sample_x_t(x_start, noise, ts)

        if self.simple_regression:
            res['t'] = th.zeros_like(ts)

        x_t_bb = x_t[..., :x_start['bbox'].shape[-1]]
        x_t_cls = x_t[..., x_start['bbox'].shape[-1]:]

        res['x_t'] = create_bbox_like(
            x_t_bb, x_t_cls, x_start['padding_mask'],
            bbox_like=x_start, class_fmt=x_start.train_cls_fmt)

        bb_noise = noise[..., :x_start['bbox'].shape[-1]]
        cls_noise = noise[..., x_start['bbox'].shape[-1]:]
        res['noise'] = {'noise': noise,
                        'bb': bb_noise,
                        'cls': cls_noise}
        return res

    def get_ts_weights(self, x_start, ts, weights):
        if ts is None:
            batch_size = x_start.shape[0]
            ts, weights = self.sampler.sample(batch_size)
            weights = weights.detach()

        return ts, weights

    def sample_x_t(self, x_start, noise, ts):
        x_start_tensor = th.cat([x_start['bbox'], x_start.classes_in_train_fmt()], dim=-1)
        if noise is None:
            noise = th.randn_like(x_start_tensor)
        x_t = self.q_sample(x_start_tensor, ts, noise=noise)
        return x_t, noise

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """

        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape, '{} {}'.format(noise.shape, x_start.shape)

        # self.beta_scheduler = self.beta_scheduler.to(self.device)
        # t = t.to(self.device)
        # print(_extract_into_tensor(self.beta_scheduler.sqrt_alphas_cumprod, t, x_start.shape).shape)
        # print(self.beta_scheduler.sqrt_alphas_cumprod.device, t.device, x_start.device, self.device)
        return (
            _extract_into_tensor(self.beta_scheduler.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.beta_scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def update_with_all_losses(self, ts, loss):
        return self.sampler.update_with_all_losses(ts, loss)

    def update_with_local_losses(self, ts, loss):
        return self.sampler.update_with_local_losses(ts, loss)


class BetaScheduler(pl.LightningModule):
    def __init__(self, schedule_name, num_diffusion_timesteps, log_scale=None, betas=None):
        super().__init__()

        if betas is None:
            betas = self._get_named_beta_schedule(schedule_name, num_diffusion_timesteps, log_scale)
        self._init_diffusion_info(betas)

    def _init_diffusion_info(self, betas):
        # Use float64 for accuracy.
        self.register_buffer(
            'betas',
            betas.type(th.float64)
        )
        assert len(self.betas.shape) == 1, 'betas must be 1D'
        assert (self.betas > 0).all() and (self.betas <= 1).all(), self.betas

        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas
        self.register_buffer(
            'alphas_cumprod',
            th.cumprod(alphas, dim=0)
        )
        self.register_buffer(
            'alphas_cumprod_prev',
            th.cat(
                [th.tensor([1]), self.alphas_cumprod[:-1]]
            )
        )
        self.register_buffer(
            'alphas_cumprod_next',
            th.cat(
                [self.alphas_cumprod[1:], th.tensor([0])]
            )
        )
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,), \
            '{} {}'.format(self.alphas_cumprod_prev.shape, self.num_timesteps)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_cumprod',
            th.sqrt(self.alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            th.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            'log_one_minus_alphas_cumprod',
            th.log(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod',
            th.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod',
            th.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_variance',
            (
                self.betas * (1.0 - self.alphas_cumprod_prev) /
                (1.0 - self.alphas_cumprod)
            )
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        if len(self.posterior_variance) > 1:
            self.register_buffer(
                'posterior_log_variance_clipped',
                th.log(
                    th.cat(
                        [self.posterior_variance[1].view(1),
                         self.posterior_variance[1:]]
                    )
                )
            )
        else:
            self.register_buffer(
                'posterior_log_variance_clipped',
                th.log(self.posterior_variance + 1e-5)
            )
        self.register_buffer(
            'posterior_mean_coef1',
            (
                self.betas * th.sqrt(self.alphas_cumprod_prev) /
                (1.0 - self.alphas_cumprod)
            )
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (
                (1.0 - self.alphas_cumprod_prev) * th.sqrt(alphas) /
                (1.0 - self.alphas_cumprod)
            )
        )

    def _get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps, log_scale):
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if schedule_name == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return th.linspace(
                beta_start, beta_end, num_diffusion_timesteps)

        elif schedule_name == "cosine_old":
            return self._betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        elif schedule_name == 'cosine':
            s = 0.008
            steps = num_diffusion_timesteps + 1
            x = th.linspace(0, num_diffusion_timesteps, steps)
            alphas_cumprod = th.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return th.clip(betas, 0.0001, 0.9999)

        elif schedule_name == 'quadratic':
            # scale = 1000 / num_diffusion_timesteps
            scale = 1
            beta_start = 0.0001 * scale
            beta_end = 0.02 * scale
            return th.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps) ** 2

        # elif schedule_name == 'quadratic_alphas':
        #     start_alpha = 0.9999
        #     end_alpha = 5e-5
        #     end_alpha = 1 - end_alpha
        #     start_alpha = 1 - start_alpha
        #     alphas_cumprod = -th.linspace((start_alpha ** 0.5), (end_alpha ** 0.5), num_diffusion_timesteps + 1) ** 2
        #     alphas_cumprod += 1
        #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        #     return th.clip(betas, 0.0001, 0.9999)

        elif schedule_name == 'log':
            start_alpha = th.tensor(0.9999)
            end_alpha = th.tensor(5e-7)
            log_scale = 4 if log_scale is None else log_scale
            alphas_cumprod = th.log(th.linspace(th.exp(start_alpha * log_scale), th.exp(end_alpha * log_scale), num_diffusion_timesteps + 1)) / log_scale
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas

        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    def _betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return th.tensor(betas)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = self.q_posterior_mean(x_start, x_t, t)
        posterior_var, posterior_log_var_clipped = self.q_posterior_variance(x_start, x_t, t)

        return posterior_mean, posterior_var, posterior_log_var_clipped

    def q_posterior_mean(self, x_start, x_t, t, **kwargs):
        assert x_start.shape == x_t.shape, f'{x_start.shape}, {x_t.shape}'
        # assert len(x_start.shape) == 3, x_start.shape
        # posterior_mean_coef1 = self.posterior_mean_coef1[t].unsqueeze(-1).unsqueeze(-1)
        # posterior_mean_coef1 = self.posterior_mean_coef2[t].unsqueeze(-1).unsqueeze(-1)
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # posterior_mean = (posterior_mean_coef1 * x_start) + (posterior_mean_coef1 * x_t)
        assert posterior_mean.shape[0] == x_start.shape[0]

        return posterior_mean

    def q_posterior_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        # assert len(x_start.shape) == 3, x_start.shape
        # posterior_variance = self.posterior_variance[t].unsqueeze(-1).unsqueeze(-1).expand(x_start.shape)
        # posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].unsqueeze(-1).unsqueeze(-1).expand(x_start.shape)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0])

        return posterior_variance, posterior_log_variance_clipped

class SpacedBetas(BetaScheduler):
    def __init__(self, schedule_name, num_diffusion_timesteps, section_counts=None, log_scale=None):
        super().__init__(schedule_name, num_diffusion_timesteps, log_scale=log_scale)

        new_timesteps = self.space_timesteps(num_diffusion_timesteps, section_counts)
        self.timestep_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in new_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        new_betas = th.tensor(new_betas)
        super().__init__(schedule_name, num_diffusion_timesteps, log_scale=log_scale, betas=new_betas)

    def space_timesteps(self, num_timesteps, section_counts=None):
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
        section_counts = [num_timesteps] if section_counts is None else section_counts

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

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape).type(th.float32)

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the th device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / th.sum(w)
        # import ipdb; ipdb.set_trace()
        # indices_np = np.random.choice(len(p), size=(batch_size,), p=p.detach().cpu())
        # indices = th.tensor(indices_np, device=device).type(th.long)
        # indices = th.from_numpy(indices_np).long().to(device)
        indices = th.multinomial(p, num_samples=batch_size, replacement=True)
        weights = 1 / (len(p) * p[indices])
        # weights = th.tensor(weights_np, device=device).type(th.float32)
        # weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler, pl.LightningModule):
    def __init__(self, num_timesteps):
        super().__init__()
        self.register_buffer(
            '_weights',
            th.ones([num_timesteps]).type(th.float64)
        )

    def weights(self):
        return self._weights

    def update_with_local_losses(self, *args, **kwargs):
        pass

    def update_with_all_losses(self, *args, **kwargs):
        pass


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler, pl.LightningModule):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        super().__init__()
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob

        self.register_buffer(
            '_loss_history',
            th.zeros((num_timesteps, history_per_term)).type(th.float64)
        )
        self.register_buffer(
            '_loss_counts',
            th.zeros(num_timesteps).type(th.int32)
        )
        self.num_timesteps = num_timesteps
        self._logged_warm_up = False

    def weights(self):
        if not self._warmed_up():
            return th.ones([self.num_timesteps], dtype=th.float64, device=self.device)
        elif self._logged_warm_up is False:
            logger.info('Sampler warmed up')
            self._logged_warm_up = True

        weights = th.sqrt(th.mean(self._loss_history ** 2, dim=-1))
        weights /= th.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        # import ipdb; ipdb.set_trace()
        self._loss_counts = self._loss_counts.to(losses[0].device)
        self._loss_history = self._loss_history.to(losses[0].device)
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t] = th.cat([self._loss_history[t, 1:], loss.view(1).detach()])

            else:
                self._loss_history[t, self._loss_counts[t]] = loss.detach()
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
