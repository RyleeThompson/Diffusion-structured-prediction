import torch as th
from utils.diffusion.gaussian_diffusion.GaussianDiffusionLosses import normal_kl, vlb_terms_bpd
import logging
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import copy

logger = logging.getLogger('my_task')

class NLLEvaluation():
    def __init__(self, cfg, set, data_loader):
        self.data_loader = data_loader
        self.set = set
        predict_class = cfg['model']['predict_class']
        if predict_class:
            self.x_t_cls_fmt = 'bits'
        else:
            self.x_t_cls_fmt = None

    def kl_with_standard_normal(self, x_start, t, beta_scheduler):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([t] * batch_size, device=x_start.device)
        # t = th.tensor([beta_scheduler.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = beta_scheduler.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return kl_prior / np.log(2.0)

    @th.no_grad()
    def evaluate(self, model, save_dir, clip_denoised=True):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        logger.info('Evaluating NLL')
        if self.data_loader is not None:
            batch = next(iter(self.data_loader)).to(model.device)
        else:
            batch = model.batch.to(model.device)
        logger.info('loaded nll batch')

        # for key, val in batch.items():
        #     if isinstance(val, th.Tensor):
        #         batch[key] = val.to(model.device)
        #     elif isinstance(val, dict):
        #         batch[key] = {}
        #         for subkey, subval in val.items():
        #             batch[key][subkey] = subval.to(model.device)
        #     elif isinstance(val, list):
        #         batch[key] = val

        batch = model.prepare_training_batch(batch)

        g_diffusion = model.g_diffusion
        beta_scheduler = g_diffusion.beta_scheduler
        reverse_diffusion = model.reverse_diffusion

        vb = []
        xstart_mse = []
        eps_mse = []
        kl_prior = []

        indices = list(range(beta_scheduler.num_timesteps))[::-1]#[: 2]
        # indices = tqdm(indices)
        bbone_preds = None
        for t in indices:
            vlb_, xstart_mse_, eps_mse_, bbone_preds = self.perform_iter(
                model, batch, t, g_diffusion, beta_scheduler,
                reverse_diffusion, clip_denoised, bbone_preds)

            # kl_prior_ = self.kl_with_standard_normal(
            #     batch['x_start'].get_features(self.x_t_cls_fmt).flatten(0, 1), t, beta_scheduler).unsqueeze(-1)
            kl_prior_ = self.kl_with_standard_normal(
                batch['x_start'].get_features(self.x_t_cls_fmt), t, beta_scheduler).mean(-1).mean(-1)

            vb.append(vlb_)
            xstart_mse.append(xstart_mse_)
            eps_mse.append(eps_mse_)
            kl_prior.append(kl_prior_)

        # import ipdb; ipdb.set_trace()
        # del model.cached_bbone_res
        # res_aggregated, res_total = self.combine_results(
        #     vb, xstart_mse, eps_mse, kl_prior, batch, beta_scheduler)
        combined_results = self.combine_results(
            vb, xstart_mse, eps_mse, kl_prior, batch, beta_scheduler)

        keep_masks = batch['x_start']['padding_mask'].unsqueeze(-1)
        assert len(keep_masks.shape) != 2

        # import ipdb; ipdb.set_trace()
        # res_aggregated = self.aggregate_results(model, res_aggregated, keep_masks)
        # res_total = self.get_all_results(model, res_total, keep_masks)

        self.plot_results(combined_results, save_dir)

        # for key, val in combined_results.items():
            # combined_results[key] = val.mean(-1)
        # return res_aggregated]# res_aggregated = {
        #     "total_bpd": total_bpd,
        #     "prior_bpd": prior_bpd,
        #     "xstart_mse": xstart_mse,
        #     "total_mse": eps_mse}
        return {'total_bpd': combined_results['total_bpd'].mean(),
                'prior_bpd': combined_results['prior_bpd'].mean(),
                'xstart_mse': combined_results['xstart_mse'].sum(-1).mean(),
                'eps_mse': combined_results['eps_mse'].sum(-1).mean()}

    def perform_iter(
        self, model, batch, t, g_diffusion, beta_scheduler, reverse_diffusion,
        clip_denoised, bbone_preds
    ):
        x_start = batch['x_start']
        batch = self.update_batch(batch, t, x_start, g_diffusion)

        # Calculate VLB term at the current timestep
        model_out, bbone_preds = model(
            batch, bbone_preds=bbone_preds, clip_denoised=clip_denoised, inference=True)

        # x_start_reshape = batch['x_start'].get_features(self.x_t_cls_fmt).flatten(0, 1)
        # x_t_reshape = batch['x_t'].get_features(self.x_t_cls_fmt).view(x_start_reshape.shape)
        # t_reshape = batch['t'].unsqueeze(-1)
        # t_reshape = t_reshape.expand(batch['x_t'].shape[:-1]).flatten()
        # vlb = vlb_terms_bpd(
        #     model_out['mean'], model_out['log_variance'],
        #     x_start=x_start_reshape, x_t=x_t_reshape, t=t_reshape,
        #     beta_scheduler=beta_scheduler)

        vlb_bb = vlb_terms_bpd(
            model_out['bb_mean'], model_out['bb_log_variance'],
            x_start=batch['x_start']['bbox'], x_t=batch['x_t']['bbox'], t=batch['t'],
            beta_scheduler=beta_scheduler)
        padding_mask = th.logical_not(batch['x_start']['padding_mask'])
        vlb_bb[padding_mask] = 0

        bb_xstart_mse = (model_out['bb_pred_xstart'] - batch['x_start']['bbox']) ** 2
        bb_xstart_mse[padding_mask] = 0

        bb_eps = reverse_diffusion._predict_eps_from_xstart(
            model_out['bb_pred_xstart'], batch['x_t']['bbox'], t)
        bb_eps_mse = (bb_eps - batch['noise'][..., :4]) ** 2
        bb_eps_mse[padding_mask] = 0

        if 'cls_mean' in model_out:
            vlb_cls = vlb_terms_bpd(
                model_out['cls_mean'], model_out['cls_log_variance'],
                x_start=batch['x_start']['classes_bits'], x_t=batch['x_t']['classes_bits'], t=batch['t'],
                beta_scheduler=beta_scheduler)
            vlb = th.cat([vlb_bb, vlb_cls], dim=-1)

            cls_xstart_mse = (model_out['cls_pred_xstart'] - batch['x_start']['classes_bits']) ** 2
            xstart_mse = th.cat([bb_xstart_mse, cls_xstart_mse], dim=-1)

            cls_eps = reverse_diffusion._predict_eps_from_xstart(
                model_out['cls_pred_xstart'], batch['x_t']['classes_bits'], t)
            cls_eps_mse = (cls_eps - batch['noise'][..., 4:]) ** 2
            eps_mse = th.cat([bb_eps_mse, cls_eps_mse], dim=-1)

        else:
            vlb = vlb_bb
            xstart_mse = bb_xstart_mse
            eps_mse = bb_eps_mse

        vlb = vlb.mean(-1).mean(-1)
        xstart_mse = xstart_mse.mean(-1).mean(-1)
        eps_mse = eps_mse.mean(-1).mean(-1)

        return vlb.detach(), xstart_mse.detach(), eps_mse.detach(), bbone_preds

    def update_batch(self, batch, t, x_start, g_diffusion):
        batch_size = batch['x_t'].shape[0]
        batch['t'] = th.tensor([t], device=x_start.device).repeat(batch_size).type(th.long)

        # x_start = x_start.view(batch['x_t'].shape)
        x_t_info = g_diffusion.sample(
            x_start=x_start, ts=batch['t'])
        batch['x_t'] = x_t_info['x_t']
        if self.x_t_cls_fmt is None:
            batch['noise'] = x_t_info['noise']['bb']
        else:
            batch['noise'] = x_t_info['noise']['noise']
        return batch

    def combine_results(self, vb, xstart_mse, eps_mse, kl_prior, batch, beta_scheduler):
        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        eps_mse = th.stack(eps_mse, dim=1)
        kl_prior = th.stack(kl_prior, dim=1)

        prior_bpd = self.kl_with_standard_normal(batch['x_start'].get_features(self.x_t_cls_fmt), beta_scheduler.num_timesteps - 1, beta_scheduler).mean(-1).mean(-1)
        total_bpd = vb.sum(dim=1) + prior_bpd

        return {
            'total_bpd': total_bpd,
            'prior_bpd': prior_bpd,
            'all_vlb': vb,
            'xstart_mse': xstart_mse,
            'eps_mse': eps_mse,
            'kl_prior': kl_prior
        }
        # res_aggregated = {
        #     "total_bpd": total_bpd,
        #     "prior_bpd": prior_bpd,
        #     "xstart_mse": xstart_mse,
        #     "total_mse": eps_mse}
        # res_total = {
        #     "total_vb": vb,
        #     "xstart_mse": xstart_mse,
        #     "total_mse": eps_mse,
        #     'kl_prior': kl_prior}

        # return res_aggregated, res_total

    def aggregate_results(self, model, res_aggregated, keep_masks):
        for key, val in res_aggregated.items():
            # temp = val.view(val.shape[0] // model.max_num_preds, model.max_num_preds, -1)
            # temp = temp * keep_masks # Mask out padding BBs
            # temp = temp.sum(dim=-1).sum(dim=-1) # Sum up nlls for BBs of single example
            # temp /= keep_masks.sum(dim=-1).sum(dim=-1).clamp(min=1) # Divide by # BBs in example (mean NLL for each image)
            # res_aggregated[key] = temp.mean().detach() # Get mean across batch
            res_aggregated[key] = val.mean(-1)
        return res_aggregated

    def get_all_results(self, model, res_total, keep_masks):
        for key, val in res_total.items():
            temp = val.view(val.shape[0] // model.max_num_preds, model.max_num_preds, -1)
            temp = temp * keep_masks # Mask out padding BBs
            res_total[key] = temp
        return res_total

    def plot_results(self, res_total, save_dir):
        fig, axs = plt.subplots(1, 4, figsize=(22, 4))
        titles = ['VLB terms', 'X start MSE', 'Eps. MSE', 'KL w/ standard normal']
        keys = ['all_vlb', 'xstart_mse', 'eps_mse', 'kl_prior']
        # import ipdb; ipdb.set_trace()
        for key, ax, title in zip(keys, axs, titles):
            # Val is num_samples x num_bbs x num_timesteps
            val = res_total[key]
            # total = val.sum(dim=1)  # Sum up value for all BBs in each image
            # mean = total.mean(dim=0).cpu()  # Mean across samples
            # std = total.std(dim=0).cpu()  # Std. across samples
            mean = val.mean(dim=0).cpu()  # Mean across samples
            std = val.std(dim=0).cpu()  # Std. across samples
            x = th.arange(len(mean)) / len(mean)

            ax.plot(x, mean)
            ax.fill_between(x, mean - std, mean + std, color='b', alpha=0.1)

            x_tick_loc = ax.get_xticks().tolist()
            ax.set_xticks(x_tick_loc)
            ax.set_xticklabels(['{:.2f}'.format(label) for label in reversed(x_tick_loc)])

            ax.set_xlabel('$t \; / \; T \;\; (T = {})$'.format(len(mean)))
            ax.set_title(title)
            ax.set_yscale('log')
        plt.savefig(save_dir + f'/{self.set}_nll_eval.jpg', bbox_inches='tight')


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
