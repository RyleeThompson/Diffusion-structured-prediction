# This file copied from https://github.com/openai/improved-diffusion

"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th
from functools import partial
import torch.nn as nn


class GDLossCalculator:
    def __init__(self, beta_scheduler, loss_type, model_var_type, model_mean_type):
        self.beta_scheduler = beta_scheduler
        self.model_var_type = model_var_type
        self._init_loss_fn(loss_type)
        self._init_target_fn(model_mean_type)

    def _init_loss_fn(self, loss_type):
        if loss_type == 'rescaled_kl':
            self.loss_fn = partial(self.kl_loss, rescale=True)
        elif loss_type == 'kl':
            self.loss_fn = partial(self.kl_loss, rescale=False)
        elif loss_type == 'rescaled_mse':
            self.loss_fn = partial(self.mse_loss, rescale=True)
        elif loss_type == 'mse':
            self.loss_fn = partial(self.mse_loss, rescale=False)
        elif loss_type == 'simple_regression':
            self.loss_fn = self.simple_mse_loss
        else:
            raise Exception('Unsupported loss ', loss_type)

    def _init_target_fn(self, model_mean_type):
        if model_mean_type == 'prev_x':
            self.target = self.beta_scheduler.q_posterior_mean
        elif model_mean_type == 'start_x':
            self.target = self.x_start_target
        elif model_mean_type == 'epsilon':
            self.target = self.eps_target

    def eps_target(self, noise, *args, **kwargs):
        return noise

    def x_start_target(self, x_start, *args, **kwargs):
        return x_start

    def __call__(self, model_out, x_start, x_t, t, noise, feat_key, **kwargs):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # import ipdb; ipdb.set_trace()
        x_t = x_t[feat_key]
        x_start = x_start[feat_key]
        # t = t.unsqueeze(-1).expand(x_t.shape[:-1])
        # t = t.flatten()
        # x_t = x_t.flatten(start_dim=0, end_dim=1)
        # x_start = x_start.flatten(start_dim=0, end_dim=1)
        # if feat_key == 'bbox':
        #     noise = noise['bb'].flatten(start_dim=0, end_dim=1)
        # else:
        #     noise = noise['cls'].flatten(start_dim=0, end_dim=1)

        noise = noise['cls'] if feat_key == 'classes_bits' else noise['bb']
        loss = self.loss_fn(model_out, x_start, x_t, t, noise=noise)
        return loss

    def kl_loss(self, model_out, x_start, x_t, t, rescale, **kwargs):
        terms = {}
        model_mean = model_out['mean']
        model_log_var = model_out['log_variance']
        terms["loss"] = vlb_terms_bpd(model_mean, model_log_var,
            x_start=x_start, x_t=x_t, t=t, beta_scheduler=self.beta_scheduler)

        terms['vb'] = terms['loss'].detach()
        if rescale:
            terms["loss"] *= self.beta_scheduler.num_timesteps

        return terms

    def mse_loss(self, model_out, x_start, x_t, t, noise, rescale):
        terms = {}
        model_mean = model_out['mean']
        model_log_var = model_out['log_variance']
        if self.model_var_type in ['learned', 'learned_range']:
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            terms["vb"] = vlb_terms_bpd(model_mean.detach(), model_log_var,
                x_start=x_start, x_t=x_t, t=t, beta_scheduler=self.beta_scheduler)

            assert model_log_var.shape == x_t.shape

        target = self.target(noise=noise, t=t, x_start=x_start, x_t=x_t)

        assert model_mean.shape == target.shape == x_start.shape, '{} {} {}'.format(model_mean.shape, target.shape, x_start.shape)
        # import ipdb; ipdb.set_trace()
        # terms["mse"] = mean_flat((target - model_mean) ** 2)
        terms['mse'] = nn.MSELoss(reduction='none')(model_mean, target)
        if self.model_var_type in ['learned', 'learned_range']:
            if rescale:
                # Divide by 1000 for equivalence with initial implementation.
                # Without a factor of 1/1000, the VB term hurts the MSE term.
                terms["loss"] = terms["mse"] + (terms["vb"] * self.beta_scheduler.num_timesteps / 1000.0)
            else:
                terms['loss'] = terms['mse'] + terms['vb']
            terms['mse'] = terms['mse'].detach()
            terms['vb'] = terms['vb'].detach()
        else:
            terms["loss"] = terms["mse"]
            terms['mse'] = terms['mse'].detach()

        return terms

    def simple_mse_loss(self, model_out, x_start, x_t, t, noise):
        terms = {}
        model_mean = model_out['mean']
        target = self.target(noise=noise, t=t, x_start=x_start, x_t=x_t)

        assert model_mean.shape == target.shape == x_start.shape
        # import ipdb; ipdb.set_trace()
        # terms['loss'] = mean_flat((target - model_mean) ** 2)
        terms['loss'] = nn.MSELoss(reduction='none')(model_mean, target)
        terms["mse"] = terms['loss'].detach()
        return terms

def vlb_terms_bpd(model_mean, model_log_var, x_start, x_t, t, beta_scheduler):
    """
    Get a term for the variational lower-bound.

    The resulting units are bits (rather than nats, as one might expect).
    This allows for comparison to other papers.

    :return: a dict with the following keys:
             - 'output': a shape [N] tensor of NLLs or KLs.
             - 'pred_xstart': the x_0 predictions.
    """
    true_mean, _, true_log_variance_clipped = beta_scheduler.q_posterior_mean_variance(
        x_start=x_start, x_t=x_t, t=t)

    kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_var)
    # kl = mean_flat(kl) / np.log(2.0)
    kl = kl / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        x_start, means=model_mean, log_std=0.5 * model_log_var)

    assert decoder_nll.shape == x_start.shape
    # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
    decoder_nll = decoder_nll / np.log(2.0)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

    # t = t.to(decoder_nll.device)
    t = t.to(decoder_nll.device)
    for _ in range(len(decoder_nll.shape) - 1):
        t = t.unsqueeze(-1)
    t = t.expand(decoder_nll.shape)
    output = th.where((t == 0), decoder_nll, kl)
    return output


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


# TODO: Investigate this. Need to change the inverse from [-1, 1] to be adaptable for BBs
def discretized_gaussian_log_likelihood(x, *, means, log_std):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_std.shape
    centered_x = x - means # Shift mean to zero
    inv_stdv = th.exp(-log_std) # Compute 1/std

    eps = 1.0 / 255
    plus_in = inv_stdv * (centered_x + eps) # Normalize & shift up by eps
    cdf_plus = approx_standard_normal_cdf(plus_in) # p(x < x_t + eps)
    min_in = inv_stdv * (centered_x - eps) # Normalize & shift down by eps
    cdf_min = approx_standard_normal_cdf(min_in) # p(x < x_t - eps)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12)) # log(p(x < x_t + eps))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12)) # log(p(x > x_t - eps))
    cdf_delta = cdf_plus - cdf_min # p(x_t - eps < x < x_t + eps)
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus, # log(p(x < x_t + eps)) (lower bound is -inf)
        th.where(
            x > 0.999,
            log_one_minus_cdf_min, # log(p(x > x_t - eps)) (upper bound is +inf)
            th.log(cdf_delta.clamp(min=1e-12))) # log(p(x_t - eps < x < x_t + eps))
    )
    assert log_probs.shape == x.shape
    return log_probs

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
