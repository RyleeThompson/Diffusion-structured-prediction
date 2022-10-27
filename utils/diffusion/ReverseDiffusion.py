from utils.diffusion.gaussian_diffusion import ReverseGaussianDiffusion
# import pytorch_lightning as pl
# import torch as th


def ReverseDiffusion(*args, **kwargs):
    return ReverseGaussianDiffusion(*args, **kwargs)

# class ReverseDiffusion(pl.LightningModule):
#     def __init__(self, beta_scheduler, model_mean_type, model_var_type, rescale_timesteps=False):
#         super().__init__()
#
#         self.reverse_gauss_diff = ReverseGaussianDiffusion(
#             beta_scheduler, model_mean_type, model_var_type, rescale_timesteps)
