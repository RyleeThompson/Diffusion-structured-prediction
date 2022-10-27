import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial

# TODO: implement a MLP similar to gain_bias MLP but using a fc layer that takes in
# sinusoidal embeddings to obtain gains & biases


class TimeConditionedMLP(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim,
                 time_conditioning, time_encode_dim=None, num_timesteps=None,
                 relu_output=False, rescale_timesteps=False):
        super().__init__()

        init_fn = self.get_linear_init_fn(
            time_conditioning, num_timesteps, time_encode_dim,
            rescale_timesteps)
        self.num_layers = num_layers
        self.linears = nn.ModuleList()

        if num_layers > 1:
            self.linears.append(init_fn(in_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(init_fn(hidden_dim, hidden_dim))
            self.linears.append(init_fn(hidden_dim, out_dim))

        else:
            self.linears.append(init_fn(in_dim, out_dim))

        self.out_activation = F.relu if relu_output else self.identity

    def identity(self, x):
        return x

    def forward(self, x, t):
        for layer in self.linears[:-1]:
            x = layer(x, t)
            x = F.relu(x)
        return self.out_activation(self.linears[-1](x, t))

    def get_linear_init_fn(
        self, timestep_conditioning, num_timesteps,
        time_encode_dim, rescale_timesteps
    ):
        if timestep_conditioning == 'gain_bias':
            return partial(GainBiasLookupLinear, num_timesteps=num_timesteps)
        elif timestep_conditioning == 'none':
            return Linear
        elif timestep_conditioning == 'cat':
            return partial(
                TimeConcatLinear, time_encode_dim=time_encode_dim,
                rescale_timesteps=rescale_timesteps, num_timesteps=num_timesteps)
        elif timestep_conditioning == 'sin':
            return partial(
                GainBiasSinusoidalLinear, rescale_timesteps=rescale_timesteps,
                num_timesteps=num_timesteps)
        else:
            raise Exception(timestep_conditioning)


class TimeConcatLinear(nn.Module):
    def __init__(
        self, input_dim, output_dim, time_encode_dim, rescale_timesteps,
        num_timesteps
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim + time_encode_dim, output_dim)
        self.time_embedding = TimeStepEncoding(time_encode_dim)
        self.rescale_timesteps = rescale_timesteps
        self.num_timesteps = num_timesteps

    def forward(self, x, ts):
        if self.rescale_timesteps:
            ts = ts.float() * (1000.0 / self.num_timesteps)

        time_embed = self.time_embedding(ts)
        time_embed = expand_time_embed_shape(time_embed, x)

        x = th.cat([x, time_embed], dim=-1)
        return self.linear(x)


class GainBiasLookupLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_timesteps):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.gain = nn.Parameter(th.randn(num_timesteps, output_dim))
        self.bias = nn.Parameter(th.randn(num_timesteps, output_dim))

        input_dim = th.tensor(input_dim)
        bound = th.sqrt(1 / input_dim)
        nn.init.uniform_(self.gain, a=-bound, b=bound)
        nn.init.uniform_(self.bias, a=-bound, b=bound)

    def forward(self, x, ts):
        ts = ts.unsqueeze(1).expand(-1, x.shape[1])
        x = self.linear(x)
        x = (x * self.gain[ts]) + self.bias[ts]
        return x


class GainBiasSinusoidalLinear(nn.Module):
    def __init__(self, input_dim, output_dim, rescale_timesteps, num_timesteps):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        time_embed_size = max(input_dim, output_dim)
        self.time_embedding = TimeStepEncoding(time_embed_size)
        self.time_embed_size = self.time_embedding.channels
        self.gain_layer = nn.Linear(self.time_embed_size, output_dim)
        self.bias_layer = nn.Linear(self.time_embed_size, output_dim)
        self.rescale_timesteps = rescale_timesteps
        self.num_timesteps = num_timesteps

    def forward(self, x, ts):
        if self.rescale_timesteps:
            ts = ts.float() * (1000.0 / self.num_timesteps)
        x = self.linear(x)

        time_embed = self.time_embedding(ts)
        time_embed = expand_time_embed_shape(time_embed, x)

        gain = self.gain_layer(time_embed)
        bias = self.bias_layer(time_embed)
        x = (x * gain) + bias
        return x


class Linear(nn.Linear):
    def forward(self, x, *args, **kwargs):
        return super().forward(x)


def expand_time_embed_shape(time_embed, x):
    while len(time_embed.shape) != len(x.shape):
        time_embed = time_embed.unsqueeze(1)
    time_embed = time_embed.expand(*x.shape[:-1], -1)
    return time_embed

# Mostly borrowed from https://github.com/tatp22/multidim-positional-encoding, I just made a few
# changes to fit my use-case

class TimeStepEncoding(nn.Module):
    def __init__(self, out_dim):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(TimeStepEncoding, self).__init__()
        self.out_dim = out_dim
        self.channels = int(np.ceil(out_dim / 2) * 2)
        inv_freq = 1.0 / (10000 ** (th.arange(0, out_dim, 2).float() / out_dim))
        self.register_buffer("inv_freq", inv_freq)
        # self.cached_penc = None

    def forward(self, timesteps):
        batch_size = len(timesteps)
        pos_x = timesteps.type(self.inv_freq.type())

        sin_inp_x = th.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = th.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (th.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor, pos_x=None):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        if pos_x is None:
            pos_x = th.arange(x, device=tensor.device).type(self.inv_freq.type())

        sin_inp_x = th.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = th.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = th.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (th.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = th.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = th.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = th.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = th.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = th.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = th.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return th.flatten(emb, -2, -1)
