import copy
import torch
import pytorch_lightning as pl

def exists(val):
    return val is not None

def is_float_dtype(dtype):
    return any([dtype == float_dtype for float_dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16)])

def clamp(value, min_value = None, max_value = None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

class EMA(pl.LightningModule):
    """
    Implements exponential moving average shadowing for your model.
    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.
    @crowsonkb's notes on EMA Warmup:
    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        ema_model=None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        param_or_buffer_names_no_ema = set(),
        ignore_names = set(),
    ):
        super().__init__()
        self.beta = beta

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self, online_model):
        for ma_params, current_params in zip(list(self.ema_model.parameters()), list(online_model.parameters())):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(list(self.ema_model.buffers()), list(online_model.buffers())):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value = 0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value = self.min_value, max_value = self.beta)

    def update(self, online_model):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema(online_model)
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema(online_model)
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if hasattr(current_model.backbone.bb_predictor, 'noise_dct'):
            ma_model.backbone.bb_predictor.noise_dct = current_model.backbone.bb_predictor.noise_dct

        if hasattr(current_model.backbone.bb_predictor, 'ix_dct'):
            ma_model.backbone.bb_predictor.ix_dct = current_model.backbone.bb_predictor.ix_dct
        
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(list(current_model.named_parameters()), list(ma_model.named_parameters())):
            if name in self.ignore_names:
                continue

            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(list(current_model.named_buffers()), list(ma_model.named_buffers())):
            if name in self.ignore_names:
                continue

            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)