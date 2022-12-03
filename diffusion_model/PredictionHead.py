from diffusion_model.model_utils import TimeConditionedMLP
import torch.nn as nn
from utils.diffusion.gaussian_diffusion.GaussianDiffusionLosses import GDLossCalculator
from utils.diffusion.gaussian_diffusion.ReverseGaussianDiffusion import ReverseGaussianDiffusion
from utils.matcher import DiffusionHungarianMatcher
import pytorch_lightning as pl
from utils.data.coco_dataset import coco_num_classes
from utils import class_formatting as cls
import torch as th
from utils.data.BBox import BBox, create_bbox_like


class PredHeadJoiner(pl.LightningModule):
    def __init__(self, cfg, beta_scheduler, g_diffusion):
        super().__init__()
        bb_mean_type = cfg['diffusion']['model_mean_type_bb']
        self.bbox_pos_head = MLPPredHead(
            cfg, beta_scheduler, g_diffusion, bb_mean_type,
            out_dim=4)
        if cfg['model']['predict_class']:
            cls_mean_type = cfg['diffusion']['model_mean_type_bb']
            dataset = cfg['dataset']
            num_classes = cfg['max_num_blocks'] + 1 if dataset == 'tower' else coco_num_classes

            if cfg['model']['class_fmt'] == 'bits':
                out_dim = int(cls.num_classes2num_bits(num_classes))
            else:
                out_dim = num_classes
            self.cls_head = MLPPredHead(
                cfg, beta_scheduler, g_diffusion, cls_mean_type,
                out_dim=out_dim)
        self.g_diffusion = g_diffusion
        self.regress_from_gt = cfg['model']['regress_from'] == 'gt'
        self.bg_loss_mult = cfg['model']['bg_loss_mult'] if not self.regress_from_gt else 0

        if cfg['model']['use_matching']:
            self.matcher = DiffusionHungarianMatcher()
        else:
            self.matcher = self.no_matching

    def no_matching(self, preds, tgt, *args, **kwargs):
        return None, None


    def forward(self, x, batch, clip_denoised, inference=False):
        pos_head_pred = self.bbox_pos_head(x, batch, clip_denoised, 'bbox', inference=inference)
        if hasattr(self, 'cls_head'):
            cls_head_pred = self.cls_head(x, batch, clip_denoised, 'train_cls_fmt', inference=inference)
        else:
            cls_head_pred = None
        
        if inference is False:
            pos_start_pred = pos_head_pred['pred_xstart']
            cls_start_pred = cls_head_pred['pred_xstart']
            x_start_pred = create_bbox_like(pos_start_pred, cls_start_pred, batch['x_start']['padding_mask'], batch['x_start'], class_fmt=batch['x_start'].train_cls_fmt)
            pred_matched_ix, tgt_matched_ix = self.matcher(x_start_pred, batch['x_start'])

            if pred_matched_ix is not None:
                self.reindex_dct(pos_head_pred, pred_matched_ix)
                if cls_head_pred is not None:
                    self.reindex_dct(cls_head_pred, pred_matched_ix)
                self.reindex_batch(batch, tgt_matched_ix)

            pos_head_loss = self.bbox_pos_head.get_loss(pos_head_pred, batch, 'bbox')
            if cls_head_pred is not None:
                cls_head_loss = self.cls_head.get_loss(cls_head_pred, batch, 'train_cls_fmt')
            else:
                cls_head_loss = None

            return self.combine_losses(pos_head_loss, batch, cls_head_res=cls_head_loss)

        else:
            return self.combine_results(pos_head_pred, cls_head_res=cls_head_pred)

    def reindex_dct(self, dct, ixs):
        ixs = ixs.clone()
        for key, val in dct.items():
            dct[key] = self.reindex_tensor(val, ixs)
        return dct

    def reindex_batch(self, batch, ixs):
        batch['x_start'] = batch['x_start'].reindex(ixs)
        batch['x_t'] = batch['x_t'].reindex(ixs)
        batch['noise'] = self.reindex_dct(batch['noise'], ixs)
        return batch

    def reindex_tensor(self, tensor, ixs):
        if len(tensor.shape) == 3:
            temp_ixs = ixs.unsqueeze(-1)
            temp_ixs = temp_ixs.expand(-1, -1, tensor.shape[-1])
        else:
            temp_ixs = ixs
        tensor = tensor.gather(1, temp_ixs)
        return tensor

    def combine_losses(self, pos_head_res, batch, cls_head_res=None):
        loss_dct = {}
        padding_mask = batch['x_t']['padding_mask'].unsqueeze(-1)
        for key, val in pos_head_res.items():
            # pos_head_res[key] = val.view(batch['x_t'].shape[:-1]) * batch['x_t']['padding_mask']
            pos_head_res[key] = val * padding_mask  # Remove padding BBs from pos loss

        if cls_head_res is not None:
            for key, val in pos_head_res.items():
                # cls_head_res[key] = cls_head_res[key].view(batch['x_t'].shape[:-1])
                # bg_preds = cls_head_res[key] * th.logical_not(batch['x_t']['padding_mask'])
                # obj_preds = cls_head_res[key] * batch['x_t']['padding_mask']
                # total_obj_loss = obj_preds + self.bg_loss_mult * bg_preds
                # loss_dct[key] = pos_head_res[key] + total_obj_loss
                # loss_dct['bb_pos_' + key] = pos_head_res[key]
                # loss_dct['bb_cls_' + key] = total_obj_loss

                bg_loss = cls_head_res[key] * th.logical_not(padding_mask)
                obj_loss = cls_head_res[key] * padding_mask
                total_obj_loss = obj_loss + self.bg_loss_mult * bg_loss
                loss_dct[key] = th.cat([pos_head_res[key], total_obj_loss], dim=-1).mean(-1)
                loss_dct['bb_pos_' + key] = pos_head_res[key].mean(-1)
                loss_dct['bb_cls_' + key] = total_obj_loss.mean(-1)
        else:
            for key, val in pos_head_res.items():
                # loss_dct[key] = pos_head_res[key]
                # loss_dct['bb_pos_' + key] = pos_head_res[key]
                loss_dct[key] = pos_head_res[key].mean(-1)
                loss_dct['bb_pos_' + key] = pos_head_res[key].mean(-1)

        for key, val in loss_dct.items():
            loss_dct[key] = val.sum(-1)
            # loss_dct[key] = val.mean(-1)
        if self.training:
            self.g_diffusion.update_with_all_losses(batch['t'], loss_dct['loss'])
        for key, val in loss_dct.items():
            loss_dct[key] = (val * batch['weights']).mean()

        return loss_dct

    def combine_results(self, pos_head_res, cls_head_res=None):
        res = {}
        for key in pos_head_res.keys():
            if cls_head_res is not None:
                res[key] = th.cat([pos_head_res[key], cls_head_res[key]], dim=-1)
                res['cls_' + key] = cls_head_res[key]
            else:
                res[key] = pos_head_res[key]
            res['bb_' + key] = pos_head_res[key]
        return res


class MLPPredHead(pl.LightningModule):
    def __init__(self, cfg, beta_scheduler, g_diffusion, model_mean_type, out_dim):
        super().__init__()

        self.__init_mean_var_fns(cfg, out_dim)
        # self.__init_class_pred_fn(cfg)
        self.__init_loss_calculator(cfg, beta_scheduler)

        diff_cfg = cfg['diffusion']
        model_var_type = diff_cfg['model_var_type']
        predict_class = cfg['model']['predict_class']
        self.reverse_diffusion = ReverseGaussianDiffusion(
            beta_scheduler, model_mean_type, model_var_type,
            sampler=diff_cfg.get('sampler'), eta=diff_cfg.get('eta'),
            predict_class=predict_class, loss_type=diff_cfg['loss_type'],
            train_cls_fmt=cfg['model']['class_fmt'])
        self.g_diffusion = g_diffusion
        # if cfg['model']['use_matching']:
        #     self.matcher = DiffusionHungarianMatcher(cfg)
        # else:
        #     self.matcher = self.no_matching

    def __init_mean_var_fns(self, cfg, out_dim):
        pred_cfg = cfg['model']['structured']['mlp']
        in_dim = cfg['model']['structured']['d_model']
        diff_cfg = cfg['diffusion']
        model_var_type = diff_cfg['model_var_type']

        time_dim = self.__get_time_encode_dim(cfg)
        if model_var_type == 'learned' or model_var_type == 'learned_range':
            self.mean_fn = TimeConditionedMLP(
                pred_cfg['num_layers'], in_dim, in_dim, out_dim,
                pred_cfg['timestep_conditioning'], time_encode_dim=time_dim,
                num_timesteps=diff_cfg['num_timesteps']
            )
            self.var_fn = TimeConditionedMLP(
                pred_cfg['num_layers'], in_dim, in_dim, out_dim,
                pred_cfg['timestep_conditioning'], time_encode_dim=time_dim,
                num_timesteps=diff_cfg['num_timesteps']
            )

        elif model_var_type == 'fixed_small' or model_var_type == 'fixed_large':
            self.mean_fn = TimeConditionedMLP(
                pred_cfg['num_layers'], in_dim, in_dim, out_dim,
                pred_cfg['timestep_conditioning'], time_encode_dim=time_dim,
                num_timesteps=diff_cfg['num_timesteps']
            )
            self.var_fn = self.no_var_fn

    def no_var_fn(self, *args, **kwargs):
        return None

    def __init_class_pred_fn(self, cfg):
        self.class_fn = self.extract_gt_class

    def extract_gt_class(self, x, t, batch):
        return batch['classes']

    def __init_loss_calculator(self, cfg, beta_scheduler):
        diff_cfg = cfg['diffusion']
        loss_type = diff_cfg['loss_type']
        model_mean_type = diff_cfg['model_mean_type_bb']
        model_var_type = diff_cfg['model_var_type']
        self.loss_calculator = GDLossCalculator(
            beta_scheduler, loss_type=loss_type, model_var_type=model_var_type,
            model_mean_type=model_mean_type)

    def forward(self, x, batch, clip_denoised, feat_key, inference=False):
        t = batch['t']
        model_mean, model_var_out = self.mean_fn(x, t), self.var_fn(x, t)
        model_out = self.reverse_diffusion.extract_p_mean_variance(
            model_mean, model_var_out, x_t=batch['x_t'][feat_key], t=batch['t'],
            clip_denoised=clip_denoised)

        return model_out

        # if inference is False:
        #     return self.get_loss(model_out, batch, feat_key)
        # else:
        #     # for key, val in model_out.items():
        #     #     model_out[key] = val.flatten(start_dim=0, end_dim=1)
        #     return model_out

    def get_loss(self, model_out, batch, feat_key):
        # import ipdb; ipdb.set_trace()
        # for key, val in model_out.items():
        #     model_out[key] = val.flatten(start_dim=0, end_dim=1)
        loss_dct = self.loss_calculator(model_out, feat_key=feat_key, **batch)

        if loss_dct['loss'].isnan().any():
            raise RuntimeError('NaN loss')

        return loss_dct

    def __get_time_encode_dim(self, cfg):
        d_model = cfg['model']['structured']['d_model']
        if cfg['model']['structured']['name'] == 'transformer':
            time_dim = \
                d_model if cfg['model']['structured']['timestep_conditioning'] != 'cat' \
                else d_model // 2
        else:
            time_dim = d_model
        return time_dim
