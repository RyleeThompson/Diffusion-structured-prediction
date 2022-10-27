import torch as th
import pytorch_lightning as pl
from torch.optim import AdamW
from utils.diffusion.gaussian_diffusion.ReverseGaussianDiffusion import ReverseGaussianDiffusion
from diffusion_model.Backbones import get_backbone
from diffusion_model.PredictionHead import MLPPredHead, PredHeadJoiner
# from diffusion_model.struc_pred_models.BaseStrucPred import get_structured_pred_model
from diffusion_model.StructuredPrediction.Transformer import get_transformer
# from diffusion_model.StructuredPrediction.MLP import StructuredDiffusionMLP
from copy import deepcopy
import time
from utils.data.BBox import BBox, create_bbox_like
from utils.matcher import DiffusionHungarianMatcher
import os
import shutil


def get_diffusion_model(cfg, beta_scheduler, g_diffusion, bbox_transformer):
    backbone = get_backbone(cfg)
    struc_pred_model = get_transformer(cfg)
    pred_head = PredHeadJoiner(cfg, beta_scheduler, g_diffusion)

    return StructuredDiffusionModel(
        cfg, backbone, struc_pred_model, pred_head,
        beta_scheduler, g_diffusion, bbox_transformer)


def get_structured_pred_model(cfg):
    if 'transformer' in cfg['model']['structured']['name']:
        return get_transformer(cfg)
    else:
        return StructuredDiffusionMLP(
            cfg, cfg['diffusion']['num_timesteps'],
            **cfg['model']['structured'])


class StructuredDiffusionModel(pl.LightningModule):
    def __init__(self, cfg, backbone, struc_pred_model, pred_head,
                 beta_scheduler, g_diffusion, bbox_transformer):
        super().__init__()

        self.backbone = backbone
        self.structured = struc_pred_model
        self.pred_head = pred_head

        self.g_diffusion = g_diffusion
        self.bbox_transformer = bbox_transformer
        self.predict_class = cfg['model']['predict_class']

        diff_cfg = cfg['diffusion']
        model_mean_type = diff_cfg['model_mean_type_bb']
        model_var_type = diff_cfg['model_var_type']
        self.reverse_diffusion = ReverseGaussianDiffusion(
            beta_scheduler, model_mean_type, model_var_type,
            sampler=diff_cfg.get('sampler'), eta=diff_cfg.get('eta'),
            predict_class=self.predict_class, loss_type=diff_cfg['loss_type'])
        self.opt_cfg = cfg['optimizer']
        self.max_num_preds = cfg['model']['max_num_preds'] if cfg['dataset'] == 'coco' else cfg['max_num_blocks']
        self.init_ema(cfg['model']['ema_rate'])
        self.regress_from_gt = cfg['model']['regress_from'] == 'gt'

        self.match_input = cfg['model']['structured']['conditioning']['match_input']
        if self.match_input:
            self.matcher = DiffusionHungarianMatcher(cost_class=1)
            self.match_ix_subdir = 'dataset/coco2017-1.1.0/match_ixs'
            if os.path.exists(self.match_ix_subdir):
                shutil.rmtree(self.match_ix_subdir)
            os.makedirs(self.match_ix_subdir)

    def forward(self, batch, clip_denoised=True, inference=False, bbone_preds=None):
        if not self.predict_class:
            batch['x_t'].classes = batch['x_start'].classes
            self.batch_size = batch['x_t']['padding_mask'].sum()
        else:
            self.batch_size = batch['x_t'].shape[0] * batch['x_t'].shape[1]

        real_x_start = batch['x_start']['bbox'][batch['x_start']['padding_mask']]
        if (real_x_start > (1 + 1e-5)).any():
            max = real_x_start.max()
            batch['x_start'].inverse_normalize()
            maxs = real_x_start.max(0).values.max(0).values
            mins = real_x_start.min(0).values.min(0).values
            raise Exception('GT out of [-1, 1]', maxs, mins, max)
        if (real_x_start < (-1 - 1e-5)).any():
            min = real_x_start.min()
            batch['x_start'].inverse_normalize()
            maxs = real_x_start.max(0).values.max(0).values
            mins = real_x_start.min(0).values.min(0).values
            raise Exception('GT out of [-1, 1]', maxs, mins, min)

        # batch['x_start'].inverse_normalize()
        # mins = batch['x_start']['bbox'].min(0).values.min(0).values
        # maxs = batch['x_start']['bbox'].max(0).values.max(0).values
        # self.maxs = self.maxs.to(mins.device)
        # self.mins = self.mins.to(mins.device)
        # if (mins < self.mins).any():
        #     self.mins[mins < self.mins] = mins[mins < self.mins]
        #     print('new mins', self.mins)
        # if (maxs > self.maxs).any():
        #     self.maxs[maxs > self.maxs] = maxs[maxs > self.maxs]
        #     print('new maxs', self.maxs)
        # return None, None
        # batch['x_start'].normalize()

        bbone_preds = self.backbone(batch, bbone_preds)
        batch['bbone_bb_preds'] = bbone_preds.get('bbone_preds')
        if self.match_input and not inference:
            pred_ixs, tgt_ixs = self.match_fn(batch)
            if bbone_preds['x_t_feats'] is not None:
                batch['bbone_bb_preds']['feats'] = bbone_preds['bbone_pred_feats']
                batch['x_t']['feats'] = bbone_preds['x_t_feats']
            batch['bbone_bb_preds'] = batch['bbone_bb_preds'].reindex(pred_ixs)
            batch['x_start'] = batch['x_start'].reindex(tgt_ixs)
            batch['x_t'] = batch['x_t'].to(self.device)
            batch['x_t'] = batch['x_t'].reindex(tgt_ixs)

            if bbone_preds['x_t_feats'] is not None:
                bbone_preds['bbone_pred_feats'] = batch['bbone_bb_preds']['feats']
                bbone_preds['x_t_feats'] = batch['x_t']['feats']

        x_t_struc_feats = self.structured(batch['x_t'], batch['t'], bbone_preds)
        model_out = self.pred_head(x_t_struc_feats, batch, clip_denoised, inference=inference)

        return model_out, bbone_preds

    def training_step(self, batch, batch_ix):
        """
            Perform a training step on the given batch
        """
        # import ipdb; ipdb.set_trace()
        batch = self.prepare_training_batch(batch)
        model_out, _ = self.forward(batch, clip_denoised=False)
        # return {'loss': th.tensor([0])}
        return model_out

    def prepare_training_batch(self, batch):
        batch = self.g_diffusion.sample_training_noise(batch)
        # self['x_start'] = self['x_start'].flatten(start_dim=0, end_dim=1)
        # self['noise'] = self['noise'].flatten(start_dim=0, end_dim=1)
        # x_start = batch['x_start']
        # batch['x_t'] = create_bbox_like(
        #     batch['x_t_bb'], batch['x_t_cls'], x_start['padding_mask'],
        #     bbox_like=x_start, class_fmt='bits')
        return batch

    def init_ema(self, ema_rate):
        for name, param in self.named_parameters():
            name = name.replace('.', '-')
            if param.requires_grad:
                self.register_buffer('ema_param_' + name, th.clone(param.detach()))
        self.ema_rate = ema_rate

    def match_fn(self, batch):
        pred_ixs = []
        tgt_ixs = []
        pred_ixs, tgt_ixs = self.matcher(batch['bbone_bb_preds'], batch['x_start'])
        return pred_ixs, tgt_ixs

        # for path in batch['path']:
        #     path = os.path.join(self.match_ix_subdir, path)
        #     # if not os.path.exists(path):
        #         pred_ixs, tgt_ixs = self.matcher(batch['bbone_bb_preds'], batch['x_start'])
        #         for ix, path in enumerate(batch['path']):
        #             path = os.path.join(self.match_ix_subdir, path)
        #             th.save({'pred_ixs': pred_ixs[ix],
        #                      'tgt_ixs': tgt_ixs[ix]}, path)
        #         return pred_ixs, tgt_ixs
        #
        #     else:
        #         loaded = th.load(path, map_location=self.device)
        #         pred_ixs.append(loaded['pred_ixs'])
        #         tgt_ixs.append(loaded['tgt_ixs'])
        # return th.stack(pred_ixs).to(self.device), th.stack(tgt_ixs).to(self.device)

    def inference_loader(self, loader, num_batches=1, clip_denoised=True,
                  num_steps_to_ret=10):
        # final_bboxes = []
        # x_t_bboxes = []
        # intermediate_steps = []
        # images = []
        # x_start = []
        # model_conditioning = []
        # classes = []
        # for batch in range(num_batches):
        # if num_batches > 1:
            # raise Exception()
        # if loader is not None:
        # else:
            # batch = self.batch.to(self.device)
        # batch = batch.to(self.device)
        # batch = self.prepare_batch_from_loader(loader)
        for batch_ix in range(num_batches):
            batch = next(iter(loader)).to(self.device)
            result = self.inference_batch(
                batch, clip_denoised=clip_denoised,
                num_steps_to_ret=num_steps_to_ret)

            if result['bbone_preds']['bbone_preds'] is not None:
                bbone_bbs = result['bbone_preds']['bbone_preds']
                bbone_bbs = bbone_bbs.inverse_normalize().to_xywh().unmask(use_cls=bbone_bbs.unmask_with_cls)
                result['bbone_preds'] = bbone_bbs['bbox']
                result['bbone_preds_cls'] = bbone_bbs['classes_softmax']
                result['bbone_scores'] = bbone_bbs['scores']
                result['bbone_cls_int'] = bbone_bbs['classes']
            else:
                del result['bbone_preds']

                # bbone_mask = result['bbone_preds']['bbone_preds']['padding_mask']
                # bbone_bbs = self.extract_unmasked_preds(bbone_bbs, bbone_mask)
                # model_conditioning += bbone_bbs
            result['images'] = [sample['image'] for sample in batch['image_info']]

            if batch_ix == 0:
                all_res = result
            else:
                for key, val in result.items():
                    if key in ['all_x_t', 'ts', 'x_t_cls_preds', 'x_t_cls_int_preds']:
                        for t in range(len(all_res[key])):
                            all_res[key][t] += val[t]
                    # elif key == 'bbone_preds':
                    #     continue
                    else:
                        # try:
                        all_res[key] += val
                        # except:
                        #     import ipdb; ipdb.set_trace()
                        #     pass


        # import ipdb; ipdb.set_trace()
        # pass
        return all_res
        # final_bboxes += result['final_bboxes']
        # x_t_bboxes += result['x_t_bboxes']
        # intermediate_steps += result['ts']
        # images += [sample['image'] for sample in batch['image_info']]
        # # images += batch['original_image']
        # x_start += result['x_start']
        # classes += result['class_preds']

        # return {'final_bboxes': final_bboxes,
        #         'all_x_t': x_t_bboxes,
        #         'ts': intermediate_steps,
        #         'images': images,
        #         'x_start': x_start,
        #         'bbone_preds': model_conditioning,
        #         'class_preds': classes}
        # return result

    # def prepare_batch_from_loader(self, loader):
    #     batch = next(iter(loader))
    #     for key, val in batch.items():
    #         try:
    #             batch[key] = val.to(self.device)
    #         except AttributeError:
    #             pass
    #     return batch

    def inference_batch(self, batch, clip_denoised=True, num_steps_to_ret=10):
        x_t_bb = th.randn_like(batch['x_start']['bbox'])
        x_t_cls = th.randn_like(batch['x_start']['classes_bits'])
        batch['x_t'] = create_bbox_like(
            x_t_bb, x_t_cls, batch['x_start']['padding_mask'],
            bbox_like=batch['x_start'], class_fmt='bits')

        batch_res = self.reverse_diffusion.sample(
            self, batch, num_steps_to_ret=num_steps_to_ret,
            clip_denoised=clip_denoised)

        # import ipdb; ipdb.set_trace()
        batch_final_res = {}
        use_pred_cls = not self.regress_from_gt
        all_x_t_unmasked = [bb.inverse_normalize().to_xywh().unmask(use_cls=use_pred_cls) for bb in batch_res['all_x_t']]
        batch_final_res['all_x_t'] = [bb['bbox'] for bb in all_x_t_unmasked]
        batch_final_res['ts'] = batch_res['steps_to_return']

        final_out_unmasked = batch_res['final_out'].inverse_normalize().to_xywh().unmask(use_cls=use_pred_cls)
        batch_final_res['final_bboxes'] = final_out_unmasked['bbox']
        x_start_unmasked = batch['x_start'].inverse_normalize().to_xywh().unmask()
        if not self.predict_class:
            batch_final_res['x_t_cls_preds'] = [x_start_unmasked['classes_softmax'] for _ in all_x_t_unmasked]
            batch_final_res['x_t_cls_int_preds'] = [x_start_unmasked['classes'] for _ in all_x_t_unmasked]
            batch_final_res['final_cls_preds'] = x_start_unmasked['classes_softmax']
            batch_final_res['final_cls_int_preds'] = x_start_unmasked['classes']
            batch_final_res['final_scores_preds'] = x_start_unmasked['scores']
        else:
            batch_final_res['x_t_cls_preds'] = [bb['classes_softmax'] for bb in all_x_t_unmasked]
            batch_final_res['x_t_cls_int_preds'] = [bb['classes'] for bb in all_x_t_unmasked]
            batch_final_res['final_cls_preds'] = final_out_unmasked['classes_softmax']
            batch_final_res['final_cls_int_preds'] = final_out_unmasked['classes']
            batch_final_res['final_scores_preds'] = final_out_unmasked['scores']

        batch_final_res['x_start'] = x_start_unmasked['bbox']
        batch_final_res['x_start_cls'] = x_start_unmasked['classes_softmax']
        batch_final_res['x_start_cls_int'] = x_start_unmasked['classes']
        batch_final_res['bbone_preds'] = batch_res['bbone_preds']

        return batch_final_res

    def postprocess_preds(self, num_bbs, final_out, all_x_t,
                          steps_to_return, class_preds, **kwargs):
        final_bboxes = self.extract_unmasked_preds(final_out, num_bbs)
        class_preds = self.extract_unmasked_preds(class_preds, num_bbs)

        all_x_t_bboxes = []
        for x_t in all_x_t:
            x_t_bboxes = self.extract_unmasked_preds(x_t, num_bbs)
            all_x_t_bboxes.append(x_t_bboxes)

        return {'final_bboxes': final_bboxes,
                'x_t_bboxes': all_x_t_bboxes,
                'ts': steps_to_return,
                'class_preds': class_preds}

    def extract_unmasked_preds(self, preds, num_bbs):
        if isinstance(num_bbs, th.Tensor):
            max_num_bbs = num_bbs.shape[1]
            vals, num_bbs = num_bbs.max(dim=-1)
            num_bbs[vals == False] = max_num_bbs

        result = []
        for ix, bb_count in enumerate(num_bbs):
            unmasked_preds = preds[ix, :bb_count]
            if unmasked_preds.shape[-1] == 4:
                unmasked_preds = self.bbox_transformer.inverse_bbox_normalization(unmasked_preds)
            result.append(unmasked_preds)
        return result

    def training_epoch_end(self, outputs):
        self.training_outputs = outputs  # So we can access training info in callbacks
        for name, model_param in self.named_parameters():
            name = name.replace('.', '-')
            if model_param.requires_grad:
                ema_param = getattr(self, 'ema_param_' + name)
                ema_param.detach().mul_(self.ema_rate).add_(model_param, alpha=1 - self.ema_rate)
                setattr(self, 'ema_param_' + name, ema_param)

    def validation_epoch_end(self, outputs):
        self.validation_outputs = outputs
        # self.log('valid_map_50', self.valid_map_50, rank_zero_only=True, batch_size=1)
        # self.log('valid_rmse', self.valid_rmse, rank_zero_only=True, batch_size=1)

    def validation_step(self, batch, batch_ix):
        val_loss = self.training_step(batch, batch_ix)
        self.log('valid_loss', val_loss['loss'], batch_size=self.batch_size)
        return val_loss

    # The rest of this is initializing the optimizer
    def configure_optimizers(self):
        cfg = self.opt_cfg
        opt = self.init_opt(cfg)
        scheduler = self.init_scheduler(opt, cfg['structured'])

        return [opt], [scheduler]

    def init_opt(self, cfg):
        structured_opt = cfg['structured']

        opt_dcts = []
        opt_name = None
        for head_name, head_cfg in cfg.items():
            head = getattr(self, head_name)
            params = head.parameters()

            if head_cfg['name'] == 'same':# and head_name != 'backbone':
                opt_dcts.append(
                    self.create_opt_dct(
                        params, structured_opt
                    )
                )

            elif head_cfg['name'] == 'frozen':# or head_name == 'backbone':
                # if head_cfg['name'] != 'frozen':
                    # print(f'Warning: freezing weights for {head_name} (unfrozen not yet supported)' + ' -' * 50)
                for param in params:
                    param.requires_grad = False
                head.eval()

            else:
                if opt_name is None:
                    opt_name = head_cfg['name']
                assert opt_name == head_cfg['name'],\
                    'got different optimizers {} {}'.format(
                    opt_name, head_cfg['name'])

                opt_dcts.append(
                    self.create_opt_dct(
                        params, head_cfg)
                )

        opt = self.create_optimizer(opt_dcts, opt_name)
        return opt

    def create_optimizer(self, opt_dcts, opt_name):
        if opt_name == 'AdamW':
            opt = AdamW(opt_dcts)
        else:
            raise Exception(opt_name)
        return opt

    def create_opt_dct(self, params, opt_cfg):
        dct = {
            key: val for key, val in opt_cfg.items()
            if key != 'name' and key != 'lr_scheduler'
        }
        dct.update({'params': params})
        return dct

    def init_scheduler(self, opt, cfg):
        cfg = cfg['lr_scheduler']
        if cfg['name'] == 'step':
            scheduler = th.optim.lr_scheduler.StepLR(
                opt, step_size=cfg['step_size'], gamma=cfg['gamma'])
        else:
            raise Exception(cfg['name'])

        return scheduler

    def train(self, train=True):
        if train:
            for head, head_cfg in self.opt_cfg.items():
                head = getattr(self, head)
                if head_cfg['name'] != 'frozen':
                    head.train()
                else:
                    head.eval()
        else:
            for head, head_cfg in self.opt_cfg.items():
                head = getattr(self, head)
                head.eval()
