import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from typing import Callable, Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, Instances, BoxMode
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, META_ARCH_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads, FastRCNNOutputLayers
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
# from detectron2.structures.boxes import , Boxes
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import os
import time
import copy
from utils.diffusion.gaussian_diffusion.ForwardGaussianDiffusion import GaussianNoiser, BetaScheduler
import logging
import math
from utils.data.BBNormalizer import create_bbox_transformer
import numpy as np
from utils.data.coco_dataset import coco_max_num_bboxes
from torchvision.ops import boxes as box_ops
import shutil
from utils.data.BBox import BBox, create_bbox_like
from utils.data.coco_dataset import coco_num_classes, coco_bg_class


logger = logging.getLogger('my_task')


def get_backbone(cfg):
    img_feat_extractor, det2_model = init_img_feat_extractor(cfg)
    bb_predictor = init_bb_predictor(cfg, det2_model)

    return Joiner(cfg, img_feat_extractor, bb_predictor)


def init_img_feat_extractor(cfg):
    det2_model = None
    conditioning = cfg['model']['structured']['conditioning']['name']
    backbone_name = cfg['model']['backbone']['name']
    if conditioning in ['x_t', 'bb_preds']: #and backbone_name != 'detectron2':
        img_feat_extractor = DummyImageFeatureExtractor()
    elif conditioning in ['h(x)', 'both']:# or backbone_name == 'detectron2':
        bbone_name = cfg['model']['backbone']['model']
        det2_model, _ = init_detectron2_model(det2_model=bbone_name)
        img_feat_extractor = Detectron2FeatExtractor(cfg, det2_model)
    return img_feat_extractor, det2_model


def init_bb_predictor(cfg, det2_model):
    if cfg['model']['backbone']['name'] == 'none' or cfg['model']['structured']['conditioning']['name'] == 'h(x)':
        bb_predictor = DummyBBPredictor()

    elif cfg['model']['backbone']['name'] == 'gt':
        bb_predictor = GTruthBBPredictor(cfg)

    else:
        bbone_name = cfg['model']['backbone']['model']
        top_k = cfg['model']['structured']['conditioning']['top_k']
        if det2_model is None:
            det2_model, _ = init_detectron2_model(det2_model=bbone_name)
        if cfg['optimizer']['backbone']['name'] == 'frozen':
            bb_predictor = FrozenDetectron2BBPredictor(cfg, det2_model, bbone_name, top_k=top_k)
        elif cfg['optimizer']['backbone']['name'] != 'frozen':
            bb_predictor = Detectron2BBPredictor(cfg, det2_model, top_k=top_k)

    return bb_predictor


def init_detectron2_model(det2_model='faster_rcnn_X_101_32x8d_FPN_3x'):
    bbone_cfg = get_cfg()
    bbone_cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{det2_model}.yaml"))
    bbone_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{det2_model}.yaml")
    bbone_cfg.MODEL.ROI_HEADS.NAME = 'ROIWithScores'
    bbone_cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNNModular'

    cfg_clone = bbone_cfg.clone()  # cfg can be modified by model
    model = build_model(cfg_clone)
    # if len(bbone_cfg.DATASETS.TEST):
    #     self.metadata = MetadataCatalog.get(bbone_cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(bbone_cfg.MODEL.WEIGHTS)

    # self.input_format = bbone_cfg.INPUT.FORMAT
    # assert self.input_format in ["RGB", "BGR"], self.input_format

    return model, cfg_clone


class Joiner(pl.LightningModule):
    def __init__(self, cfg, img_feat_extractor, bb_predictor):
        super().__init__()
        self.img_feat_extractor = img_feat_extractor
        self.bb_predictor = bb_predictor

        conditioning = cfg['model']['structured']['conditioning']['name']
        cond_method = cfg['model']['structured']['conditioning'].get('method')
        if conditioning in ['x_t', 'bb_preds']:
            self.compute_bb_img_feats = False
        elif cond_method == 'seq':
            self.compute_bb_img_feats = False
        else:
            self.compute_bb_img_feats = True

    def forward(self, batch, bbone_preds):
        if bbone_preds is None:
            # if self.compute_img_feats:
            img_feats, det2_imgs = self.img_feat_extractor.get_image_feats(batch)
            # else:
            # img_feats = det2_imgs = None
            bbox_preds = self.bb_predictor(batch, img_feats=img_feats, det2_imgs=det2_imgs)
            # if bbox_preds is None:
                # img_feats, det2_imgs = self.img_feat_extractor.get_image_feats(batch)
                # bbox_preds = self.bb_predictor(batch, img_feats=img_feats, det2_imgs=det2_imgs)
                # img_feats = det2_imgs = None

        else:
            img_feats = bbone_preds['img_feats']
            bbox_preds = bbone_preds['bbone_preds']

        if self.compute_bb_img_feats:
            x_t_feats, bbone_pred_feats = self.img_feat_extractor.get_bbox_feats(bbox_preds, img_feats, batch)
        else:
            x_t_feats = bbone_pred_feats = None

        return {'bbone_preds': bbox_preds,
                'img_feats': img_feats,
                'x_t_feats': x_t_feats,
                'bbone_pred_feats': bbone_pred_feats}


############################################################################################

class DummyImageFeatureExtractor(pl.LightningModule):
    def get_image_feats(self, *args, **kwargs):
        return None, None

    def get_bbox_feats(self, *args, **kwargs):
        return None, None


class DetectronImageFeatureExtractorBase(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        data_cfg = cfg['data']

    def get_image_feats(self, batch):
        raise Exception('Must be implemented by child class')

    def get_bbox_feats(self, top_k_preds, bbone_feats, batch):
        if top_k_preds is not None:
            return self.get_x_t_and_bbone_pred_feats(top_k_preds, bbone_feats, batch)
        else:
            return self.get_x_t_feats(bbone_feats, batch), None

    def get_x_t_and_bbone_pred_feats(self, top_k_preds, bbone_feats, batch):
        batch['x_t'] = batch['x_t'].inverse_normalize().to_xyxy()
        top_k_preds = top_k_preds.inverse_normalize().to_xyxy()
        bboxes = th.cat([batch['x_t']['bbox'], top_k_preds['bbox']], dim=1)

        feats = self.extract_single_bbox_feats(bboxes, bbone_feats, batch)
        batch['x_t'] = batch['x_t'].to_train_fmt().normalize()
        top_k_preds = top_k_preds.to_train_fmt().normalize()

        x_t_feats = feats[:, :batch['x_t'].shape[1]]
        bbone_pred_feats = feats[:, batch['x_t'].shape[1]:]
        return x_t_feats, bbone_pred_feats

    def get_x_t_feats(self, bbone_feats, batch):
        bboxes = batch['x_t']['bbox']
        x_t_feats = self.extract_single_bbox_feats(bboxes, bbone_feats, batch)
        return x_t_feats

    def extract_single_bbox_feats(self, boxes, bbone_feats, batch):
        boxes = boxes.clone()

        boxes_det2 = []
        for bb, info in zip(boxes, batch['image_info']):
            box = Boxes(bb)
            box.clip((info['height'], info['width']))
            boxes_det2.append(box)

        box_feats = self.model.roi_heads.extract_bb_feats(bbone_feats, boxes_det2)
        box_feats = box_feats.view(boxes.shape[0], boxes.shape[1], -1)
        return box_feats


class Detectron2FeatExtractor(DetectronImageFeatureExtractorBase):
    def __init__(self, cfg, det2_model):
        super().__init__(cfg)
        self.model = det2_model

    def get_image_feats(self, batch):
        return self.model.backbone_inference(batch['image_info'])


class FrozenDetectron2FeatExtractor(DetectronImageFeatureExtractorBase):
    def __init__(self, cfg, det2_model_name):
        super().__init__(cfg)
        self.save_path = f'dataset/{det2_model_name}'
        os.makedirs(self.save_path, exist_ok=True)

    def get_image_feats(self, batch):
        return [th.load(os.path.join(self.save_path, 'feats_' + path), map_location=self.device) for path in batch['path']], None


############################################################################################


class DummyBBPredictor(pl.LightningModule):
    def forward(self, *args, **kwargs):
        return None


class GTruthBBPredictor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        diff_cfg = cfg['diffusion']

        num_timesteps = cfg['model']['backbone']['num_timesteps']
        if num_timesteps == 'same':
            num_timesteps = cfg['diffusion']['num_timesteps']

        beta_scheduler = BetaScheduler(
            diff_cfg['beta_scheduler'], num_timesteps)
        self.g_diffusion = GaussianNoiser(
            beta_scheduler, diff_cfg['step_sampler'])
        timestep_frac = cfg['model']['backbone']['timestep']
        self.timestep = math.ceil(num_timesteps * timestep_frac)
        logger.info('Using gt backbone with timestep {}/{}'.format(self.timestep, num_timesteps))

        self.static_noise = cfg['model']['backbone']['static_noise']
        if self.static_noise:
            self.noise_subdir = 'dataset/coco2017-1.1.0/noise'
            if os.path.exists(self.noise_subdir):
                shutil.rmtree(self.noise_subdir)
            os.makedirs(self.noise_subdir)

        self.predict_class = cfg['model']['predict_class']
        self.randomize_order = cfg['model']['backbone']['randomize_order']
        if self.randomize_order == 'once':
            self.ix_subdir = 'dataset/coco2017-1.1.0/rand_ixs'
            if os.path.exists(self.ix_subdir):
                shutil.rmtree(self.ix_subdir)
            os.makedirs(self.ix_subdir)

    def forward(self, batch, *args, **kwargs):
        x_start = batch['x_start']
        if self.timestep > 0:
            ts = th.tensor([self.timestep] * x_start.shape[0])
            preds = self.sample_noise(x_start, ts, batch['path'], batch['x_t'])
        else:
            preds = x_start
        preds = preds.to(self.device)
        preds.unmask_with_cls = True
        if self.randomize_order in ['once', True]:
            preds = self.randomize_order_fn(preds, batch['path'])
        return preds

    def sample_noise(self, x_start, ts, paths, x_t):
        if self.static_noise:
            noisy_gt_bbs = self.sample_static_noise(x_start, ts, paths, x_t)
        else:
            noisy_gt_bbs = self.sample_stochastic_noise(x_start, ts)

        if not self.predict_class:
            noisy_gt_bbs['x_t'].classes = x_start.classes
        return noisy_gt_bbs['x_t'].to_train_fmt()

    def sample_static_noise(self, x_start, ts, paths, x_t):
        x_start_tensor = th.cat([x_start['bbox'], x_start['classes_bits']], dim=-1)
        noise = []
        for path in paths:
            path = f'{self.noise_subdir}/{path}'
            if os.path.exists(path):
                try:
                    noise_sample = th.load(path).to(self.device)
                except:
                    print(path)
                    raise Exception()
            else:
                noise_sample = th.randn_like(x_start_tensor[0]).to(self.device)
                th.save(noise_sample, path)
            noise.append(noise_sample)

        noise = th.stack(noise).to(self.device)
        ts = ts.to(self.device)
        return self.g_diffusion.sample(x_start, ts=ts, noise=noise)

    def sample_stochastic_noise(self, x_start, ts):
        return self.g_diffusion.q_sample(x_start, ts=ts)

    def randomize_order_fn(self, preds, paths):
        if self.randomize_order == 'once':
            return self.randomize_order_static(preds, paths)
        else:
            return self.randomizer_order_everytime(preds, paths)

    def randomizer_order_everytime(self, preds, paths):
        max_num_bbs = preds['bbox'].shape[1]
        num_bbs = preds['padding_mask'].sum(-1)
        rand_ixs = []
        for num_bbs_img, path in zip(num_bbs, paths):
            rand_ixs_sample = th.randperm(num_bbs_img).to(self.device)
            leftover_ixs = th.arange(num_bbs_img, max_num_bbs).to(self.device)
            rand_ixs_sample = th.cat([rand_ixs_sample, leftover_ixs])
            rand_ixs.append(rand_ixs_sample)

        rand_ixs = th.stack(rand_ixs).to(self.device)
        preds = preds.reindex(rand_ixs)
        return preds

    def randomize_order_static(self, preds, paths):
        max_num_bbs = preds['bbox'].shape[1]
        num_bbs = preds['padding_mask'].sum(-1)
        rand_ixs = []
        for num_bbs_img, path in zip(num_bbs, paths):
            path = f'{self.ix_subdir}/{path}'
            if os.path.exists(path):
                try:
                    rand_ixs_sample = th.load(path).to(self.device)
                except:
                    print(path)
                    raise Exception()
            else:
                rand_ixs_sample = th.randperm(num_bbs_img).to(self.device)
                leftover_ixs = th.arange(num_bbs_img, max_num_bbs).to(self.device)
                rand_ixs_sample = th.cat([rand_ixs_sample, leftover_ixs])
                th.save(rand_ixs_sample, path)
            rand_ixs.append(rand_ixs_sample)

        rand_ixs = th.stack(rand_ixs).to(self.device)
        # rand_ixs = th.stack(
        #     [th.randperm(max_num_bbs) for _ in range(batch_size)])
        # rand_ixs = rand_ixs.to(preds['bbox'].device)
        preds = preds.reindex(rand_ixs)
        return preds

    def sample_duped_boxes(self, x_start, padding_masks, num_bbs):
        if self.concat_gt or self.dupe_range == '[1, 1]':
            return x_start, padding_masks

        dupe_range = eval(self.dupe_range)
        x_start_altered = []
        padding_masks_altered = []
        assert len(x_start.shape) == 3
        for sample, sample_mask, sample_num_bbs in zip(x_start, padding_masks, num_bbs):
            dupe_amount = np.random.uniform(
                low=dupe_range[0], high=dupe_range[1])
            if dupe_amount == 1:
                altered_x_start = sample
            elif dupe_amount < 1:
                altered_x_start, mask = self.downsample_x_start(
                    sample, dupe_amount, sample_mask, sample_num_bbs)
            elif dupe_amount > 1:
                altered_x_start, mask = self.upsample_x_start(
                    sample, dupe_amount, sample_num_bbs)

            x_start_altered.append(altered_x_start)
            padding_masks_altered.append(mask)

        return th.stack(x_start_altered), th.stack(padding_masks_altered)

    def downsample_x_start(self, x_start, dupe_amount, sample_mask, num_bbs):
        remove_prob = 1 - dupe_amount
        num_to_remove = int(num_bbs * remove_prob)
        remove_ixs = np.random.choice(num_bbs, size=num_to_remove)


############################################################################################

class Detectron2BBPredictorBase(pl.LightningModule):
    def __init__(self, cfg, det2_model, top_k=-1):
        super().__init__()

        self.model = det2_model
        data_cfg = cfg['data']
        self.bbox_transformer = create_bbox_transformer(
            cfg['dataset'],
            log_transform=data_cfg['log_transform'],
            normalization=data_cfg['normalization'],
            bb_fmt=data_cfg['bb_fmt'])
        self.bb_train_fmt = data_cfg['bb_fmt']

        self.nms_thresh = cfg['model']['backbone']['nms_thresh']
        self.conditioning = cfg['model']['structured']['conditioning']['name']
        self.top_k = top_k
        self.pad_with = cfg['model']['backbone']['pad_with']

    def forward(self, batch, img_feats=None, det2_imgs=None):
        bbox_preds, _ = self.roi_heads_inference(batch, det2_imgs=det2_imgs, img_feats=img_feats)
        if bbox_preds is None:
            return None
        bbox_preds = self.reformat_preds(batch, bbox_preds)

        self.perform_nms(bbox_preds)

        top_k_bbs = self.get_top_k_preds(bbox_preds, batch)
        return top_k_bbs

    def roi_heads_inference(self, batch, det2_imgs, img_feats):
        return self.model.roi_heads_inference(batch['image_info'], det2_imgs, img_feats)

    def reformat_preds(self, batch, bbox_preds):
        reformatted_preds = []
        for pred in bbox_preds:
            pred = pred['instances']
            reformatted_preds.append(
                {'scores': pred.scores,
                 'all_scores': pred.all_scores,
                 'pred_boxes': pred.pred_boxes.tensor,
                 'pred_classes': pred.pred_classes
                 }
            )
        return reformatted_preds

    def perform_nms(self, preds):
        for pred in preds:
            nms_ixs = box_ops.batched_nms(
                pred['pred_boxes'], pred['scores'], pred['pred_classes'],
                # pred['pred_boxes'].tensor, pred['scores'], pred['pred_classes'],
                iou_threshold=self.nms_thresh)
            for key, val in pred.items():
                pred[key] = val[nms_ixs]

    def get_top_k_preds(self, preds, batch):
        bbone_preds = []
        bbone_padding_mask = []
        if self.top_k == -1:
            top_k = max([len(pred['scores']) for pred in preds])
        else:
            top_k = self.top_k

        for pred in preds:
            top_k_preds, padding_mask = self.get_top_k_preds_single_example(pred, top_k)
            bbone_preds.append(top_k_preds)
            bbone_padding_mask.append(padding_mask)

        top_k = th.stack(bbone_preds)
        padding_masks = th.stack(bbone_padding_mask)
        top_k_bbs = top_k[..., :4]

        # top_k_bbs = self.bbox_transformer.box_xyxy_to_xywh(top_k_bbs)
        # top_k_bbs = self.bbox_transformer.normalize_bboxes(top_k_bbs)
        # top_k_bbs = top_k_bbs * th.logical_not(padding_masks).unsqueeze(-1)
        # top_k[:, ..., :4] = top_k_bbs

        # return top_k.clone(), padding_masks.clone()
        top_k_cls = top_k[..., 4:]
        top_k_bbs = BBox(
            top_k_bbs, top_k_cls, formatter=self.bbox_transformer,
            num_classes=coco_num_classes, padding_mask=padding_masks,
            format='xyxy', normalized=False, class_fmt='softmax',
            train_fmt=self.bb_train_fmt)
        top_k_bbs.unmask_with_cls = False
        top_k_bbs = top_k_bbs.to_train_fmt().normalize()
        return top_k_bbs

        # return res

    def get_top_k_preds_single_example(self, pred, top_k):
        _, sorted_ixs = th.sort(pred['scores'], descending=True)

        pred_boxes = pred['pred_boxes']
        top_k_boxes, padding_mask = self.get_top_k_single_pred(pred_boxes, sorted_ixs, top_k)
        top_k_boxes = top_k_boxes.to(self.device)
        padding_mask = padding_mask.to(self.device)

        pred_scores = pred['all_scores']
        top_k_scores, _ = self.get_top_k_single_pred(pred_scores, sorted_ixs, top_k, scores=True)
        top_k_scores = top_k_scores.to(self.device)

        top_k_preds = th.cat([top_k_boxes, top_k_scores], dim=-1)
        return top_k_preds, padding_mask

    def get_top_k_single_pred(self, preds, sorted_ixs, top_k, scores=False):
        num_preds = min(preds.shape[0], top_k)
        num_padding = max(top_k, 100) - num_preds
        if not scores:
            if self.pad_with == 'zeros':
                padding = th.zeros(1, preds.shape[1], device=preds.device)
                padding = padding.expand(num_padding, -1)
            elif self.pad_with == 'noise':
                padding = th.randn(num_padding, preds.shape[1], device=preds.device)
        else:
            padding = th.zeros(1, preds.shape[1], device=preds.device)
            padding[:, coco_bg_class] = 1
            padding = padding.expand(num_padding, -1)

        top_k_preds = preds[sorted_ixs[:top_k]]
        top_k_preds = th.cat([top_k_preds, padding], dim=0)

        padding_mask = th.tensor(
            [True] * num_preds + [False] * num_padding, device=self.device)

        return top_k_preds, padding_mask

    # def resize_all_preds(self, preds, batch, upscale=False):
    #     resized_preds = self.resize_all_boxes([pred['pred_boxes'] for pred in preds], batch, upscale=upscale)
    #     # resized_preds = self.resize_all_boxes([pred['pred_boxes'].tensor for pred in preds], batch, upscale=upscale)
    #     for pred_dct, resized_pred in zip(preds, resized_preds):
    #         pred_dct['pred_boxes'] = resized_pred
    #
    # def resize_all_boxes(self, preds, batch, upscale=False):
    #     if upscale:
    #         get_tf_fn = lambda img_transformer, img: img_transformer.get_transform(img)
    #     else:
    #         get_tf_fn = lambda img_transformer, img: img_transformer.get_transform(img).inverse()
    #
    #     resized_boxes = []
    #     for pred, img in zip(preds, batch['original_image']):
    #         resized_bbs_single = self.resize_boxes(pred, img, batch, get_tf_fn)
    #         resized_boxes.append(resized_bbs_single)
    #     return resized_boxes
    #
    # def resize_boxes(self, boxes, img, batch, get_tf_fn):
    #     img_transformer = batch['image_transformer'][0].aug
    #     img = np.transpose(img, (1, 2, 0))
    #     # inv_transform = img_transform.get_transform(img).inverse()
    #     bb_transformer = get_tf_fn(img_transformer, img)
    #
    #     # boxes_tensor = boxes['pred_boxes'].tensor
    #     # resized_boxes = bb_transformer.apply_box(boxes_tensor.cpu())
    #     resized_boxes = bb_transformer.apply_box(boxes.cpu())
    #     resized_boxes = th.from_numpy(resized_boxes).to(self.device)#.view(orig_shape)
    #     # boxes['pred_boxes'] = resized_boxes
    #     return resized_boxes


class FrozenDetectron2BBPredictor(Detectron2BBPredictorBase):
    def __init__(self, cfg, det2_model, det2_model_name, top_k=-1):
        super().__init__(cfg, det2_model, top_k=top_k)
        self.save_path = f'dataset/{det2_model_name}'
        os.makedirs(self.save_path, exist_ok=True)

    def roi_heads_inference(self, batch, det2_imgs=None, img_feats=None):
        # pred_paths = [os.path.join(self.save_path, path) for path in batch['path']]
        for path in batch['path']:
            path = os.path.join(self.save_path, path)
            if not os.path.exists(path):
                if img_feats is None:
                    return None, None
                preds, _ = super().roi_heads_inference(batch, det2_imgs, img_feats)
                self.save_preds(preds, batch)
                return preds, None

        return [th.load(os.path.join(self.save_path, path), map_location=self.device) for path in batch['path']], None

    def save_preds(self, preds, batch):
        for path, pred in zip(batch['path'], preds):
            path = os.path.join(self.save_path, path)
            th.save(pred, path)
        # box_preds = []
        # for pred_path, img in zip(pred_paths, batch['original_image']):
            # single_boxes_pred = th.load(pred_path)
            # assert 'pred_classes' in single_pred
            # if 'pred_classes' not in single_boxes_pred:
            #     if single_boxes_pred['all_scores'].shape[0] != 0:
            #         og_scores = th.unique_consecutive(single_boxes_pred['all_scores'], dim=0)[:, :-1]
            #         filter_mask = og_scores > 0.05
            #         filter_inds = filter_mask.nonzero()
            #         assert (og_scores[filter_mask] == single_boxes_pred['scores']).all()
            #         single_boxes_pred['pred_classes'] = filter_inds[:, 1]
            #     else:
            #         single_boxes_pred['pred_classes'] = th.tensor([])
            #     th.save(single_boxes_pred, pred_path)

            # box_preds.append(single_boxes_pred)

        # return box_preds


@META_ARCH_REGISTRY.register()
class GeneralizedRCNNModular(GeneralizedRCNN):
    def backbone_inference(
        self,
        batched_inputs: List[Dict[str, th.Tensor]]
    ):
        self.eval()
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor.contiguous())
        return features, images

    def roi_heads_inference(
        self, batched_inputs, images, features, proposals=None, do_postprocess=True
    ):
        if proposals is None:
            proposals, _ = self.proposal_generator(images, features, None)

        bb_preds, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not th.jit.is_scripting(), "Scripting is not supported for postprocess."
            bb_preds = GeneralizedRCNN._postprocess(bb_preds, batched_inputs, images.image_sizes)

        return bb_preds, proposals


@ROI_HEADS_REGISTRY.register()
class ROIWithScores(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # import ipdb; ipdb.set_trace()
        box_predictor = BoxPredictorWithScores(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def _forward_box(self, features: Dict[str, th.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        # import ipdb; ipdb.set_trace()
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with th.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def extract_bb_feats(self, features, bbs):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [bb for bb in bbs])
        # box_features = self.box_head(box_features)
        return box_features


class BoxPredictorWithScores(FastRCNNOutputLayers):
    def inference(self, predictions: Tuple[th.Tensor, th.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference_with_scores(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )


def fast_rcnn_inference_with_scores(
    boxes: List[th.Tensor],
    scores: List[th.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image_with_scores(
            boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image_with_scores(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = th.isfinite(boxes).all(dim=1) & th.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores_no_bg = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores_no_bg > score_thresh  # R x K

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    all_scores = th.repeat_interleave(scores, repeats=filter_mask.sum(dim=1), dim=0)
    scores = scores_no_bg[filter_mask]

    # 2. Apply NMS for each class independently.
    # keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    # if topk_per_image >= 0:
    #     keep = keep[:topk_per_image]
    # boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    result.all_scores = all_scores
    return result, th.arange(len(all_scores))


class DummyInstance:
    def __init__(self, bb):
        self.tensor = bb

    def __iter__(self):
        return iter(self.tensor)
