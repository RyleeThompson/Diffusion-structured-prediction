import torch as th
import numpy as np
import copy
from utils.data.BBox import BBox


def create_bbox_transformer(dset, log_transform=True, normalization='[-1, 1]', bb_fmt='xywh', max_num_blocks=None):
    # Returns object for transforming bboxes
    # normalizer = get_data_normalizer(log, normalization)
    return BBoxNormalizerLog(dset, normalization, bb_fmt) if log_transform else BBoxNormalizer(dset, normalization, bb_fmt, max_num_blocks=max_num_blocks)

class BBoxNormalizer:
    def __init__(self, dset, normalization, bb_fmt, max_num_blocks=None):
        self.bb_fmt = bb_fmt
        self.max_num_blocks = max_num_blocks
        self.init_coco_consts()
        if normalization == '[-1, 1]':
            self.init_range_consts(dset, bb_fmt)
        elif normalization == 'unit-var':
            self.init_unit_var_consts(dset, bb_fmt)
        else:
            raise NotImplementedError()

    def init_coco_consts(self):
        # self.COCO_CONSTS = {'Bbox xs':
        #               {'max': 638.34, 'min': 0, 'mean': 243.4618, 'std': 169.4052},
        #                'Bbox ys':
        #               {'max': 635.13, 'min': 0, 'mean': 193.0133, 'std': 119.6732},
        #                'Bbox widths':
        #               {'max': 640, 'min': 0.23, 'mean': 103.8947, 'std': 127.6180},
        #                'Bbox heights':
        #               {'max': 640, 'min': 0, 'mean': 107.4188, 'std': 144.8525}}

        self.COCO_CONSTS_SCALED = {'x_min':
                      {'max': 1328.4388, 'min': 0},
                       'y_min':
                      {'max': 1311.1515, 'min': 0},
                       'w':
                      {'max': 1333, 'min': 0.4330},
                       'h':
                      {'max': 1333, 'min': 0},
                       'x_max':
                       {'max': 1333.001, 'min': 2.5881},
                       'y_max':
                       {'max': 1333.001, 'min': 5.5455}}

    def init_range_consts(self, dset, bb_fmt):
        self.normalize_bboxes = self.normalize_range
        self.inverse_bbox_normalization = self.inverse_range_normalization

        if bb_fmt == 'xywh':
            keys = ['x_min', 'y_min', 'w', 'h']
        elif bb_fmt == 'xyxy':
            # keys = ['x_min', 'y_min', 'w', 'h']
            keys = ['x_min', 'y_min', 'x_max', 'y_max']

        if dset == 'coco':
            self.mins = th.tensor(
                [self.COCO_CONSTS_SCALED[key]['min'] for key in keys])
            self.maxs = th.tensor(
                [self.COCO_CONSTS_SCALED[key]['max'] for key in keys])

        elif dset == 'tower':
            max_h = min(0.3 * 5 / self.max_num_blocks, 1)
            if bb_fmt == 'xywh':
                self.mins = th.tensor([0, 0, 0.1, 0.05]) * 640
                self.maxs = th.tensor([1 - 0.1, 1 - 0.05, 0.7, max_h]) * 640
            elif bb_fmt == 'xyxy':
                self.mins = th.tensor([0, 0, 0.1, 0.05]) * 640
                self.maxs = th.tensor([1 - 0.1, 1 - 0.05, 1, 1]) * 640
            else:
                raise Exception(bb_fmt)

    def init_unit_var_consts(self, dset, bb_fmt):
        raise Exception('Doesnt work anymore --- need to add mean & std constants')
        self.means = th.tensor(
                    [val['mean'] for val in self.COCO_CONSTS_SCALED.values()])
        self.stds = th.tensor(
                    [val['std'] for val in self.COCO_CONSTS_SCALED.values()])

        self.normalize_bboxes = self.normalize_unit_var
        self.inverse_bbox_normalization = self.inverse_unit_var_normalization

        self.means_np = self.means.numpy()
        self.stds_np = self.stds.numpy()

    def normalize_range(self, bboxes):
        assert bboxes.shape[-1] == 4
        maxs, mins = self.get_norm_values(bboxes)
        # Normalize bbox to [-1, 1] so that it plays nicely with noise during diffusion
        bboxes = 2 * ((bboxes - mins) / (maxs - mins)) - 1
        return bboxes

    def inverse_range_normalization(self, bboxes):
        assert bboxes.shape[-1] == 4
        maxs, mins = self.get_norm_values(bboxes)
        bboxes = (bboxes + 1) * (maxs - mins) / 2
        bboxes += mins
        return bboxes

    def normalize_unit_var(self, bboxes):
        assert bboxes.shape[-1] == 4
        means, stds = self.get_norm_values(bboxes)
        bboxes = (bboxes - means) / stds
        return bboxes

    def inverse_unit_var_normalization(self, bboxes):
        assert bboxes.shape[-1] == 4
        means, stds = self.get_norm_values(bboxes)
        bboxes *= stds
        return bboxes + means

    def target_transform(self, target):
        # Extracts Bbox annotations and normalizes them (for dataset/dataloader)
        bboxes = []
        classes = []
        for ann in target:
            # bbox = th.tensor(ann['bbox']).view(1, -1)
            # bbox_norm = self.normalize_bboxes(bbox)
            bbox = th.tensor(ann['bbox'])
            if self.bb_fmt == 'xyxy':
                bbox[..., 2] += bbox[..., 0]
                bbox[..., 3] += bbox[..., 1]

            bbox_norm = self.normalize_bboxes(bbox)
            bboxes.append(bbox_norm)
            classes.append(ann['category_id'])

        if len(target) > 0:
            bboxes = th.stack(bboxes)
            classes = th.tensor(classes)
            assert len(target) == bboxes.shape[0] # bboxes should be n_bboxes x 4
        else:
            bboxes = th.tensor([])
            classes = th.tensor([])

        return bboxes, classes


    def box_xyxy_to_xywh(self, boxes):
        assert boxes.shape[-1] == 4
        # Idea is to support all boxes and do processing in batch. Same as done in box_area
        boxes[..., 2] = boxes[..., 2] - boxes[..., 0]  # x + w
        boxes[..., 3] = boxes[..., 3] - boxes[..., 1]  # y + h
        return boxes

    def box_xywh_to_xyxy(self, boxes):
        if isinstance(boxes, th.Tensor):
            boxes = boxes.clamp(min=0)
        else:
            boxes = boxes.clip(min=0)
        # assert (boxes[:, ..., 2] > 0).all(), (boxes[:, ..., 2] > 0).type(th.float32).mean()
        # assert (boxes[:, ..., 3] > 0).all(), (boxes[:, ..., 3] > 0).type(th.float32).mean()
        boxes[..., 2] = boxes[..., 2] + boxes[..., 0]  # x + w
        boxes[..., 3] = boxes[..., 3] + boxes[..., 1]  # y + h
        return boxes

    def get_norm_values(self, bboxes):
        if isinstance(bboxes, th.Tensor):
            if hasattr(self, 'maxs'):
                self.maxs = self.maxs.to(bboxes.device)
                self.mins = self.mins.to(bboxes.device)
                return self.maxs, self.mins
            else:
                self.means = self.means.to(bboxes.device)
                self.stds = self.stds.to(bboxes.device)
                return self.means, self.stds
        else:
            if hasattr(self, 'maxs'):
                return self.maxs.cpu().numpy(), self.mins.cpu().numpy()
                # return self.maxs_np, self.mins_np
            else:
                return self.means.cpu().numpy(), self.stds.cpu().numpy()
                # return self.means_np, self.stds_np

class BBoxNormalizerLog(BBoxNormalizer):
    def __init__(self, dset, normalization, bb_fmt):
        self.bb_fmt = bb_fmt
        self.init_coco_consts()

        if normalization == '[-1, 1]':
            if self.bb_fmt == 'xyxy':
                raise Exception('You probably didnt mean to use xyxy with log transform')
            if dset == 'tower':
                raise NotImplementedError()

            super().init_range_consts(dset, bb_fmt)
            self.mins[2] = th.log(self.mins[2] + 1e-5)
            self.max[3] = th.log(self.maxs[3] + 1e-5)

        elif normalization == 'unit-var':
            raise Exception('Doesnt work anymore --- need to add mean & std consts')

        else:
            raise NotImplementedError(normalization)

    def normalize_range(self, bboxes):
        log_fn = self.get_log_fn(bboxes)
        bboxes[..., 2:] = log_fn(bboxes[..., 2:] + 1e-5)
        bboxes = super().normalize_range(bboxes)
        return bboxes

    def inverse_range_normalization(self, bboxes):
        bboxes = super().inverse_range_normalization(bboxes)
        exp_fn = self.get_exp_fn(bboxes)
        bboxes[..., 2:] = exp_fn(bboxes[..., 2:])
        return bboxes

    def normalize_unit_var(self, bboxes):
        log_fn = self.get_log_fn(bboxes)
        bboxes[..., 2:] = log_fn(bboxes[..., 2:] + 1e-5)
        bboxes = super().normalize_unit_var(bboxes)
        return bboxes

    def inverse_unit_var_normalization(self, bboxes):
        bboxes = super().inverse_unit_var_normalization(bboxes)
        exp_fn = self.get_exp_fn(bboxes)
        bboxes[..., 2:] = exp_fn(bboxes[..., 2:])
        return bboxes

    def get_log_fn(self, bboxes):
        if isinstance(bboxes, th.Tensor):
            log_fn = th.log
        else:
            log_fn = np.log
        return log_fn

    def get_exp_fn(self, bboxes):
        if isinstance(bboxes, th.Tensor):
            exp_fn = th.exp
        else:
            exp_fn = np.exp
        return exp_fn
