import torch as th
import numpy as np
from utils import class_formatting as cls


def create_bbox_like(
        bbox, classes, padding_mask, bbox_like, class_fmt=None
):
    kwargs = {}

    kwargs['formatter'] = bbox_like.formatter
    kwargs['format'] = bbox_like.format
    kwargs['normalized'] = bbox_like.normalized
    kwargs['num_classes'] = bbox_like.num_classes
    kwargs['train_fmt'] = bbox_like.train_fmt

    kwargs['padding_mask'] = padding_mask
    kwargs['bbox'] = bbox
    kwargs['classes'] = classes
    if class_fmt is not None:
        kwargs['class_fmt'] = class_fmt
    # if use_cls_in_feat is not None:
        # kwargs['use_cls_in_feat'] = use_cls_in_feat
    # else:
        # kwargs['use_cls_in_feat'] = bbox_like.use_cls_in_feat

    return BBox(**kwargs)


class BBox:
    def __init__(
        self, bbox, classes, formatter, num_classes,
        padding_mask=None, format='xywh', normalized=False, class_fmt='int',
        train_fmt='xywh',
    ):
        """
            BBox: N x M x 4 tensor, N is batch size
            padding_mask: N x M tensor, with True indicating non-padded entries
        """
        assert isinstance(bbox, th.Tensor) or isinstance(bbox, np.ndarray), type(bbox)
        assert bbox.shape[-1] == 4, bbox.shape
        assert classes.shape[0] == bbox.shape[0], f'{classes.shape}, {bbox.shape}'
        # if len(classes_int.shape) > 1:
            # assert classes_int.shape[1] == bbox.shape[1], f'{classes_int.shape}, {bbox.shape}'
            # assert classes_int.dtype == th.long, classes_int.dtype

        self.num_classes = num_classes
        self.train_fmt = train_fmt
        self.store_dct = {}
        self.store_dct['bbox'] = bbox

        self.class_fmt = class_fmt
        if class_fmt == 'int':
            self.classes = classes
            self.store_dct['scores'] = self.scores_ones()
        elif class_fmt == 'bits':
            self.classes_bits = classes
            self.store_dct['scores'] = self.scores_ones()
        elif class_fmt == 'softmax':
            self.classes_softmax = classes
            self.store_dct['scores'] = self.scores_soft()

        if padding_mask is not None:
            assert len(bbox.shape) == 3, bbox.shape
            assert padding_mask.shape == bbox.shape[:2], padding_mask.shape
            self.store_dct['padding_mask'] = padding_mask

        self.format = format
        self.normalized = normalized
        self.device = bbox.device
        self.formatter = formatter

    def __getitem__(self, key):
        return self.store_dct[key]

    def __setitem__(self, key, val):
        if len(self['bbox'].shape) > 2 and key != 'bbox':
            assert val.shape[:2] == self['bbox'].shape[:2], '{}, {}, {}'.format(key, val.shape, self['bbox'].shape)
        elif key != 'bbox':
            assert val.shape[0] == self['bbox'].shape[0], '{}, {}, {}'.format(key, val.shape, self['bbox'].shape)
        self.store_dct[key] = val

    def clone(self):
        if self.class_fmt == 'int':
            classes = self['classes']
        elif self.class_fmt == 'bits':
            classes = self['classes_bits']
        elif self.class_fmt == 'softmax':
            classes = self['classes_softmax']
        return create_bbox_like(
            self['bbox'].clone(), classes.clone(),
            self['padding_mask'].clone(), self,
            class_fmt=self.class_fmt)

    def keys(self):
        return self.store_dct.keys()

    def values(self):
        return self.store_dct.values()

    def items(self):
        return self.store_dct.items()

    def unmask(self, keys=None, use_cls=False):
        unmasked = {}
        if keys is None:
            keys = self.store_dct.keys()

        for key in keys:
            val = self.store_dct[key]
            unmasked[key] = self.unmask_single_entry(val, use_cls)
        return unmasked

    def unmask_single_entry(self, val, use_cls):
        unmasked = []
        for ix, sub_val in enumerate(val):
            result = self.unmask_single_example(ix, sub_val, use_cls)
            unmasked.append(result)
        return unmasked

    def unmask_single_example(self, ix, sub_val, use_cls):
        if not use_cls:
            padding_mask = self['padding_mask'][ix]
        else:
            padding_mask = self['classes'][ix] != (self.num_classes - 1)
        return sub_val[padding_mask]

    def reindex(self, ixs):
        assert len(ixs.shape) == (len(self['bbox'].shape) - 1), ixs.shape
        ixs = ixs.to(self['bbox'].device)
        for key, val in self.items():
            if len(val.shape) == 3:
                temp_ixs = ixs.unsqueeze(-1)
                temp_ixs = temp_ixs.expand(-1, -1, val.shape[-1])
            else:
                temp_ixs = ixs

            self[key] = val.gather(1, temp_ixs)
        return self

    def normalize(self):
        assert self.formatter is not None
        if self.format != self.train_fmt:
            self.to_train_fmt()

        if not self.normalized:
            bbs = self.store_dct['bbox']
            bbs = self.formatter.normalize_bboxes(bbs)
            self.store_dct['bbox'] = bbs
        self.normalized = True
        return self

    def inverse_normalize(self):
        assert self.formatter is not None
        if self.format != self.train_fmt:
            self.to_train_fmt()

        if self.normalized:
            bbs = self.store_dct['bbox']
            bbs = self.formatter.inverse_bbox_normalization(bbs)
            self.store_dct['bbox'] = bbs
        self.normalized = False
        return self

    def get_features(self, cls_fmt=None):
        if cls_fmt == 'softmax':
            cat_lst = [self['bbox'], self['classes_softmax']]
        elif cls_fmt == 'bits':
            cat_lst = [self['bbox'], self['classes_bits']]
        elif cls_fmt is None:
            cat_lst = [self['bbox']]
        else:
            raise Exception(cls_fmt)
        return th.cat(cat_lst, dim=-1)

    def to_train_fmt(self):
        if self.train_fmt == 'xyxy':
            return self.to_xyxy()
        elif self.train_fmt == 'xywh':
            return self.to_xywh()
        else:
            raise Exception()

    def to_xyxy(self):
        if self.format == 'xywh':
            if self.normalized:
                raise Exception('Should inv. normalize first')
            bbs = self.store_dct['bbox']
            bbs = box_xywh_to_xyxy(bbs)
            self.store_dct['bbox'] = bbs

        elif self.format == 'xyxy':
            pass
        else:
            raise NotImplementedError()
        self.format = 'xyxy'
        return self

    def to_xywh(self):
        if self.format == 'xyxy':
            if self.normalized:
                raise Exception('Should inv. normalize first')
            bbs = self.store_dct['bbox']
            bbs = box_xyxy_to_xywh(bbs)
            self.store_dct['bbox'] = bbs

        elif self.format == 'xywh':
            pass
        else:
            raise NotImplementedError()
        self.format = 'xywh'
        return self

    def to(self, device):
        for key, val in self.items():
            self.store_dct[key] = val.to(device)
        return self

    def pin_memory(self):
        for key, val in self.items():
            self.store_dct[key] = val.pin_memory()
        return self

    @property
    def tensor(self):
        return self.store_dct['bbox']

    @tensor.setter
    def tensor(self, value):
        self.store_dct['bbox'] = value

    @property
    def classes(self):
        return self.store_dct['classes']

    @classes.setter
    def classes(self, value):
        self['classes'] = value
        self['classes_bits'] = cls.int2scaledbits(
            self['classes'], num_classes=self.num_classes)
        self['classes_softmax'] = cls.int2softmax(
            self['classes'], num_classes=self.num_classes)

    @property
    def classes_bits(self):
        return self.store_dct['classes_bits']

    @classes_bits.setter
    def classes_bits(self, value):
        self['classes_bits'] = value
        self['classes'] = cls.scaledbits2int(value).clamp(max=self.num_classes - 1)
        self['classes_softmax'] = cls.int2softmax(
            self['classes'], num_classes=self.num_classes)

    @property
    def classes_softmax(self):
        return self.store_dct['classes_softmax']

    @classes_softmax.setter
    def classes_softmax(self, value):
        self['classes_softmax'] = value
        self['classes'] = cls.softmax2int(value)
        self['classes_bits'] = cls.int2scaledbits(
            self['classes'], num_classes=self.num_classes)

    @property
    def shape(self):
        return self.store_dct['bbox'].shape

    # @property
    # def scores(self):
    #     return self.scores_fn()
    #
    def scores_ones(self):
        shape = self['bbox'].shape[:-1]
        return th.ones(shape).to(self['bbox'].device)

    def scores_soft(self):
        return self['classes_softmax'].max(-1).values


def box_xywh_to_xyxy(bbox):
    assert bbox.shape[-1] == 4
    bbox[..., 2] = bbox[..., 2] + bbox[..., 0]  # x2 = x + w
    bbox[..., 3] = bbox[..., 3] + bbox[..., 1]  # y2 = y + h
    return bbox

def box_xyxy_to_xywh(bbox):
    assert bbox.shape[-1] == 4
    # Idea is to support all boxes and do processing in batch. Same as done in box_area
    bbox[..., 2] = bbox[..., 2] - bbox[..., 0]  # x + w
    bbox[..., 3] = bbox[..., 3] - bbox[..., 1]  # y + h
    return bbox
