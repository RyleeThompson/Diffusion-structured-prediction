from torchvision.datasets.coco import CocoDetection
import torch as th
import numpy as np
import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import get_cfg
import copy
import os
from PIL import Image
import pickle
from utils.data.BBox import BBox, create_bbox_like
import time
from torch.utils.data import Dataset
import shutil


coco_max_num_bboxes = 100
coco_bb_counts = np.array([1.0210e+03, 1.3893e+04, 2.1391e+04, 1.3924e+04, 9.9950e+03,
       7.7120e+03, 6.1950e+03, 5.0680e+03, 4.3060e+03, 3.6330e+03,
       3.1450e+03, 2.8020e+03, 2.4990e+03, 2.2000e+03, 2.5610e+03,
       2.3760e+03, 2.0680e+03, 1.8150e+03, 1.5860e+03, 1.3380e+03,
       1.1430e+03, 9.5400e+02, 8.8000e+02, 7.1700e+02, 6.5600e+02,
       5.3600e+02, 4.8600e+02, 4.4700e+02, 4.5200e+02, 3.6200e+02,
       3.1800e+02, 2.6200e+02, 2.1700e+02, 2.0600e+02, 1.4000e+02,
       1.2100e+02, 9.8000e+01, 1.0400e+02, 8.1000e+01, 6.3000e+01,
       6.2000e+01, 5.8000e+01, 5.8000e+01, 4.7000e+01, 5.6000e+01,
       3.6000e+01, 2.6000e+01, 1.9000e+01, 1.6000e+01, 1.5000e+01,
       1.0000e+01, 2.0000e+01, 1.5000e+01, 1.5000e+01, 8.0000e+00,
       1.1000e+01, 9.0000e+00, 4.0000e+00, 4.0000e+00, 2.0000e+00,
       3.0000e+00, 0.0000e+00, 3.0000e+00, 3.0000e+00, 0.0000e+00,
       3.0000e+00, 3.0000e+00, 1.0000e+00, 0.0000e+00, 2.0000e+00,
       2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
       1.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
       1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
       0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
       0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
       0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])

coco_valid_bb_counts = np.array([ 48., 593., 925., 544., 422., 285., 270., 215., 175., 191., 134.,
       117., 102.,  88.,  94., 115.,  91.,  72.,  81.,  49.,  56.,  48.,
        31.,  38.,  23.,  25.,  30.,  23.,  14.,  19.,   8.,   9.,   5.,
         5.,   5.,   8.,   5.,   7.,   5.,   5.,   4.,   3.,   2.,   3.,
         0.,   1.,   1.,   0.,   0.,   0.,   0.,   0.,   1.,   1.,   1.,
         0.,   2.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.])

coco_class_counts = np.array([0.00000e+00, 2.62465e+05, 7.11300e+03, 4.38670e+04, 8.72500e+03,
       5.13500e+03, 6.06900e+03, 4.57100e+03, 9.97300e+03, 1.07590e+04,
       1.28840e+04, 1.86500e+03, 0.00000e+00, 1.98300e+03, 1.28500e+03,
       9.83800e+03, 1.08060e+04, 4.76800e+03, 5.50800e+03, 6.58700e+03,
       9.50900e+03, 8.14700e+03, 5.51300e+03, 1.29400e+03, 5.30300e+03,
       5.13100e+03, 0.00000e+00, 8.72000e+03, 1.14310e+04, 0.00000e+00,
       0.00000e+00, 1.23540e+04, 6.49600e+03, 6.19200e+03, 2.68200e+03,
       6.64600e+03, 2.68500e+03, 6.34700e+03, 9.07600e+03, 3.27600e+03,
       3.74700e+03, 5.54300e+03, 6.12600e+03, 4.81200e+03, 2.43420e+04,
       0.00000e+00, 7.91300e+03, 2.06500e+04, 5.47900e+03, 7.77000e+03,
       6.16500e+03, 1.43580e+04, 9.45800e+03, 5.85100e+03, 4.37300e+03,
       6.39900e+03, 7.30800e+03, 7.85200e+03, 2.91800e+03, 5.82100e+03,
       7.17900e+03, 6.35300e+03, 3.84910e+04, 5.77900e+03, 8.65200e+03,
       4.19200e+03, 0.00000e+00, 1.57140e+04, 0.00000e+00, 0.00000e+00,
       4.15700e+03, 0.00000e+00, 5.80500e+03, 4.97000e+03, 2.26200e+03,
       5.70300e+03, 2.85500e+03, 6.43400e+03, 1.67300e+03, 3.33400e+03,
       2.25000e+02, 5.61000e+03, 2.63700e+03, 0.00000e+00, 2.47150e+04,
       6.33400e+03, 6.61300e+03, 1.48100e+03, 4.79300e+03, 1.98000e+02,
       1.95400e+03])

coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                  14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                  24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                  37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                  48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                  58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                  72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                  82, 84, 85, 86, 87, 88, 89, 90]

coco_bb_counts[0] = 0
coco_valid_bb_counts[0] = 0

coco_bb_count_pmf = coco_bb_counts / coco_bb_counts.sum()
coco_valid_bb_count_pmf = coco_valid_bb_counts / coco_valid_bb_counts.sum()

coco_class_pmf = coco_class_counts / coco_class_counts.sum()
coco_num_classes = 80 + 1
coco_bg_class = coco_num_classes - 1


class CocoDataset():
    def __init__(self, set, backbone, imgs_path, anns_path, pad_with,
                 max_num_bbs=coco_max_num_bboxes, bbox_transformer=None,
                 bb_train_fmt='xywh', randomize_order=False
                 ):

        assert os.path.exists(imgs_path), imgs_path
        assert os.path.exists(anns_path), anns_path

        self.imgs_path = imgs_path
        self.anns_path = anns_path

        self.ids = pickle.load(open(f'{self.anns_path}/ids.h5', 'rb'))

        self.set = set
        self.max_num_bbs = max_num_bbs

        self.det2_preprocessor = Detectron2Preprocesser('faster_rcnn_X_101_32x8d_FPN_3x')
        self.bbox_transformer = bbox_transformer
        self.pad_with = pad_with
        self.bb_train_fmt = bb_train_fmt
        self.class_mapping = {v: i for i, v in enumerate(coco_class_ids)}

        self.randomize_order = randomize_order
        if self.randomize_order == 'once':
            self.order_subdir = 'dataset/coco2017-1.1.0/order'
            if os.path.exists(self.order_subdir):
                shutil.rmtree(self.order_subdir)
            os.makedirs(self.order_subdir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ix):
        id = self.ids[ix]
        img = self.__getimage__(id)
        bboxes, classes = self.__getanns__(id)
        path = f'{self.set}_{ix}.h5'
        remapped_cls = [self.class_mapping[cls.item()] for cls in classes]
        classes = th.tensor(remapped_cls).to(classes.device)

        if len(bboxes) == 0:
            return None

        if self.randomize_order == 'once':
            ix_path = os.path.join(self.order_subdir, path)
            if os.path.exists(ix_path):
                random_ixs = th.load(ix_path)
            else:
                random_ixs = th.randperm(len(bboxes))
                th.save(random_ixs, ix_path)
        elif self.randomize_order == True:
            random_ixs = th.randperm(len(bboxes))
        else:
            random_ixs = th.arange(len(bboxes))

        bboxes = bboxes[random_ixs]
        classes = classes[random_ixs]

        bboxes = BBox(
            bboxes, classes, num_classes=coco_num_classes,
            formatter=self.bbox_transformer, format=self.bb_train_fmt, normalized=True,
            train_fmt=self.bb_train_fmt)

        # import ipdb; ipdb.set_trace()

        img_transformed = self.det2_preprocessor.resize_img(img)
        bboxes = self.resize_bbs(bboxes, img)

        num_bbs = bboxes.tensor.shape[0]
        mask = create_padding_masks(num_bbs, self.max_num_bbs)
        padded_bboxes, padded_classes = pad_data(
            bboxes.tensor, bboxes['classes'], self.max_num_bbs,
            pad_with=self.pad_with)

        bboxes.tensor = padded_bboxes
        bboxes.classes = padded_classes
        bboxes['padding_mask'] = mask

        return {
                'x_start': bboxes,
                'image_info':
                    {'image': img_transformed,
                     'height': img_transformed.shape[1],
                     'width': img_transformed.shape[2]},
                'path': path
                }

    def __getimage__(self, ix):
        path = self.imgs_path + '/{:012d}.jpg'.format(ix)
        return np.array(Image.open(path).convert("RGB"))

    def __getanns__(self, ix):
        path = self.anns_path + '/{}.h5'.format(ix)
        anns = pickle.load(open(path, 'rb'))
        bboxes = self.bbox_transformer.target_transform(anns)
        return bboxes

    def resize_bbs(self, bboxes, img):
        bboxes = bboxes.inverse_normalize().to_xyxy()
        resized_bbs = self.det2_preprocessor.resize_bbs(bboxes.tensor, img)
        bboxes.tensor = resized_bbs.to(bboxes.device)
        bboxes = bboxes.to_train_fmt().normalize()
        return bboxes


def pad_data(bboxes, classes, max_num_bbs, pad_with, bg_class=coco_bg_class):
    num_bbs = bboxes.shape[0]
    num_padding = max_num_bbs - num_bbs
    assert num_padding >= 0, num_bbs
    if num_padding == 0:
        return bboxes, classes

    classes = pad_classes(classes, num_padding, bg_class=bg_class)
    # path = os.path.join(self.pad_subdir, path)
    if pad_with == 'noise':
        # if not os.path.exists(path):
        padding = th.randn(num_padding, 4)
            # th.save(padding, path)
        # else:
            # padding = th.load(path)
    elif pad_with == 'gt':
        # if not os.path.exists(path):
        padding_ixs = np.random.choice(len(bboxes), size=num_padding)
            # th.save(padding_ixs, path)
        # else:
            # padding_ixs = th.load(path)
        padding = bboxes[padding_ixs].clone()

    bboxes = pad_bb_positions(bboxes, padding)

    return bboxes, classes

def pad_classes(classes, num_padding, bg_class=coco_bg_class):
    data = th.cat([classes, th.tensor([bg_class] * num_padding)])
    return data

def pad_bb_positions(positions, padding):
    data = th.cat([positions, padding], dim=0)
    return data

def create_padding_masks(num_bbs, max_num_bbs):
    masks = th.zeros(max_num_bbs)
    masks[:num_bbs] = True
    masks = masks.type(th.bool)
    return masks


class TowerDataset(Dataset):
    def __init__(
        self, set, bbox_transformer, bb_ordering, pad_with, max_blocks=5, bb_train_fmt='xywh'
    ):
        self.max_blocks = max_blocks
        self.save_path = f'dataset/synthetic_{self.max_blocks}'
        os.makedirs(self.save_path, exist_ok=True)

        self.set = set
        self.bbox_transformer = bbox_transformer
        self.bb_ordering = bb_ordering
        self.bb_train_fmt = bb_train_fmt
        self.pad_with = pad_with

    def __len__(self):
        if self.set == 'val':
            return int(10e3)
        elif self.set == 'train':
            return int(100e3)

    def rand_box(self, minW=0.1, minH=0.1, maxW=0.7, maxH=0.2):
        wh = th.tensor((minW, minH)) + th.rand(2)*th.tensor((maxW-minW, maxH-minH))
        xy = th.rand(2) * (1.0 - wh)
        return th.cat((xy, wh))

    def __getitem__(self, ix):
        bboxes = self.get_box(ix)
        if self.bb_ordering == 'position':
            classes = th.arange(len(bboxes))
        elif self.bb_ordering == 'none':
            bboxes = bboxes[th.randperm(len(bboxes))]
            classes = th.zeros(len(bboxes)).type(th.long)
        elif self.bb_ordering == 'size':
            areas = bboxes[:, 2] * bboxes[:, 3]
            sorted_ixs = th.sort(areas).indices
            bboxes = bboxes[sorted_ixs]
            classes = th.arange(len(bboxes))
        else:
            raise NotImplementedError(self.bb_ordering)

        # bboxes = self.bbox_transformer.normalize_bboxes(bboxes)
        if self.bb_train_fmt == 'xyxy':
            bboxes[..., 2] = bboxes[..., 2] + bboxes[..., 0]
            bboxes[..., 3] = bboxes[..., 3] + bboxes[..., 1]

        if len(bboxes) == 0:
            return {'batch': None}

        bboxes = BBox(
            bboxes, classes, num_classes=self.max_blocks + 1,
            formatter=self.bbox_transformer, format=self.bb_train_fmt, normalized=False,
            train_fmt=self.bb_train_fmt)
        bboxes = bboxes.normalize()

        if bboxes.tensor.max() > 1 or bboxes.tensor.min() < -1:
            import ipdb; ipdb.set_trace()

        num_bbs = bboxes.tensor.shape[0]
        mask = create_padding_masks(num_bbs, self.max_blocks)
        padded_bboxes, padded_classes = pad_data(
            bboxes.tensor, bboxes['classes'], self.max_blocks,
            pad_with=self.pad_with, bg_class=self.max_blocks)

        bboxes.tensor = padded_bboxes
        bboxes.classes = padded_classes
        bboxes['padding_mask'] = mask

        return {
                'x_start': bboxes,
                'image_info':
                    {'image': None,
                     'height': None,
                     'width': None},
                'path': f'{self.set}_{ix}.h5'
                }

        # return {'batch': {
        #         'x_start': bboxes,
        #         'classes': classes,
        #         'remove_masks': masks,
        #         'num_bbs': num_bbs,
        #         'image_info':
        #             {'image': None,
        #              'height': None,
        #              'width': None},
        #         'path': f'{self.set}_{ix}.h5',
        #         'original_image': None,
        #         'image_transformer': None},
        #         'g_diffusion': self.g_diffusion}

    def get_box(self, i):
        # bbs (num_bbs x 4 th.tensor): The bounding boxes found in the image (in [x_min, y_min, w, h] format).
        # classes (num_bbs th.tensor)
        save_path = self.save_path + f'/{self.set}_{i}.h5'
        if os.path.exists(save_path):
            return th.load(save_path)

        def valid_stack(blks):
            def stats(bl):  # returns center-of-mass (x-axis) and block's mass
                _cm = bl[0] + bl[2]/2.0
                _mass = bl[2]*bl[3]
                return _cm, _mass

            cm = 0.0
            mass = 0.0
            for j in range(len(blks))[::-1]:  # blocks from top to bottom
                b = blks[j]
                cm_b, mass_b = stats(b)
                cm = (cm * mass + cm_b * mass_b) / (mass + mass_b)
                mass += mass_b
                next_b = blks[j-1] if j > 0 else (0, 1, 1, 0.1)  # fake ground block
                if cm < next_b[0] or cm > next_b[0] + next_b[2]:
                    return False  # CM(prefix of stack) isn't in the support of the block below
                if b[1] < 0:  # too tall
                    return False

            return True

        def new_block(blks, maxH, max_retries=5):
            topblk = blks[-1] if blks else (0, 1, 1, 0.1) # fake ground block
            for _ in range(max_retries):
                b = self.rand_box(minW=0.1, minH=0.05, maxW=0.7, maxH=maxH) # generate a random new block
                b[1] = topblk[1]-b[3] # set y-coord to be resting on topblk
                if valid_stack(blks + [b]): # check if it's valid
                    return b

            return None

        blocks = []
        num_possible_blocks = np.random.choice(self.max_blocks) + 1
        for k in range(num_possible_blocks):
            b = new_block(blocks, maxH=min(0.3 * 5 / self.max_blocks, 1))  # try to add a new block
            if b is not None:
                blocks.append(b)

        bbs = th.stack(blocks) * 640
        th.save(bbs, save_path)
        return bbs


def get_dataset(
        set, dataset, max_num_tower_blocks, bb_ordering, pad_with,
        bbox_transformer=None, diffusion=True,
        backbone='none', max_num_bbs=coco_max_num_bboxes,
        bb_train_fmt='xywh', randomize_order=False
):
    path = '/scratch/ssd002/datasets/MSCOCO2017'
    # path = 'dataset/coco2017-1.1.0/data/raw'
    annotations_path = path + f'/annotations/instances_{set}2017.json'
    img_path = path + f'/{set}2017/'

    new_path = f'dataset/coco2017-1.1.0/{set}_anns'
    if not os.path.exists(new_path):
        os.makedirs(new_path, exist_ok=True)
        try:
            break_up_coco_dataset(annotations_path, new_path)
        except:
            os.rmdir(new_path)
    # if backbone['name'] == 'none':
    #     img_transform = None
    # elif backbone['name'] == 'gt':
    #     img_transform = Detectron2Preprocesser('gt')
    # else:
    #     img_transform = Detectron2Preprocesser(backbone['model'])
    # img_transform = Detectron2Preprocesser('faster_rcnn_X_101_32x8d_FPN_3x')

    if diffusion is True and dataset == 'coco':
        return CocoDataset(
            set, backbone, pad_with=pad_with, max_num_bbs=max_num_bbs, imgs_path=img_path,
            anns_path=new_path, bbox_transformer=bbox_transformer, bb_train_fmt=bb_train_fmt,
            randomize_order=randomize_order)
    elif diffusion is True and dataset == 'tower':
        return TowerDataset(
            set, bbox_transformer, bb_ordering, pad_with=pad_with,
            max_blocks=max_num_tower_blocks, bb_train_fmt=bb_train_fmt)

    else:
        return CocoDetection(
            root=img_path, annFile=annotations_path,
            target_transform=target_transform, transform=img_transform)


def collate_bbox_batch(bbox_lst):
    collate_dct = {}
    for key in bbox_lst[0].keys():
        if ('classes' in key and key != 'classes') or key == 'scores':
            continue
        collate_dct[key] = th.stack([bbox[key] for bbox in bbox_lst], dim=0)

    return create_bbox_like(**collate_dct, bbox_like=bbox_lst[0])


class DiffusionBatch:
    def __init__(self, batch):
        super().__init__()
        batch = [sample for sample in batch if sample is not None]

        result = {}
        for key in batch[0].keys():
            if key == 'x_start':
                result[key] = collate_bbox_batch(
                    [sample[key] for sample in batch])
            else:
                result[key] = [sample[key] for sample in batch]

        self.batch = result

    def __getitem__(self, key):
        return self.batch[key]

    def __setitem__(self, key, val):
        self.batch[key] = val

    def keys(self):
        return self.batch.keys()

    def values(self):
        return self.batch.values()

    def items(self):
        return self.batch.items()

    def update(self, dct):
        for key, val in dct.items():
            self.batch[key] = val

    def to(self, device):
        for key, val in self.items():
            try:
                self.batch[key] = val.to(device)
            except AttributeError:
                if key == 'image_info':
                    for img in self[key]:
                        if img['image'] is not None:
                            img['image'] = img['image'].to(device)
                        else:
                            break
        return self

    def pin_memory(self):
        for key, val in self.batch.items():
            try:
                self[key] = val.pin_memory()
            except AttributeError:
                if key == 'image_info':
                    for img in self[key]:
                        if img['image'] is not None:
                            img['image'] = img['image'].pin_memory()
                        else:
                            break
        return self


class Detectron2Preprocesser:
    def __init__(self, detectron2_model):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{detectron2_model}.yaml"))
        except:
            print('Couldnt get detectron2 config for', detectron2_model)
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def resize_img(self, img):
        # assert pil_img.mode == 'RGB'
        img = img[:, ..., ::-1]  # Convert to BGR
        img = self.aug.get_transform(img).apply_image(img)
        img = th.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return img

    def resize_bbs(self, bbs, img):
        bbs = self.aug.get_transform(img).apply_box(bbs)
        return th.from_numpy(bbs)


def break_up_coco_dataset(annotations_path, new_path):
    from pycocotools.coco import COCO
    import pickle
    coco = COCO(annotations_path)
    ids = list(sorted(coco.imgs.keys()))
    pickle.dump(ids, open(f'{new_path}/ids.h5', 'wb'))

    for id in ids:
        ann = coco.loadAnns(coco.getAnnIds(id))
        pickle.dump(ann, open(f'{new_path}/{id}.h5', 'wb'))

    # get_item_ix_to_coco_id = {ix: id for ix, id in enumerate(ids)}
    # coco_id_to_get_item_ix = {val: key for key, val in get_item_ix_to_coco_id.items()}
