import torch as th
from utils.data.BBNormalizer import create_bbox_transformer
from utils.data.coco_dataset import get_dataset, DiffusionBatch
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from diffusion_model.StructuredDiffusionModel import get_diffusion_model
from utils.diffusion.gaussian_diffusion.ForwardGaussianDiffusion import GaussianNoiser, SpacedBetas
from omegaconf import OmegaConf
from utils.data.BBox import BBox
import utils.data.coco_dataset as coco_dset
from functools import partial


def setup_config(config):
    cfg = get_config(config)
    # save_config(cfg)
    return cfg

def init_diffusion_utils(cfg):
    diff_cfg = cfg['diffusion']
    data_cfg = cfg['data']

    beta_scheduler = SpacedBetas(
        diff_cfg['beta_scheduler'], diff_cfg['num_timesteps'])
    g_diffusion = GaussianNoiser(
        beta_scheduler, diff_cfg['step_sampler'], loss_type=diff_cfg['loss_type'])
    bbox_transformer = create_bbox_transformer(
        cfg['dataset'],
        log_transform=data_cfg['log_transform'],
        normalization=data_cfg['normalization'],
        bb_fmt=data_cfg['bb_fmt'],
        max_num_blocks=cfg['max_num_blocks'])
    return beta_scheduler, g_diffusion, bbox_transformer

def initialize_model(cfg, overfit=False):
    beta_scheduler, g_diffusion, bbox_transformer = init_diffusion_utils(cfg)

    # BBox.normalizer = bbox_transformer
    DiffusionBatch.g_diffusion = g_diffusion

    model = get_diffusion_model(
        cfg, beta_scheduler, g_diffusion, bbox_transformer)

    return model


def initialize_loaders(cfg, pin_memory=True, use_valid_set=False, DEBUG=False, overfit=False):
    if not use_valid_set:
        train_loader = initialize_loader(
            cfg, 'train', cfg['batch_size'], pin_memory=pin_memory,
            overfit=overfit, DEBUG=DEBUG)
    else:
        train_loader = initialize_loader(
            cfg, 'val', cfg['batch_size'], pin_memory=pin_memory,
            overfit=overfit, DEBUG=DEBUG)

    valid_loader = initialize_loader(
        cfg, 'val', cfg['batch_size'], pin_memory=pin_memory,
        overfit=overfit, DEBUG=DEBUG)
    return train_loader, valid_loader


def initialize_loader(cfg, set, batch_size, pin_memory=True,
                      overfit=False, DEBUG=False, drop_last=False):
    _, _, bbox_transformer = init_diffusion_utils(cfg)

    dset = get_dataset(
        set, cfg['dataset'], cfg['max_num_blocks'], cfg['bb_ordering'],
        bbox_transformer=bbox_transformer,
        pad_with=cfg['data']['pad_with'],
        backbone=cfg['model']['backbone'],
        max_num_bbs=cfg['model']['max_num_preds'],
        bb_train_fmt=cfg['data']['bb_fmt'],
        randomize_order=cfg['data']['randomize_order'])

    collate_fn = DiffusionBatch

    num_workers = 0 if DEBUG else cfg['num_workers']
    shuffle = True if (not overfit and set != 'val') else False
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=pin_memory,
                        collate_fn=collate_fn, worker_init_fn=worker_init_fn,
                        drop_last=drop_last)
    return loader


def get_config(config):
    cfg = OmegaConf.to_container(config)
    return cfg


# def save_config(config):
#     save_path = config['results_dir']
#     with open(os.path.join(save_path, 'config.json'), 'w') as f:
#         json.dump(config, f, indent=4, sort_keys=True)


def worker_init_fn(worker_id):
    # torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    torch_seed = th.initial_seed()
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)
