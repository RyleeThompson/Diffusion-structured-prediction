import torch as th
import pytorch_lightning as pl
from utils.callbacks import LoggingCallback, ModelEvaluationCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig
import hydra
import json
import numpy as np
from utils import experiment_logging
from utils import setup_utils
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
import logging
import traceback
import os

real_run = False

DEBUG = False
single_gpu = False
use_valid_set = False
overfit = False
evalu_before = False
prog_bar_freq = 1
overfit_batches_init = 0.0

if real_run:
    DEBUG = False
    use_valid_set = False
    overfit = False
    evalu_before = True
    prog_bar_freq = 500
    overfit_batches_init = 0.0
    single_gpu = False


@hydra.main(config_path='configs', config_name='config')
def main(config: DictConfig):
    # logger = setup_utils.setup_logger()
    cfg = setup_utils.setup_config(config)

    results_directory = 'BB_DDPM'

    # import ipdb; ipdb.set_trace()
    if 'ckpt_path' in cfg:
        ckpt_path = '/'.join([cfg['ckpt_path'], 'experiment_results', results_directory])
        if os.path.exists(ckpt_path):
            run_dirs = os.listdir(ckpt_path)
            assert len(run_dirs) == 1, run_dirs
            ckpt_run_id = run_dirs[0].split('-')[-1]
            cfg['resume_from'] = ckpt_run_id

    helper = experiment_logging.ExperimentHelper(cfg, results_dir=results_directory)
    cfg['results_dir'] = helper.run_dir

    if 'resume_from' not in cfg and 'ckpt_path' not in cfg:
        cfg['ckpt_path'] = cfg['results_dir']
    elif 'resume_from' in cfg:
        ckpt_path = '/'.join([cfg['ckpt_path'], 'experiment_results', results_directory])
        cfg['ckpt_path'] = '/'.join([ckpt_path, os.listdir(ckpt_path)[0]])
    elif 'ckpt_path' in cfg:
        ckpt_path = '/'.join([cfg['ckpt_path'], 'experiment_results', results_directory])
        cfg['ckpt_path'] = '/'.join([ckpt_path, cfg['results_dir'].split(results_directory)[-1]])


    helper.log(json.dumps(cfg, indent=4, sort_keys=True))

    cfg['batch_size'] = int(cfg['batch_size'] // th.cuda.device_count())

    th.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    model = setup_utils.initialize_model(cfg, overfit=overfit)
    train_loader, valid_loader = setup_utils.initialize_loaders(cfg, use_valid_set=use_valid_set, DEBUG=DEBUG, overfit=overfit)

    overfit_batches = 1 if overfit else overfit_batches_init
    limit_val_batches = 1 if overfit else 1.0
    num_gpus = 1 if single_gpu else th.cuda.device_count()
    num_gpus = 0 if DEBUG else num_gpus
    callbacks = [
        LoggingCallback(),
        # EarlyStopping(
        #     'valid_loss', min_delta=1e-3, patience=1e3
        # ),
        ModelEvaluationCallback(
            cfg, evaluate_before_train=evalu_before,
            overfit=overfit, use_valid=use_valid_set),
        ModelCheckpoint(
            save_top_k=5, monitor="valid_loss", mode="min",
            filename="{epoch:05d}-{valid_loss:.2f}", every_n_epochs=1
        ),
        # ModelCheckpoint(
        #     save_top_k=1, monitor="valid_rmse", mode="min",
        #     filename="{epoch:05d}-{valid_rmse:.2f}", every_n_epochs=1
        #     # filename="{epoch:05d}-{valid_rmse:.2f}", every_n_epochs=cfg['evaluation']['gen_freq']
        # ),
        ModelCheckpoint(
            save_top_k=1, every_n_epochs=1, filename="{epoch:05d}"
        )
    ]
    # print('*'*100)
    # if cfg['model']['predict_class']:
    #     callbacks += [ModelCheckpoint(
    #         save_top_k=1, monitor="valid_map_50", mode="max",
    #         filename="{epoch:05d}-{valid_map_50:.2f}", every_n_epochs=1)]
            # filename="{epoch:05d}-{valid_map_50:.2f}", every_n_epochs=cfg['evaluation']['gen_freq'])]

    trainer = pl.Trainer(
        # limit_train_batches=1,
        limit_val_batches=limit_val_batches,
        overfit_batches=overfit_batches,
        num_sanity_val_steps=0,
        max_epochs=cfg['epochs'],
        # log_every_n_steps=1,
        # gpus=th.cuda.device_count(),
        # gpus=th.cuda.device_count(),
        # gpus=th.cuda.device_count(),
        gpus=num_gpus,
        callbacks=callbacks,
        default_root_dir=cfg['ckpt_path'],
        # progress_bar_refresh_rate=prog_bar_freq,
        # strategy='ddp',
        # strategy=DDPStrategy(find_unused_parameters=False)
    )

    trainer.max_num_epochs = cfg['epochs']
    model.num_val_batches = len(valid_loader) if type(limit_val_batches) == float else limit_val_batches
    model.num_train_batches = len(train_loader) if type(overfit_batches) == float else overfit_batches
    model.overfit = overfit
    # train_nll_loader = setup_utils.initialize_loader(
    #     cfg, 'train', cfg['evaluation']['nll_qty'], drop_last=True)
    # valid_nll_loader = setup_utils.initialize_loader(
    #     cfg, 'val', cfg['evaluation']['nll_qty'], drop_last=True)

    # trainer.fit(model, train_loader, val_dataloaders={
    #     'valid_loader': valid_loader,
    #     'train_nll_loader': train_nll_loader,
    #     'valid_nll_loader': valid_nll_loader
    # })
    if 'resume_from' in cfg:
        ckpt_dir = '/'.join([cfg['ckpt_path'], 'lightning_logs'])
        versions = sorted(os.listdir(ckpt_dir))
        ckpt_dir = '/'.join([ckpt_dir, versions[-1], 'checkpoints'])
        ckpts = [file for file in os.listdir(ckpt_dir)]
        ckpts = sorted(ckpts)
        ckpt_file = '/'.join([ckpt_dir, ckpts[-1]])
        helper.log('Loading checkpoint from', ckpt_file)
    else:
        ckpt_file = None
    try:
        trainer.helper = helper
        trainer.fit(model, train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_file)
    except:
        traceback.print_exc()
        logging.exception('')
    finally:
        helper.end_experiment()

if __name__ == '__main__':
    main()
