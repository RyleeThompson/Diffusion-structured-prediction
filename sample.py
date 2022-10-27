import torch as th
import turibolt as bolt
import os
import argparse
# from utils.data.coco_dataset import coco_valid_bb_counts
from utils.setup_utils import initialize_model, setup_logger
# from utils.evaluation.HistogramEvaluation import HistogramEvaluation, SampleVisualizer
from utils.callbacks import ModelEvaluationCallback
# from utils.evaluation.NLLEvaluation import NLLEvaluation
from utils.cluster import bolt_utils
from utils.diffusion.gaussian_diffusion.ForwardGaussianDiffusion import GaussianNoiser, SpacedBetas

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ids', nargs='+')
    parser.add_argument('--sampler', type=str, default='ddpm')
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--sec_counts', type=int, default=-1)
    # parser.add_argument('--num_samples_generate', type=int, default=5000)
    # parser.add_argument('--num_samples_bpd', type=int, default=2500)
    args = parser.parse_args()
    return args


def load_ckpt(task, model, use_ema):
    save_path = bolt_utils.get_save_path(task)
    artifacts = list(task.artifacts.list(recursive=True))
    # src_path = [file for file in artifacts if 'valid_loss' in file][-1]
    # src_path = [file for file in artifacts if 'valid_rmse' in file][0]
    print([file for file in artifacts if 'epoch' in file and 'step' in file])
    print([file for file in artifacts if '.ckpt' in file])
    # src_path = [file for file in artifacts if 'epoch' in file and 'step' in file][0]
    src_path = [file for file in artifacts if 'valid_loss' in file][-1]
    print('using', src_path)
    dest_path = save_path + '/' + src_path.split('/')[-1]
    if not os.path.exists(dest_path):
        task.artifacts.download_file(src=src_path, dest=dest_path)

    ckpt = th.load(dest_path, map_location=model.device)
    # for key, val in ckpt['state_dict'].items():
    #     if 'ema_param' in key:
    #         continue
    #     ema_key = 'ema_param_' + key.replace('.', '-')
    #     try:
    #         ckpt['state_dict'][key] = ckpt['state_dict'][ema_key]
    #     except KeyError:
    #         continue
    if use_ema:
        for param in ckpt['state_dict'].keys():
            ema_name = param.replace('.', '-')
            ema_name = 'ema_param_' + ema_name
            if ema_name in ckpt['state_dict']:
                ckpt['state_dict'][param] = ckpt['state_dict'][ema_name]

    model.load_state_dict(ckpt['state_dict'])


class DummyTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.global_step = 1
        self.has_training_started = True


if __name__ == '__main__':
    # setup_logger(use_pytorch_lightning=False)
    setup_logger()
    args = parse_args()
    ids = args.ids
    tasks = bolt_utils.get_task_list(ids)

    # hist_evalu = HistogramEvaluation()
    # sample_vis = SampleVisualizer()

    with th.no_grad():
        for task in tasks:
        # task_id = 'xz63262k9j'# rMSE, eps, fixed, trans, garbage
        # task_id = '8z93czynyw' # rMSE, prev_x, fixed, trans, garbage
        # task_id = 'zi2h96ixrs' # KL, eps, fixed, trans, good
        # task_id = 'vwpev5rp3b' # KL, eps, learned_range, trans, good
        # task_id = 'cycfekrmej' # KL, eps, learned_range, mlp, good
            cfg = bolt_utils.download_config(task)
            if cfg['model']['structured']['conditioning']['name'] in ['both', 'h(x)']:
                if 'method' not in cfg['model']['structured']['conditioning']:
                    cfg['model']['structured']['conditioning']['method'] = 'roi'
            #     cfg['model']['use_time_embeds'] = False
            #     cfg['model']['use_learned_embeds'] = False

            combos = []
            combos += [{'sampler': 'ddpm', 'eta': 1.0}]
            combos += [{'sampler': 'ddim', 'eta': eta} for eta in [0, 0.1, 0.2, 0.5, 1.0]]
            # combos += [{'sampler': 'ddim', 'eta': 0}]
            for combo in combos:
                for use_ema in [False]:

                    print('Use EMA:', use_ema, '-'*100)
                    diff_cfg = cfg['diffusion']

                    # cfg['diffusion']['sampler'] = args.sampler
                    # cfg['diffusion']['eta'] = args.eta
                    cfg['diffusion']['sampler'] = combo['sampler']
                    cfg['diffusion']['eta'] = combo['eta']
                    cfg['diffusion']['section_counts'] = [args.sec_counts] if args.sec_counts != -1 else [cfg['diffusion']['num_timesteps']]
                    print(diff_cfg['sampler'], diff_cfg['eta'])

                    if 'rescale_timesteps' not in cfg['model']:
                        cfg['model']['rescale_timesteps'] = False
                    cfg['evaluation']['eval_qty'] = 256
                    cfg['evaluation']['eval_b_size'] = 256
                    model_eval = ModelEvaluationCallback(cfg,
                        evaluators=['gen'], base_dir='{}-{}-{}'.format(diff_cfg['sampler'], diff_cfg['eta'], diff_cfg['section_counts']), task=task)

                    save_dir = bolt_utils.get_results_dir(task)
                    os.makedirs(save_dir, exist_ok=True)

                    model = initialize_model(cfg)
                    if th.cuda.is_available():
                        dev = th.device('cuda')
                    else:
                        dev = th.device('cpu')
                    model = model.to(dev)
                    load_ckpt(task, model, use_ema)

                    diff_cfg = cfg['diffusion']
                    beta_scheduler = SpacedBetas(
                        diff_cfg['beta_scheduler'], diff_cfg['num_timesteps'],
                        section_counts=diff_cfg['section_counts']).to(model.device)
                    print(len(beta_scheduler.betas))
                    model.beta_scheduler = beta_scheduler
                    model.g_diffusion.beta_scheduler = beta_scheduler
                    model.reverse_diffusion.beta_scheduler = beta_scheduler
                    model.pred_head.beta_scheduler = beta_scheduler
                    model.pred_head.bbox_pos_head.reverse_diffusion.beta_scheduler = beta_scheduler
                    if hasattr(model.pred_head, 'cls_head'):
                        model.pred_head.cls_head.reverse_diffusion.beta_scheduler = beta_scheduler

                    dum_trainer = DummyTrainer()
                    # model_eval.on_train_epoch_end(dum_trainer, model)
                    model_eval.on_validation_epoch_end(dum_trainer, model)
                    # train_loader = initialize_loader(cfg, 'train', args.num_samples_bpd, drop_last=True)
                    # nll_evalu = NLLEvaluation(train_loader, args.num_samples_bpd)

                    # model_list_bbs, all_x_t_bbs = model.inference(args.num_samples_generate)
                    # import ipdb; ipdb.set_trace()
                    # Evaluation
                    # sample_vis.evaluate(all_x_t_bbs, save_dir)
                    # hist_evalu.evaluate(model_list_bbs, save_dir, num_eval_samples=args.num_samples_generate)
                    # for bb in model_list_bbs:
                    #     del bb
                    # nll_evalu.evaluate(model, save_dir)
