from pytorch_lightning.callbacks import Callback
import logging
import torch as th
import time
import json
from utils import setup_utils as utils
from utils.evaluation import HistogramEvaluation as hist
from utils.evaluation import NLLEvaluation as nll
from utils.evaluation import RMSEEvaluation as rmse
import os
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tidecv import TIDE, datasets
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import math

logger = logging.getLogger('my_task')


class LoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.time_last_log = time.time()
        # self.sanity_check = False

    def on_train_epoch_start(self, *args, **kwargs):
        self.start_time = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, model):
        logger.info(f'Epoch {trainer.current_epoch}')
        outputs = model.training_outputs
        means = self.get_mean_from_epoch(outputs, set='train')
        log_results(trainer, means)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, model):
        outputs = model.validation_outputs
        means = self.get_mean_from_epoch(outputs, set='valid')
        logger.info(f'Iteration: {trainer.global_step} epoch \
            {trainer.current_epoch} end')

        # if self.sanity_check is False:
        msg = 'Epoch time: {:.3f}'.format(time.time() - self.start_time)
        logger.info(msg)
        log_results(trainer, means)

    def get_mean_from_epoch(self, outputs, set):
        means = {}
        for out in outputs:
            for key, val in out.items():
                if key in means:
                    means[f'{set}_{key}'].append(val)
                else:
                    means[f'{set}_{key}'] = [val]

        for key, val in means.items():
            means[key] = th.mean(th.tensor(val)).item()
            # means[f'{set}_{key}'] = th.mean(th.tensor([dct[key] for dct in outputs if key in dct])).item()
        return means
    #
    # def on_sanity_check_start(self, *args, **kwargs):
    #     self.sanity_check = True
    #
    # def on_sanity_check_end(self, *args, **kwargs):
    #     self.sanity_check = False


class ModelEvaluationCallback(Callback):
    def __init__(self, cfg, base_dir=None, evaluate_before_train=True,
                 overfit=False, evaluators=['nll', 'gen'], task=None,
                 use_valid=False):
                 
        self.base_dir = os.path.join(cfg['results_dir'], 'artifacts')
        print('saving to', self.base_dir)
        self.evaluate_before_train = evaluate_before_train

        eval_cfg = cfg['evaluation']
        if overfit is False:
            train_set = 'train' if not use_valid else 'val'
            train_loader = utils.initialize_loader(
                cfg, train_set, eval_cfg['eval_b_size'], pin_memory=False,
                drop_last=True)
            valid_loader = utils.initialize_loader(
                cfg, 'val', eval_cfg['eval_b_size'], pin_memory=False,
                drop_last=True)
        else:
            train_loader = None
            valid_loader = None

        self.evaluators = []
        for evaluator in evaluators:
            if evaluator == 'nll':
                self.evaluators.append(NLLCallback(cfg, train_loader, valid_loader))
            elif evaluator == 'gen':
                self.evaluators.append(GeneratorCallback(cfg, train_loader, valid_loader, self.base_dir, eval_cfg['eval_b_size']))
            else:
                raise NotImplementedError(evaluator)

        self.forward_process_vis = ForwardProcessVisCallback(cfg, self.base_dir)

        # self.evaluators = [nll_evaluator, gen_evaluator]
        # self.evaluators = [nll_evaluator]
        # self.evaluators = [gen_evaluator]

    @rank_zero_only
    def on_train_start(self, trainer, model):
        if self.evaluate_before_train:
            trainer.has_training_started = False
            self.forward_process_vis.on_train_start(trainer, model)
            self.on_validation_epoch_end(trainer, model)
            self.on_train_epoch_end(trainer, model)
        trainer.has_training_started = True

    @rank_zero_only
    def on_train_epoch_end(self, trainer, model):
        save_dir = get_save_dir(self.base_dir, trainer)
        for evaluator in self.evaluators:
            evaluator.on_train_epoch_end(save_dir, trainer, model)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, model):
        save_dir = get_save_dir(self.base_dir, trainer)
        for evaluator in self.evaluators:
            evaluator.on_validation_epoch_end(save_dir, trainer, model)

    @rank_zero_only
    def on_fit_end(self, trainer, model):
        self.on_validation_epoch_end(trainer, model)
        self.on_train_epoch_end(trainer, model)


class SingleEvaluatorCallback(Callback):
    def should_evaluate(self, trainer):
        # print(trainer.current_epoch)
        return trainer.current_epoch == 0 or \
                trainer.current_epoch % self.freq == 0 or \
                (trainer.max_num_epochs - 1) == trainer.current_epoch
                # ((trainer.current_epoch - 1) % self.freq == 0 and trainer.current_epoch != 1) or \
                # (trainer.current_epoch == 1 and self.freq == 1) or \

    def on_train_epoch_end(self, save_dir, trainer, evaluator_in, log=True, **kwargs):
        if self.should_evaluate(trainer):
            result = self.train_evaluator.evaluate(evaluator_in, save_dir, **kwargs)
            if log:
                self.update_and_log_results(result, 'train', trainer)
            return result

    def on_validation_epoch_end(self, save_dir, trainer, evaluator_in, log=True, **kwargs):
        if self.should_evaluate(trainer):
            result = self.valid_evaluator.evaluate(evaluator_in, save_dir, **kwargs)
            if log:
                self.update_and_log_results(result, 'valid', trainer)
            return result

    def update_and_log_results(self, dct, set, trainer, model=None):
        if dct is not None:
            try:
                dct = {set + '_' + key: val.item() for key, val in dct.items()}
            except AttributeError:
                dct = {set + '_' + key: val for key, val in dct.items()}
            log_results(trainer, dct, model=model)

    def on_train_start(self, *args, **kwargs):
        pass

    def compute_delta(self, start, end):
        assert start.keys() == end.keys()
        delta = {}
        for key in start.keys():
            delta_key = 'delta_' + key
            delta[delta_key] = start[key] - end[key]
        return delta


class NLLCallback(SingleEvaluatorCallback):
    def __init__(self, cfg, train_loader, valid_loader):
        eval_cfg = cfg['evaluation']

        self.freq = eval_cfg['nll_freq']
        self.train_evaluator = nll.NLLEvaluation(cfg, 'train', train_loader)
        self.valid_evaluator = nll.NLLEvaluation(cfg, 'valid', valid_loader)


class GeneratorCallback(SingleEvaluatorCallback):
    def __init__(self, cfg, train_loader, valid_loader, base_dir, b_size):
        self.base_dir = base_dir
        eval_cfg = cfg['evaluation']

        self.freq = eval_cfg['gen_freq']
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # self.hist_eval = HistogramEvaluationCallback(self.freq)
        self.sample_vis = SampleVisualizationCallback(cfg, self.freq)
        self.valid_blocks = ValidBlocksCallback(cfg, self.freq)
        self.obj_det_eval = ObjDetEvalCallback(cfg, self.freq)
        self.rmse_eval = rmse.RMSEEvaluation(cfg)

        self.conditioning_type = cfg['model']['structured']['conditioning']['name']
        self.dataset = cfg['dataset']
        self.predict_class = cfg['model']['predict_class']
        self.num_batches = math.ceil(cfg['evaluation']['eval_qty'] / b_size)

    def on_validation_epoch_end(self, save_dir, trainer, model):
        if self.should_evaluate(trainer):
            gens_ema = model.ema.ema_model.inference_loader(self.valid_loader, num_batches=self.num_batches)
            self.evaluate(gens_ema, 'valid_ema', trainer, model.ema.ema_model)
            gens = model.inference_loader(self.valid_loader, num_batches=self.num_batches)
            self.evaluate(gens, 'valid', trainer, model)

    def on_train_epoch_end(self, save_dir, trainer, model):
        if self.should_evaluate(trainer):
            gens_ema = model.ema.ema_model.inference_loader(self.train_loader, num_batches=self.num_batches)
            self.evaluate(gens_ema, 'train_ema', trainer, model.ema.ema_model)
            gens = model.inference_loader(self.train_loader, num_batches=self.num_batches)
            self.evaluate(gens, 'train', trainer, model)

    def evaluate(self, gens, set, trainer, model):
        save_dir = get_save_dir(self.base_dir, trainer)

        avg_bbs = {'avg_num_bbs': np.mean([bb.shape[0] for bb in gens['final_bboxes']])}
        self.update_and_log_results(avg_bbs, set, trainer, model=model)

        if self.dataset == 'tower':
            model_valid = self.valid_blocks.evaluate(gens['final_bboxes'], gens['final_cls_int_preds'])
            self.update_and_log_results(model_valid, set, trainer, model=model)

        if self.conditioning_type != 'x_t':
            model_rmse = self.rmse_eval.evaluate(
                [sample.clamp(min=1e-5) for sample in gens['final_bboxes']],
                gens['final_cls_preds'], gens['x_start'], gens['x_start_cls'])
            if self.conditioning_type in ['both', 'bb_preds']:
                cond_rmse = self.rmse_eval.evaluate(
                    [sample.clamp(min=1e-5) for sample in gens['bbone_preds']],
                    gens['x_start_cls'], gens['x_start'], gens['x_start_cls'])
                delta_rmse = self.compute_delta(cond_rmse, model_rmse)
                cond_rmse = {'cond_' + key: val for key, val in cond_rmse.items()}
                self.update_and_log_results(cond_rmse, set, trainer, model=model)
                self.update_and_log_results(delta_rmse, set, trainer, model=model)

            self.update_and_log_results(model_rmse, set, trainer, model=model)

        if self.predict_class and self.dataset != 'tower':
            model_obj_det_metrics = self.obj_det_eval.evaluate(
                [sample.clamp(min=1e-5) for sample in gens['final_bboxes']],
                gens['final_cls_int_preds'], gens['final_scores_preds'], gens['x_start'], gens['x_start_cls'])
            self.update_and_log_results(model_obj_det_metrics, set, trainer, model=model)

            if self.conditioning_type in ['both', 'bb_preds']:
                cond_metrics = self.obj_det_eval.evaluate(
                    [sample.clamp(min=1e-5) for sample in gens['bbone_preds']],
                    gens['bbone_cls_int'], gens['bbone_scores'], gens['x_start'], gens['x_start_cls'])
                delta_metrics = self.compute_delta(cond_metrics, model_obj_det_metrics)
                cond_metrics = {'cond_' + key: val for key, val in cond_metrics.items()}
                self.update_and_log_results(cond_metrics, set, trainer, model=model)
                self.update_and_log_results(delta_metrics, set, trainer, model=model)

        # if set == 'valid':
        #     hist_eval_fn = self.hist_eval.on_validation_epoch_end
        # else:
        #     hist_eval_fn = self.hist_eval.on_train_epoch_end

        # model_mmds = hist_eval_fn(
        #     save_dir, trainer, gens['final_bboxes'])
        # if self.conditioning_type in ['both', 'bb_preds']:
        #     cond_mmds = hist_eval_fn(
        #         save_dir, trainer, gens['bbone_preds'], log=False)
        #     delta_mmds = self.compute_delta(cond_mmds, model_mmds)
        #     self.update_and_log_results(delta_mmds, set, trainer, model=model)

        if set == 'valid':
            self.sample_vis.on_validation_epoch_end(
                save_dir, trainer, gens)
        else:
            self.sample_vis.on_train_epoch_end(
                save_dir, trainer, gens)


class HistogramEvaluationCallback(SingleEvaluatorCallback):
    def __init__(self, freq):
        self.freq = freq
        self.train_evaluator = hist.HistogramEvaluation(
            set='train', num_eval_samples=5000)
        self.valid_evaluator = hist.HistogramEvaluation(
            set='val', num_eval_samples=5000)


class ObjDetEvalCallback(SingleEvaluatorCallback):
    def __init__(self, cfg, freq):
        self.freq = freq
        self.tide_eval = TideEvalCallback()
        self.tide_eval.res_dir = cfg['results_dir']

    def evaluate(self, all_bb_pred_pos, all_bb_pred_cls, all_pred_scores, all_gt_pos, all_gt_cls):
        map_dct = self.tide_eval.evaluate(all_bb_pred_pos, all_bb_pred_cls, all_pred_scores, all_gt_pos, all_gt_cls)
        # print(map_dct)
        # tide_dct = self.map_eval.evaluate(all_bb_pred_pos, all_bb_pred_cls, all_gt_pos, all_gt_cls)
        # print(tide_dct)
        # map_dct.update(tide_dct)
        return map_dct


class TideEvalCallback(SingleEvaluatorCallback):
    def evaluate(self, all_bb_pred_pos, all_bb_pred_cls, all_pred_scores, all_gt_pos, all_gt_cls):
        preds = []
        gts = []
        for img_ix in range(len(all_bb_pred_pos)):
            img_pred_pos = all_bb_pred_pos[img_ix]
            img_pred_cls = all_bb_pred_cls[img_ix]
            img_pred_scores = all_pred_scores[img_ix]

            img_gt_pos = all_gt_pos[img_ix]
            img_gt_cls = all_gt_cls[img_ix]

            for bb_ix in range(len(img_pred_pos)):
                single_pred_pos = img_pred_pos[bb_ix]
                single_pred_cls = img_pred_cls[bb_ix]
                single_pred_score = img_pred_scores[bb_ix]
                preds.append({'bbox': single_pred_pos.cpu().tolist(),
                              'category_id': single_pred_cls.item(),
                              'score': single_pred_score.item(),
                              'image_id': img_ix})
            for bb_ix in range(len(img_gt_pos)):
                single_gt_pos = img_gt_pos[bb_ix]
                single_gt_cls = img_gt_cls[bb_ix]
                gts.append({'bbox': single_gt_pos.cpu().tolist(),
                              'category_id': single_gt_cls.nonzero().item(),
                              'score': 1,
                              'image_id': img_ix})

        # import ipdb; ipdb.set_trace()
        pred_path = os.path.join(self.res_dir, 'temp-preds.json')
        gt_path = os.path.join(self.res_dir, 'temp-gts.json')
        json.dump(preds, open(pred_path, 'w'))
        json.dump(gts, open(gt_path, 'w'))

        tide = TIDE()
        try:
            tide.evaluate_range(datasets.COCOResult(gt_path), datasets.COCOResult(pred_path), mode=TIDE.BOX)
            set_all_to_zero = False
            title = 'temp-preds'
        except ZeroDivisionError:
            set_all_to_zero = True
            tide.evaluate_range(datasets.COCOResult(gt_path), datasets.COCOResult(gt_path), mode=TIDE.BOX)
            title = 'temp-gts'
        errors = tide.get_all_errors()
        main_errors = errors['main'][title]
        main_errors.update(errors['special'][title])
        thresh_runs = tide.run_thresholds[title]
        aps = [trun.ap for trun in thresh_runs]
        for trun in thresh_runs:
            thresh = str(int(trun.pos_thresh * 100))
            ap = trun.ap
            main_errors['map_' + thresh] = ap
        main_errors['map_{}-{}'.format(int(thresh_runs[0].pos_thresh*100), int(thresh_runs[-1].pos_thresh*100))] = sum(aps) / len(aps)
        if set_all_to_zero:
            for key, val in main_errors.items():
                main_errors[key] = 0

        os.remove(pred_path)
        os.remove(gt_path)

        return main_errors

class ValidBlocksCallback(SingleEvaluatorCallback):
    def __init__(self, cfg, freq):
        self.freq = freq
        self.evaluator = ValidBlocksEvaluation(cfg['bb_ordering'])

    def evaluate(self, *args, **kwargs):
        return self.evaluator.evaluate(*args, **kwargs)


class ValidBlocksEvaluation:
    def __init__(self, bb_ordering):
        self.bb_ordering = bb_ordering

    def evaluate(self, blks, classes):
        valid_cm_all = []
        valid_ys_all = []
        valid_areas_all = []
        valid_all = []
        for pred_blks, pred_classes in zip(blks, classes):
            rand_ixs = th.randperm(len(pred_blks))
            pred_blks = pred_blks[rand_ixs]
            pred_classes = pred_classes[rand_ixs]

            sorted_ixs = th.sort(pred_classes, descending=False).indices
            pred_blks = pred_blks[sorted_ixs]
            valid_cm = self.validate_cms(pred_blks)
            valid_ys = self.validate_ys(pred_blks)
            if self.bb_ordering == 'size':
                valid_areas = self.validate_areas(pred_blks)
            else:
                valid_areas = True
            valid = valid_cm and valid_ys and valid_areas

            valid_cm_all.append(valid_cm)
            valid_ys_all.append(valid_ys)
            valid_areas_all.append(valid_areas)
            valid_all.append(valid)

        results = {'valid': np.mean(valid_all),
                   'valid_cm': np.mean(valid_cm_all),
                   'valid_ys': np.mean(valid_ys_all)}
        if self.bb_ordering == 'size':
            results['valid_areas'] = np.mean(valid_areas_all)
        return results

    def validate_cms(self, blks):
        def stats(bl):  # returns center-of-mass (x-axis) and block's mass
            _cm = bl[0] + bl[2]/2.0
            _mass = bl[2]*bl[3]
            return _cm, _mass

        if self.bb_ordering != 'position':
            blks = self.sort_ys(blks)

        cm = 0.0
        mass = 0.0
        epsilon = 15
        for j in range(len(blks))[::-1]:  # blocks from top to bottom
            b = blks[j]
            cm_b, mass_b = stats(b)
            cm = (cm * mass + cm_b * mass_b) / (mass + mass_b)
            mass += mass_b
            next_b = blks[j-1] if j > 0 else th.tensor([0, 1, 1, 0.1]) * 640  # fake ground block
            lower_range = next_b[0] - epsilon
            upper_range = next_b[0] + next_b[2] + epsilon
            if cm < lower_range or cm > upper_range:
                return False  # CM(prefix of stack) isn't in the support of the block below

        return True

    def validate_ys(self, blks):
        if self.bb_ordering != 'position':
            blks = self.sort_ys(blks)

        epsilon = 15
        for j in range(len(blks))[::-1]:  # blocks from top to bottom
            b = blks[j]
            next_b = blks[j-1] if j > 0 else th.tensor([0, 1, 1, 0.1]) * 640  # fake ground block
            curr_blk_y = b[1] + b[3]
            next_b_y = next_b[1]
            lower_range = next_b_y - epsilon
            upper_range = next_b_y + epsilon
            if curr_blk_y < lower_range or curr_blk_y > upper_range:
                return False
        return True

    def validate_areas(self, blks):
        areas = blks[:, 2] * blks[:, 3]
        sorted = th.sort(areas).indices
        return (sorted == th.arange(len(sorted)).to(sorted.device)).all().item()

    def sort_ys(self, blks):
        blks = blks.clone()
        sorted_ixs = th.sort(blks[:, 1], descending=True).indices
        return blks[sorted_ixs]

class SampleVisualizationCallback(SingleEvaluatorCallback):
    def __init__(self, cfg, freq):
        self.freq = freq
        self.train_evaluator = hist.SampleVisualizer(cfg, save_file='train_bb_vis_.jpg')
        self.valid_evaluator = hist.SampleVisualizer(cfg, save_file='valid_bb_vis.jpg')


class ForwardProcessVisCallback(Callback):
    def __init__(self, cfg, base_dir):
        self.loader = utils.initialize_loader(
            cfg, 'val', 15, pin_memory=False, drop_last=True)
        self.base_dir = base_dir
        self.forward_vis = hist.ForwardProcessVis(cfg, self.loader)

    def on_train_start(self, trainer, model):
        self.forward_vis.evaluate(
            model.g_diffusion, model.g_diffusion.beta_scheduler, model.bbox_transformer,
            self.base_dir)
        del self.forward_vis
        del self.loader


def log_results(trainer, means, model=None):
    epoch = trainer.current_epoch + 1 if trainer.has_training_started else 0
    pprint_results(means)
    trainer.helper.save_results(means, iteration=epoch)


def pprint_results(dct):
    res = json.dumps(dct, indent=4, sort_keys=True)
    logger.info(res)
    print(res)


def get_save_dir(base_dir, trainer):
    epoch = trainer.current_epoch + 1 if trainer.has_training_started else 0
    dir = base_dir + '/epoch_{:05d}'.format(epoch)
    os.makedirs(dir, exist_ok=True)
    return dir
