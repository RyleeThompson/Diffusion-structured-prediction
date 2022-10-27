from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import os
import pickle
import torch as th

class MMDEvaluation():
    def __init__(self, kernel='rbf', sigma='range', multiplier='median'):
        if multiplier == 'mean':
            self.__get_sigma_mult_factor = self.mean_pairwise_distance
        elif multiplier == 'median':
            self.__get_sigma_mult_factor = self.median_pairwise_distance
        elif multiplier is None:
            self.__get_sigma_mult_factor = lambda *args, **kwargs: 1
        else:
            raise Exception(multiplier)

        if 'rbf' in kernel:
            if sigma == 'range':
                self.base_sigmas = np.array([
                    0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_adaptive_median'
                else:
                    self.name = 'mmd_rbf_adaptive'

            elif sigma == 'one':
                self.base_sigmas = np.array([1])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf_single_mean'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_single_median'
                else:
                    self.name = 'mmd_rbf_single'

            else:
                raise Exception(sigma)

            self.evaluate = self.calculate_MMD_rbf_quadratic

        elif 'linear' in kernel:
            self.evaluate = self.calculate_MMD_linear_kernel

        else:
            raise Exception()

    def __get_pairwise_distances(self, generated_dataset, reference_dataset):
        return pairwise_distances(
            reference_dataset, generated_dataset,
            metric='euclidean', n_jobs=8) ** 2

    def mean_pairwise_distance(self, dists_GR):
        return np.sqrt(dists_GR.mean())

    def median_pairwise_distance(self, dists_GR):
        return np.sqrt(np.median(dists_GR))

    def get_sigmas(self, dists_GR):
        mult_factor = self.__get_sigma_mult_factor(dists_GR)
        return self.base_sigmas * mult_factor

    def calculate_MMD_rbf_quadratic(self, generated_dataset, reference_dataset):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        if generated_dataset.isnan().any() or reference_dataset.isnan().any():
            return np.nan

        generated_dataset = generated_dataset.cpu()
        reference_dataset = reference_dataset.cpu()
        GG = self.__get_pairwise_distances(generated_dataset, generated_dataset)
        GR = self.__get_pairwise_distances(generated_dataset, reference_dataset)
        RR = self.__get_pairwise_distances(reference_dataset, reference_dataset)

        max_mmd = 0
        sigmas = self.get_sigmas(GR)
        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - 2 * K_GR.mean()
            max_mmd = mmd if mmd > max_mmd else max_mmd

        return max_mmd


class HistogramEvaluation():
    def __init__(self, set='train', num_eval_samples=5000):
        self.mmd_eval = MMDEvaluation()
        self.set = set
        self.num_eval_samples = num_eval_samples

    def evaluate(self, model_bbs, save_dir):
        avg_num_bbs = np.mean([bb.shape[0] for bb in model_bbs])
        if avg_num_bbs == 0:
            keys = ['MMD heights', 'MMD joint (agg)', 'MMD joint (no agg)',
                    "MMD widths", "MMD xs", "MMD ys"]
            return {key: 0 for key in keys}
        coco_dset_bbs = self.load_dset(self.set)
        model_cat_bbs, model_agg_bbs = self.convert_bb_to_th(model_bbs)
        model_bbs = self.results_to_dct(model_cat_bbs, model_agg_bbs)

        fig, axs = plt.subplots(1, 4, figsize=(20, 3.5), sharey='row')
        self.__plot_result_hists(coco_dset_bbs, model_bbs, axs)
        joint_res, marginal_res = self.eval_joint_and_marginals(
            coco_dset_bbs, model_bbs, self.num_eval_samples)

        fontsize = 12
        for (key, val), ax in zip(marginal_res.items(), axs):
            ax.annotate('{}: {:.5f}'.format(key, val), (0,0), (75, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=fontsize)

        for ix, (key, val) in enumerate(joint_res.items()):
            axs[1].annotate('{}: {:.5f}'.format(key, val), (0, 0), (200, -40 - (20 * (ix + 1))), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=fontsize)
        plt.savefig(save_dir + f'/{self.set}_hists.jpg', bbox_inches='tight')

        joint_res.update(marginal_res)
        return joint_res

    def __plot_result_hists(self, coco_dset_bbs, model_bbs, axs):
        dset_to_plot = coco_dset_bbs['to_plot']
        model_to_plot = model_bbs['to_plot']
        for (key, dset_val), model_val, ax in zip(dset_to_plot.items(), model_to_plot.values(), axs):
            self.__plot_single_result_hist(key, dset_val, model_val, ax, labels=[self.set.title(), 'Model'])
        axs[0].legend()

    def __plot_single_result_hist(self, key, dset_val, model_val, ax, labels):
        title = key.title()
        ax.set_title(title)
        dset_w = np.ones(len(dset_val)) / len(dset_val)
        model_w = np.ones(len(model_val)) / len(model_val)

        ax.hist([dset_val.cpu(), model_val.cpu()], weights=[dset_w, model_w], label=labels)

    def eval_joint_and_marginals(self, coco_dset_bbs, model_bbs, num_samples):
        marginal_res = self.eval_marginals(coco_dset_bbs, model_bbs, num_samples)
        joint_res = self.eval_joints(coco_dset_bbs, model_bbs, num_samples)
        return joint_res, marginal_res

    def eval_marginals(self, coco_dset_bbs, model_bbs, num_samples):
        results = {'MMD xs': 0, 'MMD ys': 0, 'MMD widths': 0, 'MMD heights': 0}
        dset_results = self.__select_random_ixs(coco_dset_bbs['bboxes_cat'], num_samples)
        model_results = self.__select_random_ixs(model_bbs['bboxes_cat'], num_samples)

        for ix, key in enumerate(results.keys()):
            mmd = self.mmd_eval.evaluate(dset_results[:, ix].unsqueeze(1), model_results[:, ix].unsqueeze(1))
            results[key] = mmd
        return results

    def eval_joints(self, coco_dset_bbs, model_bbs, num_samples):
        dset_results = self.__select_random_ixs(coco_dset_bbs['bboxes_cat'], num_samples)
        model_results = self.__select_random_ixs(model_bbs['bboxes_cat'], num_samples)
        joint_no_agg_mmd = self.mmd_eval.evaluate(dset_results, model_results)

        dset_results = self.__select_random_ixs(coco_dset_bbs['bboxes_agg'], num_samples)
        model_results = self.__select_random_ixs(model_bbs['bboxes_agg'], num_samples)
        joint_agg_mmd = self.mmd_eval.evaluate(dset_results, model_results)
        return {'MMD joint (no agg)': joint_no_agg_mmd, 'MMD joint (agg)': joint_agg_mmd}

    def __select_random_ixs(self, to_select_from, num_samples):
        if num_samples >= len(to_select_from):
            return to_select_from

        ixs = np.random.choice(num_samples, replace=False, size=num_samples)
        return to_select_from[ixs]

    def results_to_dct(self, results_cat, results_agg):
        results = {'to_plot': {}}
        to_plot = results['to_plot']
        to_plot['Bbox xs'] = results_cat[:, 0]
        to_plot['Bbox ys'] = results_cat[:, 1]
        to_plot['Bbox widths'] = results_cat[:, 2]
        to_plot['Bbox heights'] = results_cat[:, 3]

        results['bboxes_cat'] = results_cat
        results['bboxes_agg'] = results_agg
        return results

    def load_dset(self, set='train'):
        coco_dset = pickle.load(open('dataset/coco-data.h5', 'rb'))[set]
        dset_cat_bbs, dset_agg_bbs = self.convert_bb_to_th(coco_dset['bboxes'])
        return self.results_to_dct(dset_cat_bbs, dset_agg_bbs)

    def convert_bb_to_th(self, bboxes):
        bboxes_agg = []
        bboxes_cat = []
        for ix, bb in enumerate(bboxes):
            if not isinstance(bb, th.Tensor):
                bb = th.tensor(bb)
            bboxes_cat.append(bb)
            if len(bb) > 0:
                bboxes_agg.append(bb.sum(dim=0).unsqueeze(0))

        bboxes_cat = th.cat(bboxes_cat, dim=0)
        bboxes_agg = th.cat(bboxes_agg, dim=0)
        return bboxes_cat, bboxes_agg


class SampleVisualizer():
    def __init__(self, cfg, save_file='bb_vis.jpg'):
        self.dataset = cfg['dataset']
        self.ordering = cfg['bb_ordering']
        self.conditioning = cfg['model']['structured']['conditioning']['name']
        self.max_num_blocks = cfg['max_num_blocks']
        self.save_file = save_file

    def get_color(self, bb_class):
        if self.dataset == 'tower' and self.ordering != 'none':
            cmap = matplotlib.cm.get_cmap('Reds')
            return cmap(bb_class / self.max_num_blocks)
        else:
            return 'r'

    def evaluate(self, evaluator_in, save_dir):
        all_x_t_bbs = evaluator_in['all_x_t']
        all_x_t_cls = evaluator_in['x_t_cls_int_preds']
        timesteps = np.flip(evaluator_in['ts']).tolist()
        sample_ixs = np.random.choice(len(all_x_t_bbs[0]), size=min(5, len(all_x_t_bbs[0])), replace=False)

        # fig, axs = plt.subplots(len(sample_ixs), len(step_ixs), figsize=(20, 10))
        num_rows = len(all_x_t_bbs)
        if 'x_start' in evaluator_in:
            num_rows += 1
            timesteps += ['GT']
            all_x_t_bbs.append(evaluator_in['x_start'])
            all_x_t_cls.append(evaluator_in['x_start_cls_int'])

        if self.conditioning in ['both', 'bb_preds'] and 'bbone_preds' in evaluator_in:
            num_rows += 1
            timesteps += ['Conditioning']
            all_x_t_bbs.append(evaluator_in['bbone_preds'])
            all_x_t_cls.append(evaluator_in['bbone_cls_int'])

        fig, axs = plt.subplots(len(sample_ixs), num_rows, figsize=(20, 10))

        axs = np.swapaxes(axs, 1, 0)
        color = 'r'
        for axs_, t, step_bbs, step_cls in zip(axs, timesteps, all_x_t_bbs, all_x_t_cls):
            # step_bbs = all_x_t_bbs[step_ix]
            for ix, (ax, sample_ix) in enumerate(zip(axs_, sample_ixs)):
                if 'images' in evaluator_in and evaluator_in['images'][0] is not None:
                    img = evaluator_in['images'][sample_ix]
                    img = img.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0)).astype(np.int32)
                    img = img[:, ..., ::-1]
                    ax.imshow(img)
                else:
                    ax.imshow(np.zeros((640, 640)))
                bbs = step_bbs[sample_ix]
                # classes = evaluator_in['final_cls_preds'][sample_ix]
                classes = step_cls[sample_ix]
                for bb, bb_class in zip(bbs, classes):
                    x, y, width, height = bb.cpu()
                    color = self.get_color(bb_class.item())
                    # patches.append(Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none'))
                    # patches.append(Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor=color, fill=True, alpha=0.2))
                    ax.add_patch(Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none'))
                    ax.add_patch(Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor=color, fill=True, alpha=0.2))

                # pc = PatchCollection(patches, cmap='Reds')
                # pc.set_array(th.repeat_interleave(classes, 2, dim=0).nonzero()[:, 1].cpu().numpy())
                # ax.add_collection(pc)

                ax.axis('off')
                if t == timesteps[0] and sample_ix == sample_ixs[0]:
                    ax.set_title('$t = T$')
                elif sample_ix == sample_ixs[0]:
                    if type(t) == int:
                        ax.set_title('$t = {}$'.format(t))
                    else:
                        ax.set_title(t)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir + '/' + self.save_file, bbox_inches='tight')
            plt.close('all')


class ForwardProcessVis():
    def __init__(self, cfg, data_loader):
        self.loader = data_loader
        self.sample_vis = SampleVisualizer(cfg, save_file='forward_vis.jpg')

    def evaluate(self, g_diffusion, beta_scheduler, bbox_transformer, save_dir=None):
        batch = next(iter(self.loader))
        ts = np.linspace(
            0, beta_scheduler.num_timesteps - 1,
            num=min(10, beta_scheduler.num_timesteps)).astype(np.int32)

        ts = th.from_numpy(ts).to(g_diffusion.device)
        batch['x_start'] = batch['x_start'].to(g_diffusion.device)

        # keep_masks = batch['x_start']['padding_mask']
        # start_bbs_no_padding = self.add_bbs_to_list(
            # batch['x_start'], keep_masks, bbox_transformer)
        # x_ts = [start_bbs_no_padding]
        # import ipdb; ipdb.set_trace()
        x_start_unmasked = batch['x_start'].inverse_normalize().to_xywh().unmask()
        x_t_bbs = [x_start_unmasked['bbox']]
        x_t_cls = [x_start_unmasked['classes']]
        # classes = batch['x_start'].unmask(keys=['classes_softmax'])['classes_softmax']
        # classes = [classes_[:num_bbs] for classes_, num_bbs in zip(batch['classes'], batch['num_bbs'])]
        for t in ts[1:]:
            x_t = g_diffusion.sample(batch['x_start'], ts=t)['x_t']
            x_t_unmasked = x_t.inverse_normalize().to_xywh().unmask(use_cls=True)
            x_t_bbs.append(x_t_unmasked['bbox'])
            x_t_cls.append(x_t_unmasked['classes'])
            # x_t_no_padding = self.add_bbs_to_list(x_t, keep_masks, bbox_transformer)
            # x_ts.append(x_t_no_padding)

        # res = {'all_x_t': list(reversed(x_ts)), 'ts': ts.cpu().numpy(), 'final_cls_preds': classes}
        res = {'all_x_t': list(reversed(x_t_bbs)),
               'x_t_cls_int_preds': list(reversed(x_t_cls)),
               'ts': ts.cpu().numpy()}
        self.sample_vis.evaluate(res, save_dir)

    def add_bbs_to_list(self, bbs, masks, bb_transformer):
        num_bbs = masks.sum(dim=-1)
        bbs_to_add = []
        bbs = bbs.inverse_normalize()
        for bb, bb_count in zip(bbs['bbox'], num_bbs):
            bbs_to_add.append(bb[:bb_count])
        return bbs_to_add
