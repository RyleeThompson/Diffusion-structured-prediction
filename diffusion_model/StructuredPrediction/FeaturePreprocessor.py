import torch as th
import pytorch_lightning as pl
import torch.nn as nn
from diffusion_model.model_utils import TimeConditionedMLP
from utils.data.BBNormalizer import create_bbox_transformer
from utils.data.coco_dataset import coco_num_classes
from utils import class_formatting as cls


def get_feat_preprocessor(cfg):
    return Joiner(cfg)


class Joiner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.x_t_preprocessor = XTBBPreprocessor(cfg)
        self.bbone_preprocessor = BBoneBBPreprocessor(cfg)
        self.img_feats_preprocessor = ImgFeatPreprocessor(cfg)
        # self.preprocessors = [self.x_t_preprocessor, self.bb_pred_preprocessor, self.img_feats_preprocessor]
        model_cfg = cfg['model']['structured']
        # concat_seqs = model_cfg['conditioning'].get('concat_gt', False)
        backbone = cfg['model']['backbone']['name']
        # if concat_seqs and backbone != 'gt':
        #     raise Exception('Concat seqs without gt not yet implemented')

        # if concat_seqs:
        #     self.forward = self.forward_concat_seqs
        # else:
        #     self.forward = self.forward_no_concat

    def forward(self, *args, **kwargs):
        ret_dct = {}
        for mod in [self.x_t_preprocessor, self.bbone_preprocessor, self.img_feats_preprocessor]:
            ret_dct.update(mod(*args, **kwargs))
        return ret_dct

    # def forward_concat_seqs(self, *args, **kwargs):
    #     no_concat = self.forward_no_concat(*args, **kwargs)
    #     x_t_bb_feats = no_concat['x_t_bb_feats']
    #     bbone_bb_feats = no_concat['bbone_bb_feats']
    #     assert bbone_bb_feats is not None
    #     assert bbone_bb_feats.shape == x_t_bb_feats.shape
    #
    #     with_concat = {}
    #     with_concat['x_t_bb_feats'] = th.cat([x_t_bb_feats, bbone_bb_feats], dim=-1)
    #     with_concat['bbone_bb_feats'] = None
    #     with_concat['img_feats_preprocessed'] = no_concat['img_feats_preprocessed']
    #     return with_concat


class Preprocessor(pl.LightningModule):
    def get_output_dim(self, cfg, img_feats=False):
        model_cfg = cfg['model']['structured']
        d_model = model_cfg['d_model']
        if 'transformer' in model_cfg['name']:
            if model_cfg['timestep_conditioning']['name'] != 'cat':
                out_dim = d_model
            else:
                assert d_model % 2 == 0
                out_dim = d_model // 2
            concat_gt = model_cfg['conditioning']['concat_method'] == 'feats'
            if concat_gt and img_feats is False:
                assert out_dim % 2 == 0
                out_dim = out_dim // 2
        else:
            out_dim = d_model
        return out_dim


class BBPreprocessor(pl.LightningModule):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()

        conditioning_type = cfg['model']['structured']['conditioning']['name']
        img_feat_cond_type = cfg['model']['structured']['conditioning'].get('method')
        print(conditioning_type, img_feat_cond_type)
        if conditioning_type in ['bb_preds', 'x_t'] or img_feat_cond_type in ['seq']:
            self.preprocess = self.preprocess_bbs_info
        else:
            self.preprocess = self.preprocess_bbs_info_and_feats
            in_dim += 12544
        # else:
            # raise Exception(conditioning_type, img_feat_cond_type)
            # in_dim += 1024
        # elif conditioning_type == 'h(x)' and img_feat_cond_type in ['both', 'roi']:
        #     self.preprocess = self.preprocess_bbs_feats
        #     in_dim = 12544
            # in_dim = 1024

        model_cfg = cfg['model']['structured']
        preprocessing_mlp_cfg = model_cfg['mlp']
        timestep_conditioning = preprocessing_mlp_cfg['timestep_conditioning']
        num_timesteps = cfg['diffusion']['num_timesteps']

        self.preprocessing_mlp = TimeConditionedMLP(
            preprocessing_mlp_cfg['num_layers'], in_dim,
            out_dim, out_dim, timestep_conditioning,
            time_encode_dim=out_dim, num_timesteps=num_timesteps,
            relu_output=True, rescale_timesteps=cfg['model']['rescale_timesteps'])

    def preprocess_bbs_info(self, bbs_info, *args, **kwargs):
        return bbs_info

    def preprocess_bbs_info_and_feats(self, bbs_info, bbs_feats, *args, **kwargs):
        return th.cat([bbs_info, bbs_feats], dim=-1)

    def preprocess_bbs_feats(self, bbs_info, bbs_feats, *args, **kwargs):
        return bbs_feats

    def forward(self, bbs_info, bbs_feats, t):
        x = self.preprocess(bbs_info, bbs_feats)
        return self.preprocessing_mlp(x, t)


class XTBBPreprocessor(Preprocessor):
    def __init__(self, cfg):
        super().__init__()
        out_dim = self.get_output_dim(cfg)
        in_dim = 4
        self.predict_class = cfg['model']['predict_class']
        self.train_cls_fmt = cfg['model']['class_fmt']
        if self.predict_class:
            dataset = cfg['dataset']
            num_classes = cfg['max_num_blocks'] + 1 if dataset == 'tower' else coco_num_classes
            if self.train_cls_fmt == 'bits':
                in_dim += int(cls.num_classes2num_bits(num_classes))
            else:
                in_dim += num_classes
        self.x_t_preprocessor = BBPreprocessor(cfg, in_dim=in_dim, out_dim=out_dim)

    def forward(self, x_t_bbs, t, bbone_res, *args, **kwargs):
        if self.predict_class:
            cls_fmt = self.train_cls_fmt
        else:
            cls_fmt = None
        x_t_bbs = x_t_bbs.get_features(cls_fmt=cls_fmt)

        return {'x_t_bb_feats': self.x_t_preprocessor(x_t_bbs, bbone_res.get('x_t_feats'), t)}


class BBoneBBPreprocessor(Preprocessor):
    def __init__(self, cfg):
        super().__init__()
        self.dataset = cfg['dataset']
        out_dim = self.get_output_dim(cfg)

        conditioning = cfg['model']['structured']['conditioning']['name']
        if conditioning in ['bb_preds', 'both']:
            self.initialize_bbone_pred_processor(cfg, out_dim)
        else:
            self.bb_pred_preprocessor = self.do_nothing

    def initialize_bbone_pred_processor(self, cfg, out_dim):
        if self.dataset == 'coco':
            num_classes = 81 if cfg['model']['backbone']['name'] != 'gt' else coco_num_classes
        elif self.dataset == 'tower':
            num_classes = cfg['max_num_blocks'] + 1

        self.bb_pred_preprocessor = BBPreprocessor(
            cfg, in_dim=4 + num_classes, out_dim=out_dim)
        # self.bb_pred_normalizer = BBNormalizer(cfg)

    def do_nothing(self, *args, **kwargs):
        return None

    def forward(self, bbone_res, t, *args, **kwargs):
        bbone_preds = bbone_res['bbone_preds']
        # bbone_preds = th.cat([bbone_preds['bbox'], bbone_preds['classes_softmax']], dim=-1)
        if bbone_preds is not None:
            bbone_preds = bbone_preds.get_features(cls_fmt='softmax')
        bbone_bb_feats = self.bb_pred_preprocessor(bbone_preds, bbone_res.get('bbone_pred_feats'), t)
        return {'bbone_bb_feats': bbone_bb_feats}


class ImgFeatPreprocessor(Preprocessor):
    def __init__(self, cfg):
        super().__init__()
        conditioning_type = cfg['model']['structured']['conditioning']['name']
        img_feat_cond_type = cfg['model']['structured']['conditioning'].get('method')

        if conditioning_type in ['x_t', 'bb_preds'] or img_feat_cond_type == 'roi':
            self.forward = self.do_nothing
        else:
            model_cfg = cfg['model']['structured']
            preprocessing_mlp_cfg = model_cfg['mlp']
            timestep_conditioning = preprocessing_mlp_cfg['timestep_conditioning']
            num_timesteps = cfg['diffusion']['num_timesteps']
            out_dim = self.get_output_dim(cfg, img_feats=True)
            in_dim = 256
            self.preprocessing_mlp = TimeConditionedMLP(
                preprocessing_mlp_cfg['num_layers'], in_dim,
                out_dim, out_dim, timestep_conditioning,
                time_encode_dim=out_dim, num_timesteps=num_timesteps,
                relu_output=True, rescale_timesteps=cfg['model']['rescale_timesteps'])
            self.forward = self.project_to_model_d

    def do_nothing(self, *args, **kwargs):
        return {'img_feats_preprocessed': None}

    def project_to_model_d(self, bbone_res, t, *args, **kwargs):
        img_feats = bbone_res['img_feats']['p5']
        img_feats = img_feats.permute(0, 2, 3, 1)
        img_feats_preprocessed = self.preprocessing_mlp(img_feats, t)
        return {'img_feats_preprocessed': img_feats_preprocessed}
