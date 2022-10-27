import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from diffusion_model.StructuredPrediction.FeaturePreprocessor import get_feat_preprocessor
from diffusion_model.StructuredPrediction.SeqMaskPreprocessor import get_seq_mask_preprocessor


def get_transformer(cfg):
    struc_cfg = cfg['model']['structured']
    if struc_cfg['name'] == 'transformer':
        return FullTransformer(cfg, **cfg['model']['structured'])
    elif struc_cfg['name'] == 'transformer-encoder':
        return TransformerEncoder(cfg, **cfg['model']['structured'])
    elif struc_cfg['name'] == 'transformer-decoder':
        return TransformerDecoder(cfg, **cfg['model']['structured'])
    else:
        raise NotImplementedError(struc_cfg['name'])


class TransformerBase(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.preprocessor = get_feat_preprocessor(cfg)
        self.seq_mask_preprocessor = get_seq_mask_preprocessor(cfg)
        self.max_num_preds = cfg['model']['max_num_preds'] if cfg['dataset'] == 'coco' else cfg['max_num_blocks']
        self.predict_class = cfg['model']['predict_class']
        self.attend_padded_conditioning = cfg['model'].get('attend_padding_conditioning', False)

    def preprocess(self, x_t, t, bbone_res):
        preprocessed_feats = self.preprocess_feats(x_t, t, bbone_res)
        if self.predict_class:
            x_t_mask = th.tensor([False]).expand(x_t.shape[0], x_t.shape[1]).to(x_t.device)
        else:
            x_t_mask = th.logical_not(x_t['padding_mask'])

        if bbone_res['bbone_preds'] is not None:
            if self.attend_padded_conditioning:
                preds = bbone_res['bbone_preds']
                bbone_mask = th.tensor([False]).expand(preds.shape[0], preds.shape[1]).to(x_t.device)
            else:
                bbone_mask = th.logical_not(bbone_res['bbone_preds']['padding_mask'])
        else:
            bbone_mask = None
        preprocessed_seqs_masks = self.seq_mask_preprocessor(
            preprocessed_feats, t, x_t_mask=x_t_mask, bbone_mask=bbone_mask)
        return preprocessed_seqs_masks

    def preprocess_feats(self, x_t, t, bbone_res):
        # import ipdb; ipdb.set_trace()
        preprocessed_feats = self.preprocessor(
            x_t_bbs=x_t, bbone_res=bbone_res, t=t)
        return preprocessed_feats

    def postprocess_preds(self, preds):
        return preds[:, -self.max_num_preds:]

    def forward(self, x_t, t, bbone_res):
        preprocessed_seqs_masks = self.preprocess(x_t, t, bbone_res)
        preds = self.model(**preprocessed_seqs_masks)
        return self.postprocess_preds(preds)


class FullTransformer(TransformerBase):
    def __init__(self, cfg, d_model=64, nhead=4, dim_feedforward=64,
                 dropout=0.1, num_layers=4, use_class_input=False, **kwargs):
        super().__init__(cfg)
        self.model = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )


class TransformerEncoder(TransformerBase):
    def __init__(self, cfg, d_model=64, nhead=4, dim_feedforward=64,
                 dropout=0.1, num_layers=4, use_class_input=False, **kwargs):
        super().__init__(cfg)
        trans_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.model = nn.TransformerEncoder(
            trans_layer, num_layers=num_layers)


class TransformerDecoder(TransformerBase):
    def __init__(self, cfg, d_model=64, nhead=4, dim_feedforward=64,
                 dropout=0.1, num_layers=4, use_class_input=False, **kwargs):
        super().__init__(cfg)
        trans_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.model = nn.TransformerDecoder(
            trans_layer, num_layers=num_layers)
