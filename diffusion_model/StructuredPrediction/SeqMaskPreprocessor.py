import torch as th
import torch.nn as nn
from diffusion_model.model_utils import TimeStepEncoding, PositionalEncoding2D
import pytorch_lightning as pl


def get_seq_mask_preprocessor(cfg):
    return SeqMaskPreprocessor(cfg)
    # model_cfg = cfg['model']['structured']
    # if model_cfg['name'] == 'transformer':
    #     if model_cfg['use_learned_tgt_embeds']:
    #         embed_prep = FullTransformerLearnedEmbedsPreprocessor(
    #             cfg, d_model=model_cfg['d_model'])
    #     elif not model_cfg['use_learned_tgt_embeds']:
    #         embed_prep = FullTransformerNoLearnedEmbedsPreprocessor()
    #     else:
    #         raise RuntimeError(model_cfg['use_learned_tgt_embeds'])
    #
    #     if model_cfg['timestep_conditioning'] == 'node':
    #         time_cond_prep = FullTransformerNodeConditioningPreprocessor(
    #             d_model=model_cfg['d_model'])
    #     elif model_cfg['timestep_conditioning'] == 'cat':
    #         time_cond_prep = FullTransformerCatConditioningPreprocessor(
    #             d_model=model_cfg['d_model'])
    #     elif model_cfg['timestep_conditioning'] == 'none':
    #         time_cond_prep = FullTransformerNoConditioningPreprocessor()
    #     else:
    #         raise RuntimeError(model_cfg['timestep_conditioning'])
    #
    #     return FullTransformerJoiner(embed_prep, time_cond_prep)
    #
    # elif model_cfg['name'] == 'transformer-encoder':
    #     if model_cfg['timestep_conditioning'] == 'node':
    #         return TransformerEncoderNodeConditioningPreprocessor(
    #             cfg=cfg, d_model=model_cfg['d_model'])
    #     elif model_cfg['timestep_conditioning'] == 'cat':
    #         return TransformerEncoderCatConditioningPreprocessor(
    #             cfg=cfg, d_model=model_cfg['d_model'])
    #     elif model_cfg['timestep_conditioning'] == 'none':
    #         return TransformerEncoderNoConditioningPreprocessor(cfg=cfg)
    #     else:
    #         raise RuntimeError(model_cfg['timestep_conditioning'])
    #
    # elif model_cfg['name'] == 'transformer-decoder':
    #     if model_cfg['timestep_conditioning'] == 'node':
    #         return TransformerDecoderNodeConditioningPreprocessor(
    #             d_model=model_cfg['d_model'])
    #     elif model_cfg['timestep_conditioning'] == 'cat':
    #         return TransformerDecoderCatConditioningPreprocessor(
    #             d_model=model_cfg['d_model'])
    #     elif model_cfg['timestep_conditioning'] == 'none':
    #         return TransformerDecoderNoConditioningPreprocessor()
    #     else:
    #         raise RuntimeError(model_cfg['timestep_conditioning'])
    #
    # else:
    #     raise NotImplementedError(model_cfg['name'])


class SeqEmbedModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        out_dim = cfg['model']['structured']['d_model']
        if cfg['model']['structured']['timestep_conditioning']['name'] == 'cat':
            out_dim = out_dim // 2

        self.time_embed = SeqTimeEmbedModule(cfg, out_dim)
        self.learned_embed = SeqLearnedEmbedModule(cfg, out_dim)

    def forward(self, seqs, ts):
        self.time_embed(seqs, ts)
        self.learned_embed(seqs)


class SeqLearnedEmbedModule(pl.LightningModule):
    def __init__(self, cfg, out_dim):
        super().__init__()
        if cfg['model']['use_learned_embeds']:
            self.learned_embeds = nn.Embedding(3, out_dim).weight
        else:
            self.register_buffer(
                'learned_embeds',
                th.zeros((3, out_dim))
            )

    def forward(self, seqs):
        for seq_ix, (key, seq) in enumerate(seqs.items()):
            if seq is not None:
                seqs[key] = seqs[key] + self.learned_embeds[seq_ix]


class SeqTimeEmbedModule(pl.LightningModule):
    def __init__(self, cfg, out_dim):
        super().__init__()

        cond_method = cfg['model']['structured']['conditioning'].get('method')
        if cond_method == 'both' or cond_method == 'seq':
            self.img_feat_time_embed_fn = PositionalEncoding2D(out_dim)

        use_time_embeds = cfg['model'].get('use_time_embeds')
        if use_time_embeds is None or use_time_embeds is False:
            self.get_and_add_bb_time_embed = self.do_nothing
            self.learned_embed = False
        elif use_time_embeds is True or use_time_embeds == 'separate':
            self.get_and_add_bb_time_embed = self.add_bb_time_embeds_separate
            self.time_embed = TimeStepEncoding(out_dim)
            self.learned_embed = False
        elif use_time_embeds == 'combined':
            self.get_and_add_bb_time_embed = self.add_bb_time_embeds_combined
            self.time_embed = TimeStepEncoding(out_dim)
            self.learned_embed = False
        elif use_time_embeds == 'learned_1d' or use_time_embeds == 'learned_2d':
            self.get_and_add_bb_time_embed = self.add_bb_time_embeds_combined
            self.time_embed = LearnedTimeEmbedEncoding(cfg, out_dim)
            self.learned_embed = True

    def do_nothing(self, seqs, *args, **kwargs):
        return seqs

    def get_bb_time_embeds(self, seq, start_t=0, step_ts=None):
        end_t = start_t + seq.shape[1]
        ts = th.arange(start_t, end_t)
        if not isinstance(self.time_embed, LearnedTimeEmbedEncoding):
            time_embeds = self.time_embed(ts)
        else:
            time_embeds = self.time_embed(ts, step_ts=step_ts)

        if len(time_embeds.shape) != len(seq.shape):
            time_embeds = time_embeds.unsqueeze(0).expand(seq.shape[0], -1, -1)
        return time_embeds

    def get_and_add_img_feat_time_embed(self, seqs):
        img_feat_seq = seqs['img_feats_preprocessed']
        if img_feat_seq is not None:
            self.get_and_add_time_embeds(
                seqs, 'img_feats_preprocessed', self.img_feat_time_embed_fn)
            bb_start_t = max(img_feat_seq.shape[1], img_feat_seq.shape[2])
        else:
            bb_start_t = 0

        return bb_start_t

    def get_and_add_time_embeds(self, seqs, key, embed_fn, *args, **kwargs):
        seq = seqs[key]
        embeds = embed_fn(seq, *args, **kwargs)
        seq = seq + embeds
        seqs[key] = seq

    def add_bb_time_embeds_separate(self, seqs, start_t=0, step_ts=None):
        if seqs['bbone_bb_feats'] is not None:
            self.get_and_add_time_embeds(
                seqs, 'bbone_bb_feats', self.get_bb_time_embeds,
                start_t=start_t, step_ts=step_ts
            )

        if seqs['x_t_bb_feats'] is not None:
            self.get_and_add_time_embeds(
                seqs, 'x_t_bb_feats', self.get_bb_time_embeds,
                start_t=start_t, step_ts=step_ts
            )

    def add_bb_time_embeds_combined(self, seqs, start_t=0, step_ts=None):
        if seqs['bbone_bb_feats'] is not None:
            self.get_and_add_time_embeds(
                seqs, 'bbone_bb_feats', self.get_bb_time_embeds,
                start_t=start_t, step_ts=step_ts
            )
            start_t += seqs['bbone_bb_feats'].shape[1]

        if seqs['x_t_bb_feats'] is not None:
            self.get_and_add_time_embeds(
                seqs, 'x_t_bb_feats', self.get_bb_time_embeds,
                start_t=start_t, step_ts=step_ts
            )

    def forward(self, seqs, ts):
        bb_seq_start_t = self.get_and_add_img_feat_time_embed(seqs)
        if self.learned_embed:
            bb_seq_start_t = 0
        self.get_and_add_bb_time_embed(seqs, start_t=bb_seq_start_t, step_ts=ts)


class LearnedTimeEmbedEncoding(pl.LightningModule):
    def __init__(self, cfg, out_dim):
        super().__init__()
        if cfg['model']['backbone']['name'] == 'gt':
            num_embeds = cfg['model']['max_num_preds'] * 2
        elif cfg['model']['structured']['conditioning'].get('top_k') is not None:
            topk = cfg['model']['structured']['conditioning']['top_k']
            topk = 300 if topk == -1 else topk
            num_embeds = cfg['model']['max_num_preds'] + topk
        else:
            num_embeds = cfg['model']['max_num_preds']

        if cfg['model']['use_time_embeds'] == 'learned_1d':
            self.embeds = nn.Embedding(num_embeds, out_dim).weight
            self.forward = self.forward_1d
        else:
            num_steps = cfg['diffusion']['num_timesteps']
            embeds = th.stack([nn.Embedding(num_embeds, out_dim).weight for _ in range(num_steps)])
            self.register_parameter('embeds', nn.Parameter(embeds))
            self.forward = self.forward_2d

    def forward_1d(self, ts, step_ts=None):
        return self.embeds[ts]

    def forward_2d(self, seq_ts, step_ts):
        return self.embeds[step_ts][:, seq_ts]

class SeqMaskPreprocessor(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.add_seq_embeds_fn = SeqEmbedModule(cfg)
        model_cfg = cfg['model']['structured']
        if model_cfg['name'] == 'transformer-encoder':
            # self.model_preprocessor = TransformerEncoderPreprocessor(cfg)
            self.forward = self.forward_encoder
        elif model_cfg['name'] == 'transformer':
            self.forward = self.forward_transformer
        elif model_cfg['name'] == 'transformer-decoder':
            self.forward = self.forward_decoder
        else:
            raise NotImplementedError('Havent modified decoder and transformer yet')

        self.concat_method = model_cfg['conditioning']['concat_method']

        tstep_cond = model_cfg['timestep_conditioning']
        num_timesteps = cfg['diffusion']['num_timesteps']
        if tstep_cond['name'] == 'node':
            self.time_preprocessor = NodeConditioningPreprocessor(
                d_model=model_cfg['d_model'], num_timesteps=num_timesteps,
                rescale_timesteps=cfg['model']['rescale_timesteps'], **tstep_cond)
        elif tstep_cond['name'] == 'cat':
            self.time_preprocessor = CatConditioningPreprocessor(
                d_model=model_cfg['d_model'], num_timesteps=num_timesteps,
                rescale_timesteps=cfg['model']['rescale_timesteps'], **tstep_cond)
        elif tstep_cond['name'] == 'none':
            self.time_preprocessor = NoConditioningPreprocessor()

        self.regress_from_gt = cfg['model']['regress_from'] == 'gt'

    def combine_bbone_and_padding_masks(self, bbone_mask, x_t_mask):
        combined_mask = th.cat([bbone_mask, x_t_mask], dim=-1)
        return combined_mask

    def combine_bbone_and_padding_masks_and_mask_out_bbone(self, bbone_mask, x_t_mask):
        bbone_pred_mask = th.tensor([[True]]).expand(bbone_mask.shape).to(x_t_mask.device)
        combined_mask = th.cat([bbone_pred_mask, x_t_mask], dim=-1)
        return combined_mask

    def forward_transformer(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        self.add_seq_embeds_fn(preprocessed_feats, t)
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']
        img_feats = preprocessed_feats['img_feats_preprocessed']
        if img_feats is not None:
            img_feats = img_feats.flatten(start_dim=1, end_dim=2)

        src = bbone_bb_feats
        tgt = x_t_bb_feats
        src_key_padding_mask = bbone_mask
        memory_key_padding_mask = bbone_mask
        tgt_key_padding_mask = x_t_mask

        if img_feats is not None and src is not None:
            src = th.cat([img_feats, src], dim=1)
            img_feat_mask = th.tensor([False]).expand(img_feats.shape[0], img_feats.shape[1]).to(img_feats.device)
            src_key_padding_mask = th.cat([img_feat_mask, src_key_padding_mask], dim=-1)
            memory_key_padding_mask = th.cat([img_feat_mask, memory_key_padding_mask], dim=-1)

        elif img_feats is not None and src is None:
            src = img_feats
            img_feat_mask = th.tensor([False]).expand(img_feats.shape[0], img_feats.shape[1]).to(img_feats.device)
            src_key_padding_mask = img_feat_mask
            memory_key_padding_mask = img_feat_mask

        src, src_key_padding_mask = \
            self.time_preprocessor.add_time_conditioning_seq_mask(
                src, t, src_key_padding_mask)
        tgt, tgt_key_padding_mask = \
            self.time_preprocessor.add_time_conditioning_seq_mask(
                tgt, t, tgt_key_padding_mask)
        memory_key_padding_mask = self.time_preprocessor.add_time_conditioning_mask(memory_key_padding_mask)

        return {'src': src,
                'tgt': tgt,
                'src_key_padding_mask': src_key_padding_mask,
                'tgt_key_padding_mask': tgt_key_padding_mask,
                'memory_key_padding_mask': memory_key_padding_mask}

    def forward_encoder(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        if self.concat_method == 'feats':
            preprocessed_feats['x_t_bb_feats'] = th.cat([
                preprocessed_feats['x_t_bb_feats'], preprocessed_feats['bbone_bb_feats']
            ], dim=-1)
            preprocessed_feats['bbone_bb_feats'] = None

        self.add_seq_embeds_fn(preprocessed_feats, t)
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']
        img_feats = preprocessed_feats['img_feats_preprocessed']
        if img_feats is not None:
            img_feats = img_feats.flatten(start_dim=1, end_dim=2)

        # src = th.cat([bbone_bb_feats, x_t_bb_feats], dim=0)
        if bbone_bb_feats is not None:
            if not self.regress_from_gt:
                src = th.cat([bbone_bb_feats, x_t_bb_feats], dim=1)
                src_key_padding_mask = \
                    self.combine_bbone_and_padding_masks(bbone_mask, x_t_mask)
            else:
                src = bbone_bb_feats
                src_key_padding_mask = bbone_mask
        else:
            src = x_t_bb_feats
            src_key_padding_mask = x_t_mask

        if img_feats is not None:
            src = th.cat([img_feats, src], dim=1)
            img_feat_mask = th.tensor([False]).expand(img_feats.shape[0], img_feats.shape[1]).to(img_feats.device)
            src_key_padding_mask = th.cat([img_feat_mask, src_key_padding_mask], dim=-1)

        src, src_key_padding_mask = \
            self.time_preprocessor.add_time_conditioning_seq_mask(
                src, t, src_key_padding_mask)

        return {'src': src,
                'src_key_padding_mask': src_key_padding_mask}

    def forward_decoder(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        self.add_seq_embeds_fn(preprocessed_feats, t)
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']
        img_feats = preprocessed_feats['img_feats_preprocessed']
        if img_feats is not None:
            img_feats = img_feats.flatten(start_dim=1, end_dim=2)

        memory = bbone_bb_feats
        memory_key_padding_mask = bbone_mask

        tgt = x_t_bb_feats
        tgt_key_padding_mask = x_t_mask

        if img_feats is not None and memory is not None:
            memory = th.cat([img_feats, memory], dim=1)
            img_feat_mask = th.tensor([False]).expand(img_feats.shape[0], img_feats.shape[1]).to(img_feats.device)
            memory_key_padding_mask = th.cat([img_feat_mask, memory_key_padding_mask], dim=-1)

        elif img_feats is not None and memory is None:
            memory = img_feats
            img_feat_mask = th.tensor([False]).expand(img_feats.shape[0], img_feats.shape[1]).to(img_feats.device)
            memory_key_padding_mask = img_feat_mask

        memory, memory_key_padding_mask = \
            self.time_preprocessor.add_time_conditioning_seq_mask(
                memory, t, memory_key_padding_mask)
        tgt, tgt_key_padding_mask = \
            self.time_preprocessor.add_time_conditioning_seq_mask(
                tgt, t, tgt_key_padding_mask)

        # memory, memory_key_padding_mask = \
        #     self.add_time_conditioning_seq_mask(
        #         tgt, t, tgt_key_padding_mask)

        return {'memory': memory,
                'memory_key_padding_mask': memory_key_padding_mask,
                'tgt': tgt,
                'tgt_key_padding_mask': tgt_key_padding_mask}


class FullTransformerPreprocessor(SeqMaskPreprocessor):
    def forward(self, src, tgt, t, src_key_padding_mask, tgt_key_padding_mask,
                memory_key_padding_mask):

        src, src_key_padding_mask = \
            self.add_time_conditioning_seq_mask(
                src, t, src_key_padding_mask)
        # tgt, tgt_key_padding_mask = \
        #     self.add_time_conditioning_seq_mask(
        #         tgt, t, tgt_key_padding_mask)
        memory_key_padding_mask = \
            self.add_time_conditioning_mask(
                memory_key_padding_mask)

        return {'src': src,
                'tgt': tgt,
                'src_key_padding_mask': src_key_padding_mask,
                'tgt_key_padding_mask': tgt_key_padding_mask,
                'memory_key_padding_mask': memory_key_padding_mask}


class TransformerEncoderPreprocessor(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.regress_from_gt = cfg['model']['regress_from'] == 'gt'

    def forward(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        self.add_seq_embeds_fn(preprocessed_feats, ts=t)
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']
        img_feats = preprocessed_feats['img_feats_preprocessed']

        # src = th.cat([bbone_bb_feats, x_t_bb_feats], dim=0)
        if bbone_bb_feats is not None:
            if not self.regress_from_gt:
                src = th.cat([bbone_bb_feats, x_t_bb_feats], dim=1)
                src_key_padding_mask = \
                    self.combine_bbone_and_padding_masks(bbone_mask, x_t_mask)
            else:
                src = bbone_bb_feats
                src_key_padding_mask = bbone_mask
        else:
            src = x_t_bb_feats
            src_key_padding_mask = x_t_mask

        if img_feats is not None:
            src = th.cat([img_feats, src], dim=1)
            img_feat_mask = th.tensor([False]).expand(img_feats.shape[0], img_feats.shape[1]).to(img_feats.device)
            src_key_padding_mask = th.cat([img_feat_mask, src_key_padding_mask], dim=-1)

        src, src_key_padding_mask = \
            self.add_time_conditioning_seq_mask(
                src, t, src_key_padding_mask)

        return {'src': src,
                'src_key_padding_mask': src_key_padding_mask}


class TransformerDecoderPreprocessor(SeqMaskPreprocessor):
    def forward(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']

        if bbone_bb_feats is None:
            raise Exception('Decoder with learned decoder tokens not implemented')

        memory = bbone_bb_feats
        memory_key_padding_mask = bbone_mask

        tgt = x_t_bb_feats
        tgt_key_padding_mask = x_t_mask

        tgt, tgt_key_padding_mask = \
            self.add_time_conditioning_seq_mask(
                tgt, t, tgt_key_padding_mask)
        # memory, memory_key_padding_mask = \
        #     self.add_time_conditioning_seq_mask(
        #         tgt, t, tgt_key_padding_mask)

        return {'memory': memory,
                'memory_key_padding_mask': memory_key_padding_mask,
                'tgt': tgt,
                'tgt_key_padding_mask': tgt_key_padding_mask}


# Full transformer learned decoder input vs. using noisy bbs as input to decoder
class FullTransformerLearnedEmbedsPreprocessor(SeqMaskPreprocessor):
    def __init__(self, cfg, d_model=64):
        super().__init__()
        self.embeds = nn.Embedding(cfg['model']['max_num_preds'], d_model).weight
        self.mask_out_bbone_mem = cfg['model']['structured']['mask_out_bbone_mem']

    def forward(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']

        # src = th.cat([bbone_bb_feats, x_t_bb_feats], dim=0)
        # tgt = self.embeds.unsqueeze(1).expand(-1, src.shape[1], -1)
        if bbone_bb_feats is not None:
            src = th.cat([bbone_bb_feats, x_t_bb_feats], dim=1)
            src_key_padding_mask = \
                self.combine_bbone_and_padding_masks(bbone_mask, x_t_mask)
            if self.mask_out_bbone_mem:
                memory_key_padding_mask = \
                    self.combine_bbone_and_padding_masks_and_mask_out_bbone(
                        bbone_mask, x_t_mask)
            else:
                memory_key_padding_mask = \
                    self.combine_bbone_and_padding_masks(
                        bbone_mask, x_t_mask)
        else:
            src = x_t_bb_feats
            src_key_padding_mask = x_t_mask
            memory_key_padding_mask = x_t_mask

        tgt = self.embeds.unsqueeze(0).expand(src.shape[0], -1, -1)
        tgt_key_padding_mask = x_t_mask

        return {'src': src,
                'tgt': tgt,
                'src_key_padding_mask': src_key_padding_mask,
                'tgt_key_padding_mask': tgt_key_padding_mask,
                'memory_key_padding_mask': memory_key_padding_mask}


class FullTransformerNoLearnedEmbedsPreprocessor(SeqMaskPreprocessor):
    def forward(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        x_t_bb_feats = preprocessed_feats['x_t_bb_feats']
        bbone_bb_feats = preprocessed_feats['bbone_bb_feats']

        if bbone_bb_feats is None:
            raise Exception('Transformer w/ no learned embeds & no conditioning not implemented')

        src = bbone_bb_feats
        tgt = x_t_bb_feats

        src_key_padding_mask = bbone_mask
        memory_key_padding_mask = bbone_mask
        tgt_key_padding_mask = x_t_mask

        return {'src': src,
                'tgt': tgt,
                'src_key_padding_mask': src_key_padding_mask,
                'tgt_key_padding_mask': tgt_key_padding_mask,
                'memory_key_padding_mask': memory_key_padding_mask}


class TimeConditioningPreprocessor(pl.LightningModule):
    def add_time_conditioning_seq_mask(self, seq, t, mask):
        assert seq is not None
        seq = self.add_time_conditioning_seq(seq, t)

        if mask is not None:
            mask = self.add_time_conditioning_mask(mask)
        return seq, mask

    def add_time_conditioning_seq(self, seq, t):
        raise RuntimeError('Must be implemented by child class')

    def add_time_conditioning_mask(self, mask):
        raise RuntimeError('Must be implemented by child class')

    def get_learned_embed_sin(self, ts):
        if self.rescale_timesteps:
            ts = self.rescale_timesteps_fn(ts)
        sin_embeds = self.time_embed(ts)
        learned_embeds = self.time_embed_layer(sin_embeds)
        return learned_embeds

    def rescale_timesteps_fn(self, ts):
        ts = ts.float() * (1000.0 / self.num_timesteps)
        return ts

    def get_learned_embed(self, ts):
        return self.embeds[ts]

    def init_time_embed_fn(self, type, d_model, num_timesteps):
        if type == 'sin':
            self.time_embedding_fn = TimeStepEncoding(d_model)
        elif type == 'learned':
            self.embeds = nn.Embedding(num_timesteps, d_model).weight
            self.time_embedding_fn = self.get_learned_embed
        elif type == 'l_sin':
            self.time_embed = TimeStepEncoding(d_model)
            self.time_embed_layer = nn.Linear(d_model, d_model)
            self.time_embedding_fn = self.get_learned_embed_sin


# Node conditioning preprocessors
class NodeConditioningPreprocessor(TimeConditioningPreprocessor):
    def __init__(
        self, num_timesteps, d_model=64, type='sin', method='seq',
        rescale_timesteps=False, **kwargs
    ):
        super().__init__()

        self.init_time_embed_fn(type, d_model, num_timesteps)

        if method == 'seq':
            self.add_time_conditioning_seq = self.add_time_conditioning_seq_seq
            self.add_time_conditioning_mask = self.add_time_conditioning_mask_seq
        elif method == 'add':
            self.add_time_conditioning_seq = self.add_time_conditioning_seq_add
            self.add_time_conditioning_mask = self.add_time_conditioning_mask_add

        self.rescale_timesteps = rescale_timesteps
        self.num_timesteps = num_timesteps

    def add_time_conditioning_seq_seq(self, seq,  t):
        # time_encoding = self.time_embedding_(t).unsqueeze(0)
        # seq = th.cat([time_encoding, seq], dim=0)
        time_encoding = self.time_embedding_fn(t).unsqueeze(1)
        seq = th.cat([time_encoding, seq], dim=1)
        return seq

    def add_time_conditioning_mask_seq(self, mask):
        time_mask = th.tensor([False], device=mask.device)
        time_mask = time_mask.unsqueeze(-1).expand(mask.shape[0], 1)
        mask = th.cat([time_mask, mask], dim=1)
        return mask

    def add_time_conditioning_seq_add(self, seq, t):
        time_encoding = self.time_embedding_fn(t).unsqueeze(1)
        seq = seq + time_encoding
        return seq

    def add_time_conditioning_mask_add(self, mask):
        return mask


# class FullTransformerNodeConditioningPreprocessor(NodeConditioningPreprocessor, FullTransformerPreprocessor):
#     pass
#
#
# class TransformerEncoderNodeConditioningPreprocessor(NodeConditioningPreprocessor, TransformerEncoderPreprocessor):
#     pass
#
#
# class TransformerDecoderNodeConditioningPreprocessor(NodeConditioningPreprocessor, TransformerDecoderPreprocessor):
#     pass


# Concat conditioning preprocessors
class CatConditioningPreprocessor(TimeConditioningPreprocessor):
    def __init__(
        self, num_timesteps, d_model=64, type='sin', rescale_timesteps=False,
        **kwargs
    ):
        super().__init__()
        self.init_time_embed_fn(type, d_model // 2, num_timesteps)

        self.rescale_timesteps = rescale_timesteps
        self.num_timesteps = num_timesteps

    def add_time_conditioning_seq(self, seq, t):
        # time_encoding = self.time_embedding_(t).unsqueeze(0)
        # time_encoding = time_encoding.expand(seq.shape[0], 1, 1)
        # seq = th.cat([seq, time_encoding], dim=-1)
        time_encoding = self.time_embedding_fn(t).unsqueeze(1)
        time_encoding = time_encoding.expand(-1, seq.shape[1], -1)
        seq = th.cat([seq, time_encoding], dim=-1)
        return seq

    def add_time_conditioning_mask(self, mask):
        return mask


# class FullTransformerCatConditioningPreprocessor(CatConditioningPreprocessor, FullTransformerPreprocessor):
#     pass
#
#
# class TransformerEncoderCatConditioningPreprocessor(CatConditioningPreprocessor, TransformerEncoderPreprocessor):
#     pass
#
#
# class TransformerDecoderCatConditioningPreprocessor(CatConditioningPreprocessor, TransformerDecoderPreprocessor):
#     pass


# No time conditioning preprocessors
class NoConditioningPreprocessor(TimeConditioningPreprocessor):
    def add_time_conditioning_seq(self, seq, t):
        return seq

    def add_time_conditioning_mask(self, mask):
        return mask


class FullTransformerNoConditioningPreprocessor(NoConditioningPreprocessor, FullTransformerPreprocessor):
    pass


class TransformerEncoderNoConditioningPreprocessor(NoConditioningPreprocessor, TransformerEncoderPreprocessor):
    pass


class TransformerDecoderNoConditioningPreprocessor(NoConditioningPreprocessor, TransformerDecoderPreprocessor):
    pass


class FullTransformerJoiner(pl.LightningModule):
    def __init__(self, embed_prep, condition_prep):
        super().__init__()
        self.embed_prep = embed_prep
        self.condition_prep = condition_prep

    def forward(self, preprocessed_feats, t, x_t_mask, bbone_mask):
        out = self.embed_prep(preprocessed_feats, t, x_t_mask, bbone_mask)
        return self.condition_prep(**out, t=t)
