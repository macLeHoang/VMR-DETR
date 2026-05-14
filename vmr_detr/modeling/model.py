# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from vmr_detr.ops.span_utils import generalized_temporal_iou, span_cxw_to_xx, span_xx_to_cxw, temporal_iou

from vmr_detr.modeling.matcher import build_matcher, build_hungarian_matcher, build_one_to_many_matcher
from vmr_detr.modeling.transformer import build_transformer
from vmr_detr.modeling.position_encoding import build_position_encoding
from vmr_detr.ops.misc import accuracy
import numpy as np

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def dfl_logits_to_spans(span_logits, dfl_num_bins):
    """Decode DFL start/end distributions to normalized cxw spans."""
    num_bins = dfl_num_bins
    logits = span_logits.reshape(*span_logits.shape[:-1], 2, num_bins)
    prob = F.softmax(logits, dim=-1)
    bins = torch.arange(num_bins, dtype=prob.dtype, device=prob.device)
    boundaries = (prob * bins).sum(dim=-1) / float(num_bins - 1)
    start = torch.minimum(boundaries[..., 0], boundaries[..., 1])
    end = torch.maximum(boundaries[..., 0], boundaries[..., 1])
    spans_xx = torch.stack([start, end], dim=-1).clamp(0, 1)
    return span_xx_to_cxw(spans_xx)


def dfl_reference_prior_logits(reference_spans, dfl_num_bins, sigma):
    """Create start/end DFL logit priors centered on decoder reference spans."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    ref_xx = span_cxw_to_xx(reference_spans).clamp(0, 1)
    centers = ref_xx * float(dfl_num_bins - 1)
    bins = torch.arange(dfl_num_bins, dtype=centers.dtype, device=centers.device)
    bins = bins.view(*([1] * centers.dim()), dfl_num_bins)
    prior = -0.5 * ((bins - centers.unsqueeze(-1)) / float(sigma)) ** 2
    prior = prior - prior.max(dim=-1, keepdim=True).values
    return prior.reshape(*reference_spans.shape[:-1], 2 * dfl_num_bins)


def fdr_offset_support(num_bins, reg_scale, device=None, dtype=None):
    """Symmetric non-uniform FDR offset bins, dense around zero."""
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2.")
    if reg_scale <= 0:
        raise ValueError("reg_scale must be > 0.")
    base = torch.linspace(-1.0, 1.0, num_bins, device=device, dtype=dtype)
    return base.sign() * base.abs().pow(float(reg_scale))


def fdr_logits_to_spans(span_logits, reference_spans, fdr_num_bins, fdr_reg_scale, fdr_min_ref_width):
    """Decode residual start/end offset distributions around decoder references."""
    if fdr_min_ref_width <= 0:
        raise ValueError("fdr_min_ref_width must be > 0.")
    logits = span_logits.reshape(*span_logits.shape[:-1], 2, fdr_num_bins)
    prob = F.softmax(logits, dim=-1)
    support = fdr_offset_support(
        fdr_num_bins, fdr_reg_scale, device=prob.device, dtype=prob.dtype
    )
    offsets = (prob * support.view(*([1] * (prob.dim() - 1)), fdr_num_bins)).sum(dim=-1)

    ref_xx = span_cxw_to_xx(reference_spans).clamp(0, 1)
    ref_width = (ref_xx[..., 1] - ref_xx[..., 0]).clamp(min=fdr_min_ref_width)
    pred_start = ref_xx[..., 0] + offsets[..., 0] * ref_width
    pred_end = ref_xx[..., 1] + offsets[..., 1] * ref_width
    start = torch.minimum(pred_start, pred_end).clamp(0, 1)
    end = torch.maximum(pred_start, pred_end).clamp(0, 1)
    return span_xx_to_cxw(torch.stack([start, end], dim=-1))


class StreamTextConditionedEncoder(nn.Module):
    """Lightweight cross-attention block that conditions a video stream on text."""

    def __init__(self, hidden_dim, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 4
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, stream_feat, stream_mask, txt_feat, txt_mask, stream_pos=None, txt_pos=None):
        # stream_feat/txt_feat: (bsz, L, d), masks: 1 for valid, 0 for pad
        stream_query = stream_feat if stream_pos is None else stream_feat + stream_pos
        txt_key = txt_feat if txt_pos is None else txt_feat + txt_pos
        q = stream_query.transpose(0, 1)
        k = txt_key.transpose(0, 1)
        v = txt_feat.transpose(0, 1)
        txt_key_padding_mask = ~txt_mask.bool()
        attn_out = self.cross_attn(
            q, k, value=v, key_padding_mask=txt_key_padding_mask
        )[0].transpose(0, 1)
        stream_feat = self.norm1(stream_feat + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(F.relu(self.linear1(stream_feat), inplace=True)))
        stream_feat = self.norm2(stream_feat + self.dropout2(ff))
        return stream_feat * stream_mask.float().unsqueeze(-1)


class ResidualMultiScaleTemporalAdapter(nn.Module):
    """Weak parallel temporal adapter that starts as an identity residual."""

    def __init__(self, hidden_dim, dropout=0.1, text_conditioned=True):
        super().__init__()
        self.text_conditioned = text_conditioned
        self.branch_k3 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False
        )
        self.branch_k5 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim, bias=False
        )

        gate_input_dim = hidden_dim * (4 if text_conditioned else 3)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, x_mask, text_global=None):
        if self.text_conditioned and text_global is None:
            raise ValueError("text_global is required when text_conditioned=True.")

        mask = x_mask.float().unsqueeze(-1)
        x_masked = x * mask
        x_t = x_masked.transpose(1, 2)

        b3 = self.branch_k3(x_t).transpose(1, 2) * mask
        b5 = self.branch_k5(x_t).transpose(1, 2) * mask

        gate_inputs = [x_masked, b3, b5]
        if self.text_conditioned:
            text_global_expanded = text_global.unsqueeze(1).expand(-1, x.shape[1], -1)
            gate_inputs.append(text_global_expanded)
        gate_input = torch.cat(gate_inputs, dim=-1)
        gate = self.gate_mlp(gate_input).view(x.shape[0], x.shape[1], 3, x.shape[2])
        gate = torch.softmax(gate, dim=2)

        branches = torch.stack([x_masked, b3, b5], dim=2)
        refined = (gate * branches).sum(dim=2)
        residual = self.residual_norm(refined - x_masked)
        residual = self.dropout(residual) * self.residual_scale.view(1, 1, -1)
        return (x + residual) * mask


class VMRDETR(nn.Module):
    """ VMR DETR. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0,
                 use_gated_video_fusion=False, use_late_gated_video_fusion=False,
                 slowfast_dim=2304, clip_dim=512, tef_dim=2, dropout=0.1, dim_feedforward=None,
                 use_multiscale_stream_adapter=False, multiscale_adapter_dropout=0.1,
                 dfl_num_bins=16, dfl_ref_prior_sigma=2.0,
                 fdr_num_bins=32, fdr_reg_scale=1.5, fdr_min_ref_width=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         vmr-detr can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            dfl_num_bins: int, number of boundary distribution bins for dfl spans.
            dfl_ref_prior_sigma: float, Gaussian prior sigma in DFL bins.
            span_loss_type: str, one of [l1, ce, dfl, fdr]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
                dfl: start/end boundary distributions with expectation decoding.
                fdr: residual start/end boundary-offset distributions around decoder references.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        if dfl_num_bins < 2:
            raise ValueError("dfl_num_bins must be >= 2.")
        if dfl_ref_prior_sigma <= 0:
            raise ValueError("dfl_ref_prior_sigma must be > 0.")
        self.dfl_num_bins = dfl_num_bins
        self.dfl_ref_prior_sigma = dfl_ref_prior_sigma
        if fdr_num_bins < 2:
            raise ValueError("fdr_num_bins must be >= 2.")
        if fdr_reg_scale <= 0:
            raise ValueError("fdr_reg_scale must be > 0.")
        if fdr_min_ref_width is None:
            fdr_min_ref_width = 1.0 / float(max_v_l)
        if fdr_min_ref_width <= 0:
            raise ValueError("fdr_min_ref_width must be > 0.")
        self.fdr_num_bins = fdr_num_bins
        self.fdr_reg_scale = fdr_reg_scale
        self.fdr_min_ref_width = fdr_min_ref_width
        if span_loss_type == "l1":
            span_pred_dim = 2
        elif span_loss_type == "dfl":
            span_pred_dim = 2 * dfl_num_bins
        elif span_loss_type == "fdr":
            span_pred_dim = 2 * fdr_num_bins
        else:
            span_pred_dim = max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: foreground, 1: background
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, 2)
        
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        if use_gated_video_fusion and use_late_gated_video_fusion:
            raise ValueError("Use only one of use_gated_video_fusion or use_late_gated_video_fusion.")

        self.use_gated_video_fusion = use_gated_video_fusion
        self.use_late_gated_video_fusion = use_late_gated_video_fusion
        self.use_multiscale_stream_adapter = use_multiscale_stream_adapter
        self.slowfast_dim = slowfast_dim
        self.clip_dim = clip_dim
        self.tef_dim = tef_dim

        if self.use_gated_video_fusion or self.use_late_gated_video_fusion:
            self.input_slowfast_proj = nn.Sequential(*[
                LinearLayer(slowfast_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])
            self.input_clip_proj = nn.Sequential(*[
                LinearLayer(clip_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])
            self.video_gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.input_tef_proj = nn.Sequential(*[
                LinearLayer(tef_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])

        if self.use_late_gated_video_fusion:
            self.slowfast_txt_encoder = StreamTextConditionedEncoder(
                hidden_dim=hidden_dim, nhead=transformer.nhead,
                dim_feedforward=dim_feedforward, dropout=dropout
            )
            self.clip_txt_encoder = StreamTextConditionedEncoder(
                hidden_dim=hidden_dim, nhead=transformer.nhead,
                dim_feedforward=dim_feedforward, dropout=dropout
            )
            if self.use_multiscale_stream_adapter:
                self.clip_multiscale_adapter = ResidualMultiScaleTemporalAdapter(
                    hidden_dim=hidden_dim, dropout=multiscale_adapter_dropout
                )

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

    def _masked_text_global(self, src_txt, src_txt_mask):
        text_mask = src_txt_mask.float().unsqueeze(-1)
        text_denom = text_mask.sum(dim=1).clamp(min=1.0)
        return (src_txt * text_mask).sum(dim=1) / text_denom

    def _validate_text_mask(self, src_txt, src_txt_mask):
        if src_txt.shape[1] == 0 or not src_txt_mask.bool().any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid text token.")

    def _split_video_streams(self, src_vid):
        expected_vid_dim = self.slowfast_dim + self.clip_dim + self.tef_dim
        if src_vid.shape[-1] != expected_vid_dim:
            raise ValueError(
                f"Expected src_vid dim={expected_vid_dim} "
                f"(slowfast={self.slowfast_dim}, clip={self.clip_dim}, tef={self.tef_dim}), "
                f"but got {src_vid.shape[-1]}."
            )

        slowfast = src_vid[..., :self.slowfast_dim]
        clip = src_vid[..., self.slowfast_dim:self.slowfast_dim + self.clip_dim]
        tef = src_vid[..., self.slowfast_dim + self.clip_dim:]
        return slowfast, clip, tef

    def _early_fuse_streams_with_text(self, src_vid, src_txt, src_txt_mask):
        slowfast, clip, tef = self._split_video_streams(src_vid)

        slowfast_h = self.input_slowfast_proj(slowfast)
        clip_h = self.input_clip_proj(clip)

        text_global = self._masked_text_global(src_txt, src_txt_mask)
        text_global_expanded = text_global.unsqueeze(1).expand(-1, slowfast_h.shape[1], -1)
        gate_input = torch.cat(
            [slowfast_h, clip_h, slowfast_h * clip_h, text_global_expanded], dim=-1
        )
        gate = torch.sigmoid(self.video_gate_mlp(gate_input))
        fused_mem = gate * clip_h + (1.0 - gate) * slowfast_h
        return fused_mem + self.input_tef_proj(tef)

    def _late_fuse_streams_with_text(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        slowfast, clip, tef = self._split_video_streams(src_vid)

        slowfast_h = self.input_slowfast_proj(slowfast)
        clip_h = self.input_clip_proj(clip)
        text_global = self._masked_text_global(src_txt, src_txt_mask)

        if self.use_multiscale_stream_adapter:
            clip_h = self.clip_multiscale_adapter(clip_h, src_vid_mask, text_global=text_global)

        # Shared temporal positions: both streams are already aligned to the same video grid.
        stream_pos = self.position_embed(slowfast_h, src_vid_mask)
        txt_pos = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        slowfast_mem = self.slowfast_txt_encoder(
            slowfast_h, src_vid_mask, src_txt, src_txt_mask, stream_pos=stream_pos, txt_pos=txt_pos
        )
        clip_mem = self.clip_txt_encoder(
            clip_h, src_vid_mask, src_txt, src_txt_mask, stream_pos=stream_pos, txt_pos=txt_pos
        )

        text_global_expanded = text_global.unsqueeze(1).expand(-1, slowfast_mem.shape[1], -1)
        gate_input = torch.cat(
            [slowfast_mem, clip_mem, slowfast_mem * clip_mem, text_global_expanded], dim=-1
        )
        gate = torch.sigmoid(self.video_gate_mlp(gate_input))
        fused_mem = gate * clip_mem + (1.0 - gate) * slowfast_mem
        return fused_mem + self.input_tef_proj(tef)

    def _run_text_video_transformer(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        pos = torch.cat([pos_vid, pos_txt], dim=1)

        mask_global = torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask_global, mask], dim=1)
        src_global = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_global, src], dim=1)
        pos_global = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_global, pos], dim=1)

        return self.transformer(
            src, ~mask, self.query_embed.weight, pos, video_length=src_vid.shape[1]
        )

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, src_aud=None, src_aud_mask=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if src_aud is not None and (self.use_gated_video_fusion or self.use_late_gated_video_fusion):
            raise ValueError("gated video fusion currently supports non-audio runs only.")
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)

        self._validate_text_mask(src_txt, src_txt_mask)
        src_vid_input = src_vid
        src_txt = self.input_txt_proj(src_txt)

        if self.use_late_gated_video_fusion:
            src_vid = self._late_fuse_streams_with_text(src_vid_input, src_vid_mask, src_txt, src_txt_mask)
        elif self.use_gated_video_fusion:
            src_vid = self._early_fuse_streams_with_text(src_vid, src_txt, src_txt_mask)
        else:
            src_vid = self.input_vid_proj(src_vid)

        hs, reference, vid_mem, memory_global, txt_mem = self._run_text_video_transformer(
            src_vid, src_vid_mask, src_txt, src_txt_mask
        )

        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        if self.span_loss_type == "l1":
            outputs_coord = tmp + reference_before_sigmoid
            outputs_coord = outputs_coord.sigmoid()
            outputs_span_logits = None
            outputs_span_refs = None
        elif self.span_loss_type == "dfl":
            reference_prior_logits = dfl_reference_prior_logits(
                reference, self.dfl_num_bins, self.dfl_ref_prior_sigma
            )
            outputs_span_logits = tmp + reference_prior_logits
            outputs_coord = dfl_logits_to_spans(outputs_span_logits, self.dfl_num_bins)
            outputs_span_refs = None
        elif self.span_loss_type == "fdr":
            outputs_span_logits = torch.cumsum(tmp, dim=0)
            outputs_coord = fdr_logits_to_spans(
                outputs_span_logits,
                reference,
                self.fdr_num_bins,
                self.fdr_reg_scale,
                self.fdr_min_ref_width,
            )
            outputs_span_refs = reference
        else:
            outputs_coord = tmp
            outputs_span_logits = None
            outputs_span_refs = None
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}
        if outputs_span_logits is not None:
            out["pred_span_logits"] = outputs_span_logits[-1]
        if outputs_span_refs is not None:
            out["pred_span_refs"] = outputs_span_refs[-1]

        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem,
                proj_txt_mask=src_txt_mask.bool(),
            ))
            
        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        if self.use_late_gated_video_fusion:
            src_vid_neg = self._late_fuse_streams_with_text(src_vid_input, src_vid_mask, src_txt_neg, src_txt_mask_neg)
        else:
            src_vid_neg = src_vid

        _, _, vid_mem_neg, memory_global_neg, _ = self._run_text_video_transformer(
            src_vid_neg, src_vid_mask, src_txt_neg, src_txt_mask_neg
        )


        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        # print(src_vid_mask.shape, src_vid.shape, vid_mem_neg.shape, vid_mem.shape)
        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            if outputs_span_refs is not None:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b, 'pred_span_logits': c, 'pred_span_refs': d}
                    for a, b, c, d in zip(
                        outputs_class[:-1], outputs_coord[:-1],
                        outputs_span_logits[:-1], outputs_span_refs[:-1]
                    )]
            elif outputs_span_logits is not None:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b, 'pred_span_logits': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_span_logits[:-1])]
            else:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 dfl_num_bins=16,
                 fdr_num_bins=32, fdr_reg_scale=1.5, fdr_min_ref_width=None,
                 saliency_margin=1, use_matcher=True, contrastive_start_epoch=0,
                 contrastive_decay_epoch=-1, contrastive_decay_coef=0.0,
                 aux_matcher=None, aux_matching_type="hungarian",
                 matching_type="hungarian", tal_alpha=1.0, tal_beta=6.0):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce, dfl, fdr]
            max_v_l: int,
            dfl_num_bins: int,
            fdr_num_bins: int,
            fdr_reg_scale: float,
            fdr_min_ref_width: float
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        assert aux_matching_type in ("hungarian", "one_to_many")
        assert matching_type in ("hungarian", "tal")
        self.aux_matching_type = aux_matching_type
        self.aux_matcher = aux_matcher or matcher
        self.matching_type = matching_type
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        if self.matching_type == "tal" and self.span_loss_type not in ("l1", "dfl", "fdr"):
            raise ValueError("TAL matching requires span_loss_type to be 'l1', 'dfl', or 'fdr'.")
        if tal_alpha < 0 or tal_beta < 0:
            raise ValueError("tal_alpha and tal_beta must be non-negative.")
        self.tal_alpha = tal_alpha
        self.tal_beta = tal_beta
        if dfl_num_bins < 2:
            raise ValueError("dfl_num_bins must be >= 2.")
        self.dfl_num_bins = dfl_num_bins
        if fdr_num_bins < 2:
            raise ValueError("fdr_num_bins must be >= 2.")
        if fdr_reg_scale <= 0:
            raise ValueError("fdr_reg_scale must be > 0.")
        if fdr_min_ref_width is None:
            fdr_min_ref_width = 1.0 / float(max_v_l)
        if fdr_min_ref_width <= 0:
            raise ValueError("fdr_min_ref_width must be > 0.")
        self.fdr_num_bins = fdr_num_bins
        self.fdr_reg_scale = fdr_reg_scale
        self.fdr_min_ref_width = fdr_min_ref_width
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher
        self.contrastive_start_epoch = contrastive_start_epoch
        self.contrastive_decay_epoch = contrastive_decay_epoch
        self.contrastive_decay_coef = contrastive_decay_coef
        self.contrastive_base_coef = self.weight_dict.get("loss_contrastive_align", 0.0)
        self.current_epoch = 0
        self._update_contrastive_weight()

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self._update_contrastive_weight()

    def _update_contrastive_weight(self):
        if "loss_contrastive_align" not in self.weight_dict:
            return
        if self.current_epoch < self.contrastive_start_epoch:
            coef = 0.0
        elif self.contrastive_decay_epoch > 0 and self.current_epoch >= self.contrastive_decay_epoch:
            coef = self.contrastive_decay_coef
        else:
            coef = self.contrastive_base_coef
        self.weight_dict["loss_contrastive_align"] = coef

    def _loss_dfl(self, src_span_logits, tgt_spans):
        n_spans = src_span_logits.shape[0]
        num_bins = self.dfl_num_bins
        scale = float(num_bins - 1)
        logits = src_span_logits.reshape(n_spans, 2, num_bins)
        target_bins = span_cxw_to_xx(tgt_spans).clamp(0, 1) * scale

        target_left = target_bins.floor().long().clamp(min=0, max=num_bins - 1)
        target_right = (target_left + 1).clamp(max=num_bins - 1)
        weight_right = target_bins - target_left.float()
        weight_left = 1.0 - weight_right

        logits = logits.reshape(-1, num_bins)
        target_left = target_left.reshape(-1)
        target_right = target_right.reshape(-1)
        weight_left = weight_left.reshape(-1)
        weight_right = weight_right.reshape(-1)

        loss_left = F.cross_entropy(logits, target_left, reduction='none')
        loss_right = F.cross_entropy(logits, target_right, reduction='none')
        loss = loss_left * weight_left + loss_right * weight_right
        return (loss / float(np.log(num_bins))).view(n_spans, 2)

    def _fdr_offset_targets(self, src_span_refs, tgt_spans):
        ref_xx = span_cxw_to_xx(src_span_refs).clamp(0, 1)
        tgt_xx = span_cxw_to_xx(tgt_spans).clamp(0, 1)
        ref_width = (ref_xx[:, 1] - ref_xx[:, 0]).clamp(min=self.fdr_min_ref_width)
        target_offsets = torch.stack([
            (tgt_xx[:, 0] - ref_xx[:, 0]) / ref_width,
            (tgt_xx[:, 1] - ref_xx[:, 1]) / ref_width,
        ], dim=-1)

        support = fdr_offset_support(
            self.fdr_num_bins,
            self.fdr_reg_scale,
            device=target_offsets.device,
            dtype=target_offsets.dtype,
        )
        flat_offsets = target_offsets.reshape(-1).clamp(min=support[0].item(), max=support[-1].item())
        target_right = torch.searchsorted(support, flat_offsets).clamp(min=1, max=self.fdr_num_bins - 1)
        target_left = target_right - 1
        left_values = support[target_left]
        right_values = support[target_right]
        denom = (right_values - left_values).clamp(min=1e-6)
        weight_right = (flat_offsets - left_values) / denom
        weight_left = 1.0 - weight_right
        return target_left, target_right, weight_left, weight_right

    def _loss_fgl(self, src_span_logits, src_span_refs, src_spans, tgt_spans):
        n_spans = src_span_logits.shape[0]
        logits = src_span_logits.reshape(n_spans, 2, self.fdr_num_bins).reshape(-1, self.fdr_num_bins)
        target_left, target_right, weight_left, weight_right = self._fdr_offset_targets(
            src_span_refs.detach(), tgt_spans
        )

        loss_left = F.cross_entropy(logits, target_left, reduction='none')
        loss_right = F.cross_entropy(logits, target_right, reduction='none')
        loss = loss_left * weight_left + loss_right * weight_right

        with torch.no_grad():
            ious = torch.diag(
                temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans))[0]
            ).clamp(0, 1)
            weights = ious.unsqueeze(-1).expand(-1, 2).reshape(-1)
        loss = loss * weights
        return (loss / float(np.log(self.fdr_num_bins))).view(n_spans, 2)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        if sum(len(src) for src, _ in indices) == 0:
            zero = outputs["pred_spans"].sum() * 0
            if "pred_span_logits" in outputs:
                zero = zero + outputs["pred_span_logits"].sum() * 0
            if self.span_loss_type == "fdr":
                return {"loss_fgl": zero, "loss_giou": zero}
            return {"loss_span": zero, "loss_giou": zero}

        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        elif self.span_loss_type == "dfl":
            src_span_logits = outputs['pred_span_logits'][idx]
            loss_span = self._loss_dfl(src_span_logits, tgt_spans)
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        elif self.span_loss_type == "fdr":
            src_span_logits = outputs['pred_span_logits'][idx]
            src_span_refs = outputs['pred_span_refs'][idx]
            loss_fgl = self._loss_fgl(src_span_logits, src_span_refs, src_spans, tgt_spans)
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        if self.span_loss_type == "fdr":
            losses['loss_fgl'] = loss_fgl.mean()
        else:
            losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def _tal_quality_targets(self, outputs, targets, indices):
        src_logits = outputs['pred_logits']
        target_quality = src_logits.new_zeros(src_logits.shape[:2])
        targets = targets["span_labels"]
        pred_scores = src_logits.softmax(-1)[..., self.foreground_label]
        pred_spans_xx = span_cxw_to_xx(outputs["pred_spans"])

        with torch.no_grad():
            for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
                if len(src_idx) == 0:
                    continue
                src_idx = src_idx.to(src_logits.device)
                tgt_idx = tgt_idx.to(src_logits.device)
                tgt_spans = targets[batch_idx]["spans"]
                tgt_spans_xx = span_cxw_to_xx(tgt_spans)
                ious = temporal_iou(pred_spans_xx[batch_idx], tgt_spans_xx)[0].clamp(min=0)
                alignment = pred_scores[batch_idx].unsqueeze(1).pow(self.tal_alpha) * ious.pow(self.tal_beta)

                for tgt in tgt_idx.unique(sorted=True):
                    pos_mask = tgt_idx == tgt
                    pos_src = src_idx[pos_mask]
                    pos_alignment = alignment[pos_src, tgt]
                    pos_iou = ious[pos_src, tgt]
                    max_alignment = pos_alignment.max().clamp(min=1e-9)
                    max_iou = pos_iou.max()
                    quality = (pos_alignment / max_alignment * max_iou).clamp(0, 1)
                    target_quality[batch_idx, pos_src] = quality

        return target_quality

    def _loss_labels_tal(self, outputs, targets, indices, log=True):
        src_logits = outputs['pred_logits']
        target_quality = self._tal_quality_targets(outputs, targets, indices)
        fg_logits = src_logits[..., self.foreground_label] - src_logits[..., self.background_label]
        loss_bce = F.binary_cross_entropy_with_logits(fg_logits, target_quality, reduction="none")
        loss_weight = torch.full_like(target_quality, self.eos_coef)
        loss_weight = torch.where(target_quality > 0, torch.ones_like(loss_weight), loss_weight)
        losses = {'loss_label': (loss_bce * loss_weight).mean()}

        idx = self._get_src_permutation_idx(indices)
        if log and len(idx[0]) > 0:
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        elif log:
            losses['class_error'] = src_logits.new_tensor(100.)
        return losses

    def loss_labels(self, outputs, targets, indices, log=True, matching_type=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        matching_type = self.matching_type if matching_type is None else matching_type
        if matching_type == "tal":
            return self._loss_labels_tal(outputs, targets, indices, log=log)

        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log and len(idx[0]) > 0:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        elif log:
            losses['class_error'] = src_logits.new_tensor(100.)
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        # loss_neg_pair = torch.sigmoid(saliency_scores_neg).mean()
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        # for rand_idx in range(1, 13, 3):
        #     # 1, 4, 7, 10 --> 5 stages
        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3

            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]

            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)

            loss = - mean_log_prob_pos * batch_drop_mask

            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        # print(loss_saliency, loss_rank_contrastive)
        # loss_saliency = loss_saliency + loss_rank_contrastive
        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        # loss_saliency = loss_rank_contrastive
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d) text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        if self.current_epoch < self.contrastive_start_epoch:
            return {"loss_contrastive_align": normalized_img_embed.new_zeros(())}

        txt_mask = outputs.get("proj_txt_mask")
        if txt_mask is None:
            txt_mask = torch.ones(
                normalized_text_embed.shape[:2], dtype=torch.bool, device=normalized_text_embed.device)
        txt_mask = txt_mask.float()
        txt_global = (normalized_text_embed * txt_mask.unsqueeze(-1)).sum(1)
        txt_global = txt_global / txt_mask.sum(1, keepdim=True).clamp(min=1.0)
        txt_global = F.normalize(txt_global, p=2, dim=-1)  # (bsz, d)
        normalized_img_embed = F.normalize(normalized_img_embed, p=2, dim=-1)
        logits = torch.einsum("bqd,bd->bq", normalized_img_embed, txt_global) / self.temperature  # (bsz, #queries)

        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        valid_rows = positive_map.sum(1) > 0
        if not valid_rows.any():
            return {"loss_contrastive_align": logits.new_zeros(())}

        pos_term = (logits * positive_map.float()).sum(1)  # (bsz,)
        num_pos = positive_map.sum(1).clamp(min=1)  # (bsz,)
        neg_term = logits.logsumexp(1)  # (bsz,)
        loss_nce = -pos_term / num_pos + neg_term  # (bsz,)
        losses = {"loss_contrastive_align": loss_nce[valid_rows].mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        if len(indices) == 0 or sum(len(src) for src, _ in indices) == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        if len(indices) == 0 or sum(len(tgt) for _, tgt in indices) == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return empty, empty
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    aux_indices = self.aux_matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    aux_indices = None
                    losses_target = ["saliency"]    
                # for loss in self.losses:
                for loss in losses_target:
                    if loss in ("saliency", "contrastive_align"):  # final layer only
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs["matching_type"] = "hungarian"
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)
    dfl_num_bins = getattr(args, "dfl_num_bins", 16)
    dfl_ref_prior_sigma = getattr(args, "dfl_ref_prior_sigma", 2.0)
    fdr_num_bins = getattr(args, "fdr_num_bins", 32)
    fdr_reg_scale = getattr(args, "fdr_reg_scale", 1.5)
    fdr_min_ref_width = getattr(args, "fdr_min_ref_width", None)
    if fdr_min_ref_width is not None and fdr_min_ref_width <= 0:
        fdr_min_ref_width = 1.0 / float(args.max_v_l)

    if args.a_feat_dir is None:
        model = VMRDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            max_v_l=args.max_v_l,
            span_loss_type=args.span_loss_type,
            dfl_num_bins=dfl_num_bins,
            dfl_ref_prior_sigma=dfl_ref_prior_sigma,
            fdr_num_bins=fdr_num_bins,
            fdr_reg_scale=fdr_reg_scale,
            fdr_min_ref_width=fdr_min_ref_width,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            use_gated_video_fusion=args.use_gated_video_fusion,
            use_late_gated_video_fusion=args.use_late_gated_video_fusion,
            slowfast_dim=args.slowfast_dim,
            clip_dim=args.clip_dim,
            tef_dim=args.tef_dim,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            use_multiscale_stream_adapter=args.use_multiscale_stream_adapter,
            multiscale_adapter_dropout=args.multiscale_adapter_dropout,
        )
    else:
        model = VMRDETR(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            max_v_l=args.max_v_l,
            span_loss_type=args.span_loss_type,
            dfl_num_bins=dfl_num_bins,
            dfl_ref_prior_sigma=dfl_ref_prior_sigma,
            fdr_num_bins=fdr_num_bins,
            fdr_reg_scale=fdr_reg_scale,
            fdr_min_ref_width=fdr_min_ref_width,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            use_gated_video_fusion=args.use_gated_video_fusion,
            use_late_gated_video_fusion=args.use_late_gated_video_fusion,
            slowfast_dim=args.slowfast_dim,
            clip_dim=args.clip_dim,
            tef_dim=args.tef_dim,
            dropout=args.dropout,
            dim_feedforward=args.dim_feedforward,
            use_multiscale_stream_adapter=args.use_multiscale_stream_adapter,
            multiscale_adapter_dropout=args.multiscale_adapter_dropout,
        )

    matching_type = getattr(args, "matching_type", "hungarian")
    matcher = build_matcher(args)
    aux_matching_type = getattr(args, "aux_matching_type", "hungarian")
    aux_matcher = build_one_to_many_matcher(args) if aux_matching_type == "one_to_many" else build_hungarian_matcher(args)
    span_loss_key = "loss_fgl" if args.span_loss_type == "fdr" else "loss_span"
    span_loss_coef = (
        args.span_loss_coef
        if span_loss_key == "loss_span" or getattr(args, "fgl_loss_coef", None) is None
        else args.fgl_loss_coef
    )
    weight_dict = {span_loss_key: span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()
                 if k not in ("loss_saliency", "loss_contrastive_align")}
            )
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        dfl_num_bins=dfl_num_bins,
        fdr_num_bins=fdr_num_bins,
        fdr_reg_scale=fdr_reg_scale,
        fdr_min_ref_width=fdr_min_ref_width,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
        contrastive_start_epoch=args.contrastive_start_epoch,
        contrastive_decay_epoch=args.contrastive_decay_epoch,
        contrastive_decay_coef=args.contrastive_decay_coef,
        aux_matcher=aux_matcher,
        aux_matching_type=aux_matching_type,
        matching_type=matching_type,
        tal_alpha=getattr(args, "tal_alpha", 1.0),
        tal_beta=getattr(args, "tal_beta", 6.0),
    )
    criterion.to(device)
    return model, criterion
