"""CPU tests for the optional decoder text cross-attention sub-layer (--decoder_text_xattn).

Covers: flag-off leaves the model architecturally untouched (no new modules), flag-on
warm-starts as an exact identity (text_gate == 0 at init), the gate is live (a non-zero
gate changes the output), the text padding mask is honored by the new sub-layer, and the
full model (via build_model) produces identically-shaped outputs with the flag on or off.
"""

import types
import unittest

import torch

from vmr_detr.modeling.model import build_model
from vmr_detr.modeling.transformer import Transformer

# Deterministic dims for the Transformer-level tests. d_model is pinned to 256 (not
# actually "tiny") because gen_sineembed_for_position() hard-codes a 128-per-axis /
# 256-total sinusoidal embedding regardless of d_model, and TransformerDecoder.forward
# feeds that embedding straight into ref_point_head = MLP(d_model, ...) unconditionally
# every layer -- a pre-existing repo characteristic (see the other transformer tests,
# which all use d_model/hidden_dim=256 for the same reason), not something introduced
# here. Sequence lengths / layer counts / feedforward width are kept tiny.
D_MODEL = 256
NHEAD = 4
NUM_QUERIES = 3
NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 64
VIDEO_LENGTH = 4
TXT_LENGTH = 3
TOTAL_LENGTH = 1 + VIDEO_LENGTH + TXT_LENGTH  # global token + video + text


def _build_transformer(decoder_text_xattn, seed=0):
    torch.manual_seed(seed)
    return Transformer(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_queries=NUM_QUERIES,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=2,
        decoder_text_xattn=decoder_text_xattn,
    )


def _build_inputs(bs=2, seed=0):
    torch.manual_seed(seed)
    src = torch.randn(bs, TOTAL_LENGTH, D_MODEL)
    pos_embed = torch.randn(bs, TOTAL_LENGTH, D_MODEL)
    query_embed = torch.randn(NUM_QUERIES, 2)
    mask = torch.zeros(bs, TOTAL_LENGTH, dtype=torch.bool)  # True == padded
    mask[:, -1] = True  # last text token is padding for every sample
    return src, mask, query_embed, pos_embed


class TestDecoderTextXAttnFlagOff(unittest.TestCase):
    def test_no_text_modules_created_and_shapes_ok(self):
        transformer = _build_transformer(decoder_text_xattn=False)
        for layer in transformer.decoder.layers:
            self.assertFalse(hasattr(layer, "text_cross_attn"))
            self.assertFalse(hasattr(layer, "ca_text_qcontent_proj"))
            self.assertFalse(hasattr(layer, "text_gate"))
            self.assertFalse(getattr(layer, "text_xattn", False))

        transformer.eval()
        src, mask, query_embed, pos_embed = _build_inputs()
        with torch.no_grad():
            hs, references, memory_local, memory_global, txt_memory = transformer(
                src, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )

        self.assertEqual(tuple(hs.shape), (NUM_DECODER_LAYERS, 2, NUM_QUERIES, D_MODEL))
        self.assertEqual(tuple(references.shape), (NUM_DECODER_LAYERS, 2, NUM_QUERIES, 2))
        self.assertEqual(tuple(txt_memory.shape), (2, TXT_LENGTH, D_MODEL))


class TestDecoderTextXAttnFlagOn(unittest.TestCase):
    def test_gate_zero_is_identity_at_init(self):
        transformer = _build_transformer(decoder_text_xattn=True)
        transformer.eval()

        for layer in transformer.decoder.layers:
            self.assertTrue(hasattr(layer, "text_cross_attn"))
            self.assertTrue(torch.equal(layer.text_gate, torch.zeros_like(layer.text_gate)))

        src, mask, query_embed, pos_embed = _build_inputs()
        with torch.no_grad():
            hs_on, references_on, *_ = transformer(
                src, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )

        # Flip the same weights to behave as flag-off (rather than constructing a second
        # Transformer): _reset_parameters() walks torch's global RNG stream over every
        # dim>1 parameter in registration order, so two separately-constructed models
        # (one with the extra text_xattn params, one without) would desync xavier init
        # for every decoder layer after the first even with an identical manual_seed.
        # Reusing one model's weights isolates exactly what we want to prove: gate==0
        # makes the text branch a no-op.
        for layer in transformer.decoder.layers:
            layer.text_xattn = False
        with torch.no_grad():
            hs_off, references_off, *_ = transformer(
                src, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )
        for layer in transformer.decoder.layers:
            layer.text_xattn = True

        self.assertTrue(torch.allclose(hs_on, hs_off))
        self.assertTrue(torch.allclose(references_on, references_off))

    def test_nonzero_gate_changes_output(self):
        transformer = _build_transformer(decoder_text_xattn=True)
        transformer.eval()
        src, mask, query_embed, pos_embed = _build_inputs()

        with torch.no_grad():
            hs_zero, references_zero, *_ = transformer(
                src, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )

        for layer in transformer.decoder.layers:
            layer.text_gate.data.fill_(1.0)

        with torch.no_grad():
            hs_nonzero, references_nonzero, *_ = transformer(
                src, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )

        self.assertFalse(torch.allclose(hs_zero, hs_nonzero))

    def test_padding_mask_respected(self):
        transformer = _build_transformer(decoder_text_xattn=True)
        transformer.eval()
        for layer in transformer.decoder.layers:
            layer.text_gate.data.fill_(1.0)

        src, mask, query_embed, pos_embed = _build_inputs()
        with torch.no_grad():
            hs_base, references_base, *_ = transformer(
                src, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )

        # Mutate ONLY the padded text position (index -1, per _build_inputs' mask) with
        # wildly different values; a correctly-masked text_cross_attn must ignore it.
        src_mutated = src.clone()
        src_mutated[:, -1, :] = torch.randn_like(src_mutated[:, -1, :]) * 100.0
        with torch.no_grad():
            hs_mutated, references_mutated, *_ = transformer(
                src_mutated, mask, query_embed, pos_embed,
                video_length=VIDEO_LENGTH, level1_length=VIDEO_LENGTH,
            )

        self.assertTrue(torch.allclose(hs_base, hs_mutated, atol=1e-6))
        self.assertTrue(torch.allclose(references_base, references_mutated, atol=1e-6))


class TestDecoderTextXAttnFullModel(unittest.TestCase):
    """Full build_model() smoke + shape-parity tests."""

    @staticmethod
    def _args(decoder_text_xattn):
        return types.SimpleNamespace(
            hidden_dim=256, nheads=4, enc_layers=1, dec_layers=2,
            dim_feedforward=256, dropout=0.1, pre_norm=False,
            position_embedding="sine", t_feat_dim=32, v_feat_dim=32,
            num_queries=5, input_dropout=0.0, aux_loss=False,
            contrastive_align_loss=False, contrastive_hdim=16,
            max_v_l=12, span_loss_type="fdr", use_txt_pos=False,
            n_input_proj=2, a_feat_dir=None, dfl_num_bins=4,
            dfl_ref_prior_sigma=2.0, fdr_num_bins=8,
            fdr_reg_scale=1.5, fdr_min_ref_width=None,
            query_init="random", query_anchor_widths=None,
            query_text_init="none",
            video_input_proj="linear",
            use_hybrid_queries=False, hybrid_one_to_many_queries=10,
            hybrid_one_to_many_k=2, hybrid_one_to_many_loss_coef=1.0,
            eos_coef=0.1, temperature=0.07, span_loss_coef=1.0,
            giou_loss_coef=1.0, label_loss_coef=4.0, lw_saliency=1.0,
            saliency_margin=0.2, contrastive_align_loss_coef=0.0,
            contrastive_start_epoch=0, contrastive_decay_epoch=-1,
            contrastive_decay_coef=0.0, matching_type="hungarian",
            aux_matching_type="hungarian", aux_one_to_many_k=2,
            tal_topk=2, tal_alpha=1.0, tal_beta=6.0,
            label_loss_type="ce", vfl_alpha=0.75, vfl_gamma=2.0,
            quality_label_loss=False, quality_label_strength=0.5,
            quality_label_iou_gamma=1.0, quality_label_warmup_epoch=10,
            quality_label_ramp_epoch=30, width_loss_type="none",
            width_loss_coef=0.0, span_xx_loss_coef=0.0,
            go_lsd_loss_coef=0.0, go_lsd_temperature=2.0,
            go_lsd_start_epoch=0, fgl_loss_coef=None,
            saliency_hardneg_margin=0.4, hardneg_loss_coef=0.0,
            hardneg_warmup_epoch=0, hardneg_ramp_epoch=-1,
            dset_name="charades_sta", set_cost_span=10,
            set_cost_giou=1, set_cost_class=4, max_q_l=32,
            device=torch.device("cpu"),
            decoder_text_xattn=decoder_text_xattn,
        )

    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        return (
            torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, 12, 32), torch.ones(1, 12),
        )

    def test_flag_on_shapes_match_flag_off(self):
        model_off, _ = build_model(self._args(decoder_text_xattn=False))
        model_on, _ = build_model(self._args(decoder_text_xattn=True))
        model_off.eval()
        model_on.eval()

        src_txt, src_txt_mask, src_vid, src_vid_mask = self._inputs()
        with torch.no_grad():
            out_off = model_off(src_txt, src_txt_mask, src_vid, src_vid_mask)
            out_on = model_on(src_txt, src_txt_mask, src_vid, src_vid_mask)

        self.assertGreater(out_off["pred_logits"].numel(), 0)
        self.assertEqual(out_on["pred_logits"].shape, out_off["pred_logits"].shape)
        self.assertEqual(out_on["pred_spans"].shape, out_off["pred_spans"].shape)

    def test_flag_on_decoder_layers_have_text_modules(self):
        model, _ = build_model(self._args(decoder_text_xattn=True))
        for layer in model.transformer.decoder.layers:
            self.assertTrue(hasattr(layer, "text_cross_attn"))

    def test_flag_off_decoder_layers_have_no_text_modules(self):
        model, _ = build_model(self._args(decoder_text_xattn=False))
        for layer in model.transformer.decoder.layers:
            self.assertFalse(hasattr(layer, "text_cross_attn"))


if __name__ == "__main__":
    unittest.main()
