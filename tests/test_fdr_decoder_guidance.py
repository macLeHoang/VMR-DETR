"""Tests for FDR-guided decoder reference and attention updates."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from vmr_detr.modeling.fdr import fdr_logits_to_spans, fdr_num_logits
from vmr_detr.modeling.model import VMRDETR
from vmr_detr.modeling.transformer import (
    TransformerDecoder,
    build_transformer,
    inverse_sigmoid,
)


class _IdentityDecoderLayer(nn.Module):
    def forward(self, tgt, memory, **kwargs):
        return tgt


class _CaptureDecoderLayer(nn.Module):
    def forward(self, tgt, memory, **kwargs):
        self.query_sine_embed = kwargs["query_sine_embed"]
        return tgt


class _FixedFDRGuide(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits, requires_grad=False)

    def forward(self, hidden):
        return self.logits.expand(hidden.shape[0], hidden.shape[1], -1)


class _LegacyTransformer(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.d_model = hidden_dim
        self.nhead = 1
        self.calls = 0

    def forward(self, src, mask, query_embed, pos_embed,
                video_length=None, level1_length=None):
        self.calls += 1
        batch_size = src.shape[0]
        references = query_embed.sigmoid().unsqueeze(0).unsqueeze(0)
        references = references.expand(1, batch_size, -1, -1)
        hidden = src.new_zeros(1, batch_size, query_embed.shape[0], self.d_model)
        video_memory = src[:, 1:1 + video_length]
        global_memory = src[:, 0]
        text_memory = src[:, 1 + video_length:]
        return hidden, references, video_memory, global_memory, text_memory


class _ZeroPosition(nn.Module):
    def forward(self, src, mask):
        return torch.zeros_like(src)


class TestFDRDecoderGuidance(unittest.TestCase):
    @staticmethod
    def _decoder(num_layers=2, decoder_layer=None, modulate_t_attn=False):
        return TransformerDecoder(
            decoder_layer or _IdentityDecoderLayer(),
            num_layers=num_layers,
            norm=nn.LayerNorm(256),
            return_intermediate=True,
            d_model=256,
            query_dim=2,
            modulate_t_attn=modulate_t_attn,
            fdr_decoder_refine=True,
            fdr_num_bins=4,
            fdr_reg_scale=1.5,
            fdr_min_ref_width=1.0 / 20.0,
            fdr_guide_start_epoch=10,
            fdr_guide_ramp_epochs=10,
        )

    def test_fdr_mode_does_not_create_legacy_bbox_refiner(self):
        decoder = self._decoder()
        self.assertIsNone(decoder.bbox_embed)

    def test_fdr_reference_updates_between_layers(self):
        decoder = self._decoder()
        initial = torch.tensor([[[0.5, 0.2]], [[0.5, 0.2]]])
        num_logits = fdr_num_logits(4)
        logits = torch.zeros(1, 1, 2 * num_logits)
        logits[..., 0] = 20.0
        logits[..., -1] = 20.0
        guide = _FixedFDRGuide(logits)
        _, references = decoder(
            torch.zeros(1, 1, 256),
            torch.zeros(1, 1, 256),
            memory_key_padding_mask=torch.zeros(1, 1, dtype=torch.bool),
            refpoints_unsigmoid=inverse_sigmoid(initial),
            fdr_guide_head=guide,
        )
        expected = fdr_logits_to_spans(
            logits.expand(2, -1, -1),
            initial,
            fdr_num_bins=4,
            fdr_reg_scale=1.5,
            fdr_min_ref_width=1.0 / 20.0,
        )
        self.assertTrue(torch.allclose(references[0], initial))
        self.assertTrue(torch.allclose(references[1], expected, atol=1e-5))
        self.assertFalse(references[1].requires_grad)

    def test_zero_fdr_logits_preserve_reference(self):
        decoder = self._decoder()
        initial = torch.tensor([[[0.5, 0.2]], [[0.5, 0.2]]])
        guide = _FixedFDRGuide(torch.zeros(1, 1, 2 * fdr_num_logits(4)))
        _, references = decoder(
            torch.zeros(1, 1, 256),
            torch.zeros(1, 1, 256),
            refpoints_unsigmoid=inverse_sigmoid(initial),
            fdr_guide_head=guide,
        )
        self.assertTrue(torch.allclose(references[1], initial, atol=1e-6))

    def test_missing_guide_preserves_reference_shape_and_values(self):
        decoder = self._decoder(num_layers=3)
        initial = torch.tensor([[[0.5, 0.2]]])

        _, references = decoder(
            torch.zeros(1, 1, 256),
            torch.zeros(1, 1, 256),
            refpoints_unsigmoid=inverse_sigmoid(initial),
        )

        self.assertEqual(references.shape, (3, 1, 1, 2))
        self.assertTrue(torch.allclose(references, initial.expand_as(references)))

    def test_collapsed_fdr_width_keeps_modulated_attention_finite(self):
        decoder = self._decoder(
            decoder_layer=_CaptureDecoderLayer(),
            modulate_t_attn=True,
        )
        initial = torch.tensor([[[0.1, 0.2]]])
        num_logits = fdr_num_logits(4)
        logits = torch.full((1, 1, 2 * num_logits), -1000.0)
        logits[..., num_logits // 2] = 1000.0
        logits[..., num_logits] = 1000.0

        _, references = decoder(
            torch.zeros(1, 1, 256),
            torch.zeros(1, 1, 256),
            refpoints_unsigmoid=inverse_sigmoid(initial),
            fdr_guide_head=_FixedFDRGuide(logits),
        )

        self.assertEqual(references[1, 0, 0, 1].item(), 0.0)
        self.assertTrue(torch.isfinite(decoder.layers[1].query_sine_embed).all())

    def test_build_transformer_accepts_none_min_reference_width(self):
        args = SimpleNamespace(
            hidden_dim=32,
            dropout=0.0,
            nheads=4,
            dim_feedforward=64,
            enc_layers=1,
            dec_layers=2,
            pre_norm=False,
            max_v_l=20,
            span_loss_type="fdr",
            fdr_num_bins=4,
            fdr_reg_scale=1.5,
            fdr_min_ref_width=None,
        )

        transformer = build_transformer(args)

        self.assertEqual(transformer.decoder.fdr_min_ref_width, 1.0 / 20.0)

    def test_non_fdr_model_supports_legacy_transformer_signature(self):
        transformer = _LegacyTransformer()
        model = VMRDETR(
            transformer=transformer,
            position_embed=_ZeroPosition(),
            txt_position_embed=None,
            txt_dim=4,
            vid_dim=4,
            num_queries=2,
            input_dropout=0.0,
            max_v_l=8,
            span_loss_type="l1",
            n_input_proj=1,
        )

        outputs = model(
            torch.randn(1, 3, 4),
            torch.ones(1, 3),
            torch.randn(1, 5, 4),
            torch.ones(1, 5),
        )

        self.assertEqual(transformer.calls, 2)
        self.assertEqual(outputs["pred_spans"].shape, (1, 2, 2))

    def test_attention_schedule_and_eval_strength(self):
        decoder = self._decoder()
        decoder.train()
        decoder.set_epoch(5)
        self.assertEqual(decoder._guide_attention_strength(), 1.0)
        decoder.set_epoch(15)
        self.assertEqual(decoder._guide_attention_strength(), 1.0)
        decoder.set_epoch(20)
        self.assertEqual(decoder._guide_attention_strength(), 1.0)
        decoder.eval()
        decoder.set_epoch(0)
        self.assertEqual(decoder._guide_attention_strength(), 1.0)


if __name__ == "__main__":
    unittest.main()
