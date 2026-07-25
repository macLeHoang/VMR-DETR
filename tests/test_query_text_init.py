"""CPU tests for pooled-text decoder query content initialization."""

import types
import unittest

import torch

from vmr_detr.modeling.model import build_model


class TestQueryTextInit(unittest.TestCase):
    @staticmethod
    def _args(query_text_init="none"):
        return types.SimpleNamespace(
            hidden_dim=256, nheads=4, enc_layers=1, dec_layers=1,
            dim_feedforward=256, dropout=0.1, pre_norm=False,
            position_embedding="sine", t_feat_dim=32, v_feat_dim=32,
            num_queries=5, input_dropout=0.0, aux_loss=False,
            contrastive_align_loss=False, contrastive_hdim=16,
            max_v_l=12, span_loss_type="fdr", use_txt_pos=False,
            n_input_proj=2, a_feat_dir=None, dfl_num_bins=4,
            dfl_ref_prior_sigma=2.0, fdr_num_bins=8,
            fdr_reg_scale=1.5, fdr_min_ref_width=None,
            query_init="random", query_anchor_widths=None,
            query_text_init=query_text_init,
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
        )

    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        return (
            torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, 12, 32), torch.ones(1, 12),
        )

    def test_zero_init_matches_baseline(self):
        model, _ = build_model(self._args(query_text_init="mean"))
        model.eval()
        src_txt, src_txt_mask, src_vid, src_vid_mask = self._inputs()

        with torch.no_grad():
            out_text_init = model(src_txt, src_txt_mask, src_vid, src_vid_mask)

        model.query_text_init = "none"
        with torch.no_grad():
            out_baseline = model(src_txt, src_txt_mask, src_vid, src_vid_mask)

        for key in ("pred_spans", "pred_logits", "saliency_scores"):
            self.assertTrue(torch.allclose(out_text_init[key], out_baseline[key]))

    def test_zero_init_preserves_existing_parameter_initialization(self):
        torch.manual_seed(1234)
        baseline, _ = build_model(self._args(query_text_init="none"))
        torch.manual_seed(1234)
        text_init, _ = build_model(self._args(query_text_init="mean"))

        text_init_state = text_init.state_dict()
        for name, value in baseline.state_dict().items():
            self.assertIn(name, text_init_state)
            self.assertTrue(torch.equal(value, text_init_state[name]), msg=name)

    def test_masked_mean_ignores_padding(self):
        model, _ = build_model(self._args(query_text_init="mean"))
        with torch.no_grad():
            torch.nn.init.normal_(model.txt_query_proj.weight)
            torch.nn.init.normal_(model.txt_query_proj.bias)

        bsz, length, d = 2, 5, 6
        mask = torch.zeros(bsz, length)
        mask[:, :3] = 1  # first 3 tokens valid, rest is padding

        src_a = torch.randn(bsz, length, d)
        src_b = src_a.clone()
        src_b[:, 3:] = torch.randn(bsz, length - 3, d)  # different padding garbage

        pooled_a = model._pool_query_text(src_a, mask)
        pooled_b = model._pool_query_text(src_b, mask)
        self.assertTrue(torch.allclose(pooled_a, pooled_b, atol=1e-6))

    def test_last_mode_picks_last_valid_token(self):
        model, _ = build_model(self._args(query_text_init="last"))
        bsz, length, d = 2, 4, 3
        src_txt = torch.arange(bsz * length * d, dtype=torch.float32).reshape(bsz, length, d)
        mask = torch.zeros(bsz, length)
        mask[0, :2] = 1  # sample 0 has 2 valid tokens -> last valid index 1
        mask[1, :4] = 1  # sample 1 has 4 valid tokens -> last valid index 3

        pooled = model._pool_query_text(src_txt, mask)
        self.assertTrue(torch.equal(pooled[0], src_txt[0, 1]))
        self.assertTrue(torch.equal(pooled[1], src_txt[1, 3]))

    def test_gradients_flow_to_text_projection(self):
        model, _ = build_model(self._args(query_text_init="mean"))
        model.train()
        src_txt, src_txt_mask, src_vid, src_vid_mask = self._inputs()

        outputs = model(src_txt, src_txt_mask, src_vid, src_vid_mask)
        loss = outputs["pred_logits"].sum() + outputs["pred_spans"].sum()
        loss.backward()

        self.assertIsNotNone(model.txt_query_proj.weight.grad)
        self.assertIsNotNone(model.query_content_embed.weight.grad)
        self.assertGreater(model.txt_query_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.query_content_embed.weight.grad.abs().sum().item(), 0.0)

    def test_none_mode_creates_no_modules(self):
        model, _ = build_model(self._args())
        self.assertFalse(hasattr(model, "txt_query_proj"))
        self.assertFalse(hasattr(model, "query_content_embed"))

    def test_invalid_mode_raises(self):
        args = self._args(query_text_init="bogus")
        with self.assertRaises(ValueError):
            build_model(args)


if __name__ == "__main__":
    unittest.main()
