"""CPU tests for the within-group hard-negative margin loss (--rank_within_loss_coef).

loss_rank_within pushes each Hungarian-matched query's foreground logit (pred_logits,
the deployed ranking score) above wrong-region queries (low IoU with the matched GT),
so the loss operates directly on class_embed(hs) rather than a throwaway projection.

The criterion is built via build_model() (mirroring the SimpleNamespace `_args` pattern
in tests/test_decoder_text_xattn.py, extended with the five rank_within_* keys) so the
params are threaded through SetCriterion.__init__ exactly as training would. Each test
then calls criterion.loss_rank_within(outputs, targets, indices) directly on small,
hand-built tensors -- the model's forward pass is never exercised.
"""

import types
import unittest

import torch

from vmr_detr.modeling.model import build_model

# d_model/hidden_dim is pinned to 256 (not "tiny") when going through build_model():
# gen_sineembed_for_position() hard-codes a 128-per-axis / 256-total sinusoidal
# embedding regardless of hidden_dim, and TransformerDecoder.forward feeds that
# embedding into ref_point_head = MLP(d_model, ...) unconditionally every layer (see
# tests/test_decoder_text_xattn.py). Irrelevant here since we never call the model's
# forward pass, but kept for parity with the established pattern.
HIDDEN_DIM = 256


def _args(rank_within_loss_coef=0.0, rank_within_margin=0.2, rank_within_neg_iou=0.3,
          rank_within_pos_iou=0.5, rank_within_warmup_epoch=0):
    return types.SimpleNamespace(
        hidden_dim=HIDDEN_DIM, nheads=4, enc_layers=1, dec_layers=2,
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
        use_stage2=False, stage2_dim=16, stage2_inner_bins=2,
        stage2_boundary_samples=1, stage2_max_shift_clips=1.0, stage2_shift_frac=0.0,
        stage2_exp_width=False, stage2_width_beta=0.7,
        stage2_positive_iou=0.2, stage2_start_epoch=10,
        stage2_joint_epoch=30, stage2_boundary_loss_coef=0.0,
        stage2_giou_loss_coef=0.0,
        stage2_quality_loss_coef=0.0,
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
        decoder_text_xattn=False,
        rank_within_loss_coef=rank_within_loss_coef,
        rank_within_margin=rank_within_margin,
        rank_within_neg_iou=rank_within_neg_iou,
        rank_within_pos_iou=rank_within_pos_iou,
        rank_within_warmup_epoch=rank_within_warmup_epoch,
    )


def _logits(values):
    """(1, nq, 1) pred_logits from a flat list of per-query logit values."""
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1)


def _spans(cxw_rows):
    """(1, nq, 2) pred_spans (cxw) from a list of [center, width] rows."""
    return torch.tensor(cxw_rows, dtype=torch.float32).unsqueeze(0)


def _gt(cxw_rows):
    """targets["span_labels"] for a single-sample batch, spans in cxw format."""
    return {"span_labels": [{"spans": torch.tensor(cxw_rows, dtype=torch.float32)}]}


class TestLossRankWithin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # rank_within_loss_coef > 0 so "rank_within" is threaded into criterion.losses
        # and the four rank_within_* params are set from our explicit _args() values
        # (margin=0.2, neg_iou=0.3, pos_iou=0.5, warmup_epoch=0).
        _, cls.criterion = build_model(_args(rank_within_loss_coef=0.5))
        cls.criterion.set_epoch(100)  # clear warmup (default rank_within_warmup_epoch=0)

    # (a) wrong-region query given a high logit while the matched query has a low
    # logit -> the loss must be > 0.
    def test_wrong_region_high_logit_vs_matched_low_logit_gives_positive_loss(self):
        outputs = {
            "pred_logits": _logits([-2.0, 2.0]),
            # q0 == gt0 (IoU 1.0, passes pos_iou gate); q1 is far away (IoU 0 < neg_iou).
            "pred_spans": _spans([[0.1, 0.2], [0.5, 0.1]]),
        }
        targets = _gt([[0.1, 0.2]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = self.criterion.loss_rank_within(outputs, targets, indices)["loss_rank_within"]
        self.assertGreater(loss.item(), 0.0)

    # (b) matched-query logit set far above all others -> loss ~= 0 (hinge inactive).
    def test_matched_logit_dominant_gives_near_zero_loss(self):
        outputs = {
            "pred_logits": _logits([10.0, -10.0]),
            "pred_spans": _spans([[0.1, 0.2], [0.5, 0.1]]),
        }
        targets = _gt([[0.1, 0.2]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = self.criterion.loss_rank_within(outputs, targets, indices)["loss_rank_within"]
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    # (c) pos-IoU gate: the matched query's own IoU with its target is below pos_iou
    # -> the pair is skipped entirely, loss == 0 even with an extreme negative logit.
    def test_pos_iou_gate_skips_low_iou_match(self):
        outputs = {
            "pred_logits": _logits([-5.0, 100.0]),
            # q0 ("matched") misses gt0 badly: IoU(q0, gt0) == 0 < pos_iou (0.5).
            "pred_spans": _spans([[0.5, 0.1], [0.9, 0.1]]),
        }
        targets = _gt([[0.1, 0.2]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = self.criterion.loss_rank_within(outputs, targets, indices)["loss_rank_within"]
        self.assertEqual(loss.item(), 0.0)

    # (d) a query with IoU >= neg_iou is not a wrong-region query -- it must not be
    # treated as a negative, even with a very high logit.
    def test_query_at_or_above_neg_iou_is_not_treated_as_negative(self):
        outputs = {
            "pred_logits": _logits([0.0, 100.0]),
            # q0 == gt0 (IoU 1.0). q1 vs gt0 has IoU exactly 0.5 (>= neg_iou 0.3).
            "pred_spans": _spans([[0.1, 0.2], [0.2, 0.4]]),
        }
        targets = _gt([[0.1, 0.2]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = self.criterion.loss_rank_within(outputs, targets, indices)["loss_rank_within"]
        self.assertEqual(loss.item(), 0.0)

    # (e) gradient: grad flows only through pred_logits (class_embed -> hs -> decoder),
    # never through pred_spans (IoU selection path is fully detached).
    def test_gradient_flows_only_through_logits_not_spans(self):
        logits = _logits([-2.0, 2.0]).requires_grad_(True)
        spans = _spans([[0.1, 0.2], [0.5, 0.1]]).requires_grad_(True)
        outputs = {"pred_logits": logits, "pred_spans": spans}
        targets = _gt([[0.1, 0.2]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = self.criterion.loss_rank_within(outputs, targets, indices)["loss_rank_within"]

        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)
        self.assertIsNone(spans.grad)

    # (f) multi-GT: q1 is a matched (positive) query for gt1. It must be excluded from
    # gt0's negative pool even though IoU(q1, gt0) == 0 and it carries a huge logit --
    # only q2 (a genuine wrong-region query for both targets) may contribute.
    def test_multi_gt_positive_for_other_target_excluded_from_negatives(self):
        outputs = {
            "pred_logits": _logits([0.0, 100.0, 1.0]),
            # q0 == gt0, q1 == gt1 (both IoU 1.0 with their own target); q2 overlaps
            # neither GT.
            "pred_spans": _spans([[0.1, 0.2], [0.9, 0.2], [0.5, 0.1]]),
        }
        targets = _gt([[0.1, 0.2], [0.9, 0.2]])
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        loss = self.criterion.loss_rank_within(outputs, targets, indices)["loss_rank_within"]

        # Correct (q1 excluded from both pairs' negatives, leaving only q2):
        #   pair(q0,gt0): viol = clamp(0.2 + 1.0 - 0.0, 0)   = 1.2
        #   pair(q1,gt1): viol = clamp(0.2 + 1.0 - 100.0, 0) = 0.0
        #   mean over 2 pairs = 0.6
        # A broken exclusion (only removing the pair's own q_star) would instead let
        # q1's logit of 100 leak into pair(q0,gt0)'s negatives, giving ~25.35.
        self.assertAlmostEqual(loss.item(), 0.6, places=4)
        self.assertLess(loss.item(), 10.0)

    # (g) build_model wiring: coef==0 must keep "rank_within" out of criterion.losses
    # (byte-for-byte unchanged training); coef>0 must add it.
    def test_build_model_toggles_rank_within_in_criterion_losses(self):
        _, criterion_off = build_model(_args(rank_within_loss_coef=0.0))
        self.assertNotIn("rank_within", criterion_off.losses)

        _, criterion_on = build_model(_args(rank_within_loss_coef=0.5))
        self.assertIn("rank_within", criterion_on.losses)


if __name__ == "__main__":
    unittest.main()
