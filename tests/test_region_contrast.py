"""CPU tests for the region-contrast InfoNCE loss (--region_contrast_loss_coef).

loss_region_contrast is a text<->window-pooled-video InfoNCE: it pulls the GT-region
(window-pooled rc_vid, mean-mean-pooled per clip) toward the pooled text feature (rc_txt)
and pushes wrong-region windows (same-width shifts + the model's own top-scoring
wrong-region proposals) apart. Grads flow through rc_vid/rc_txt (the encoder is NOT
detached); only the *selection* of adversarial negatives is detached (built from
pred_spans/pred_logits.detach()).

The criterion (and, for the projection-toggle test, the model) is built via build_model()
(mirroring the SimpleNamespace `_args` pattern in tests/test_decoder_text_xattn.py /
tests/test_rank_within_loss.py, extended with the nine region_contrast_* keys) so the
params are threaded through SetCriterion.__init__ exactly as training would. Most tests
then call criterion.loss_region_contrast(outputs, targets, indices) directly on small,
hand-built tensors -- the model's forward pass is never exercised.
"""

import types
import unittest

import torch

from vmr_detr.modeling.model import build_model, SetCriterion
from vmr_detr.ops.span_utils import span_cxw_to_xx, temporal_iou

# d_model/hidden_dim is pinned to 256 (not "tiny") when going through build_model():
# gen_sineembed_for_position() hard-codes a 128-per-axis / 256-total sinusoidal
# embedding regardless of hidden_dim, and TransformerDecoder.forward feeds that
# embedding into ref_point_head = MLP(d_model, ...) unconditionally every layer (see
# tests/test_decoder_text_xattn.py). Irrelevant here since we never call the model's
# forward pass, but kept for parity with the established pattern.
HIDDEN_DIM = 256


def _args(region_contrast_loss_coef=0.0, region_contrast_dim=128,
          region_contrast_temperature=0.1, region_contrast_jitter_iou=0.7,
          region_contrast_neg_iou=0.3, region_contrast_n_jitter=2,
          region_contrast_n_shift=4, region_contrast_n_adversarial=3,
          region_contrast_warmup_epoch=0):
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
        region_contrast_loss_coef=region_contrast_loss_coef,
        region_contrast_dim=region_contrast_dim,
        region_contrast_temperature=region_contrast_temperature,
        region_contrast_jitter_iou=region_contrast_jitter_iou,
        region_contrast_neg_iou=region_contrast_neg_iou,
        region_contrast_n_jitter=region_contrast_n_jitter,
        region_contrast_n_shift=region_contrast_n_shift,
        region_contrast_n_adversarial=region_contrast_n_adversarial,
        region_contrast_warmup_epoch=region_contrast_warmup_epoch,
    )


class TestRegionContrastBuildModelWiring(unittest.TestCase):
    """(a) coef==0 must keep the projections unbuilt and "region_contrast" out of
    criterion.losses (byte-for-byte unchanged training); coef>0 must add both."""

    def test_flag_off_no_projections_flag_on_has_projections(self):
        model_off, criterion_off = build_model(_args(region_contrast_loss_coef=0.0))
        self.assertFalse(hasattr(model_off, "region_contrast_vid_proj"))
        self.assertFalse(hasattr(model_off, "region_contrast_txt_proj"))
        self.assertFalse(getattr(model_off, "region_contrast", False))
        self.assertNotIn("region_contrast", criterion_off.losses)
        self.assertEqual(criterion_off.weight_dict.get("loss_region_contrast"), 0.0)

        model_on, criterion_on = build_model(_args(region_contrast_loss_coef=1.0))
        self.assertTrue(hasattr(model_on, "region_contrast_vid_proj"))
        self.assertTrue(hasattr(model_on, "region_contrast_txt_proj"))
        self.assertTrue(getattr(model_on, "region_contrast", False))
        self.assertIn("region_contrast", criterion_on.losses)
        self.assertEqual(criterion_on.weight_dict.get("loss_region_contrast"), 1.0)

    def test_rejects_ce_span_mode_when_enabled(self):
        args = _args(region_contrast_loss_coef=1.0)
        args.span_loss_type = "ce"
        with self.assertRaisesRegex(ValueError, "span_loss_type"):
            build_model(args)

    def test_rejects_invalid_enabled_hyperparameters(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_model(_args(region_contrast_loss_coef=1.0, region_contrast_temperature=0.0))

        with self.assertRaisesRegex(ValueError, "dim"):
            build_model(_args(region_contrast_loss_coef=1.0, region_contrast_dim=0))


class TestRegionContrastGeoWindows(unittest.TestCase):
    """(b) jittered positives must clear jitter_iou vs GT; shifted negatives must stay
    below neg_iou vs GT."""

    def test_positives_meet_jitter_iou_negatives_below_neg_iou(self):
        torch.manual_seed(0)
        g = torch.tensor([0.5, 0.2])  # cxw: center=0.5, width=0.2 -> xx [0.4, 0.6]
        pos_xx, neg_xx = SetCriterion._rc_make_geo_windows(
            g, n_jitter=2, n_shift=4, jitter_iou=0.7, neg_iou=0.3)
        g_xx = span_cxw_to_xx(g.unsqueeze(0)).clamp(0, 1)

        # The GT window itself is always included, so positives is never empty.
        self.assertGreaterEqual(pos_xx.shape[0], 1)
        pos_ious = temporal_iou(pos_xx, g_xx)[0].squeeze(1)
        self.assertTrue((pos_ious >= 0.7 - 1e-6).all())

        self.assertGreater(neg_xx.shape[0], 0)
        neg_ious = temporal_iou(neg_xx, g_xx)[0].squeeze(1)
        self.assertTrue((neg_ious < 0.3 + 1e-6).all())


class TestRegionContrastPoolWindows(unittest.TestCase):
    """(c) pooling returns the right shape and falls back to the nearest valid clip
    (non-nan) for a window that contains no clip center."""

    def test_shape_and_nearest_clip_fallback(self):
        torch.manual_seed(0)
        L, dim = 4, 5
        rc_vid_b = torch.randn(L, dim)
        valid_len = torch.tensor(float(L))
        # Clip centers = [0.125, 0.375, 0.625, 0.875]. [0.2, 0.2] is a degenerate point
        # window containing no clip center -> nearest-clip fallback (clip 0, dist 0.075
        # vs clip 1's dist 0.175).
        windows_xx = torch.tensor([[0.0, 1.0], [0.2, 0.2]])
        pooled = SetCriterion._rc_pool_windows(rc_vid_b, windows_xx, valid_len)

        self.assertEqual(tuple(pooled.shape), (2, dim))
        self.assertFalse(torch.isnan(pooled).any())
        self.assertTrue(torch.allclose(pooled[1], rc_vid_b[0]))
        # Sanity: the full-video window pools the mean of all four clips.
        self.assertTrue(torch.allclose(pooled[0], rc_vid_b.mean(0)))


class TestRegionContrastLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # region_contrast_loss_coef > 0 so "region_contrast" is threaded into
        # criterion.losses and the model builds the rc_vid/rc_txt projections
        # (unused here since we hand-build `outputs` directly, but this mirrors
        # how build_model would be invoked for an actual training run).
        cls.model, cls.criterion = build_model(_args(region_contrast_loss_coef=1.0))
        cls.criterion.set_epoch(100)  # clear warmup (default region_contrast_warmup_epoch=0)

    # (d) loss is lower when rc_txt is set parallel to the GT-window-pooled rc_vid
    # feature than when set parallel to a (disjoint) negative-window feature.
    def test_loss_lower_when_text_parallels_positive_region(self):
        dim = 4
        pos_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        neg_vec = torch.tensor([0.0, 1.0, 0.0, 0.0])
        mid_vec = torch.zeros(dim)
        # 6 clips: [0,1] carry pos_vec, [4,5] carry neg_vec, [2,3] are neutral filler.
        rc_vid_base = torch.stack([pos_vec, pos_vec, mid_vec, mid_vec, neg_vec, neg_vec]).unsqueeze(0)
        video_mask = torch.ones(1, 6)
        pred_spans = torch.tensor([[[0.5, 0.2]]])  # (1, 1, 2) cxw, arbitrary (unused: n_adversarial=0)
        pred_logits = torch.zeros(1, 1, 1)
        targets = {"span_labels": [{"spans": torch.tensor([[0.15, 0.1]])}]}  # value irrelevant (windows fixed below)

        # Fix the geometry deterministically: positive window covers clips [0,1]
        # (centers 0.083, 0.25 -> pools to pos_vec); negative window covers clips
        # [4,5] (centers 0.75, 0.917 -> pools to neg_vec). This isolates the InfoNCE
        # math from _rc_make_geo_windows' randomized jitter/shift search.
        def fixed_windows(g_cxw, n_jitter, n_shift, jitter_iou, neg_iou, all_gt_xx=None):
            return torch.tensor([[0.0, 0.33]]), torch.tensor([[0.67, 1.0]])

        criterion = self.criterion
        orig_n_adversarial = criterion.region_contrast_n_adversarial
        criterion._rc_make_geo_windows = fixed_windows
        criterion.region_contrast_n_adversarial = 0
        try:
            outputs_low = {
                "rc_vid": rc_vid_base.clone(), "rc_txt": pos_vec.clone().unsqueeze(0),
                "video_mask": video_mask, "pred_spans": pred_spans, "pred_logits": pred_logits,
            }
            outputs_high = {
                "rc_vid": rc_vid_base.clone(), "rc_txt": neg_vec.clone().unsqueeze(0),
                "video_mask": video_mask, "pred_spans": pred_spans, "pred_logits": pred_logits,
            }
            loss_low = criterion.loss_region_contrast(outputs_low, targets, None)["loss_region_contrast"]
            loss_high = criterion.loss_region_contrast(outputs_high, targets, None)["loss_region_contrast"]
        finally:
            del criterion._rc_make_geo_windows
            criterion.region_contrast_n_adversarial = orig_n_adversarial

        self.assertLess(loss_low.item(), loss_high.item())

    # (e) grad flows into rc_vid and rc_txt (encoder path); pred_spans/pred_logits
    # receive no grad (the adversarial-negative SELECTION is fully detached).
    def test_gradient_flows_into_rc_vid_and_rc_txt_not_spans_or_logits(self):
        torch.manual_seed(0)
        bs, L, dim, nq = 1, 8, 6, 4
        rc_vid = torch.randn(bs, L, dim, requires_grad=True)
        rc_txt = torch.randn(bs, dim, requires_grad=True)
        video_mask = torch.ones(bs, L)
        pred_spans = torch.rand(bs, nq, 2, requires_grad=True)
        pred_logits = torch.randn(bs, nq, 1, requires_grad=True)
        targets = {"span_labels": [{"spans": torch.tensor([[0.3, 0.2]])}]}
        outputs = {
            "rc_vid": rc_vid, "rc_txt": rc_txt, "video_mask": video_mask,
            "pred_spans": pred_spans, "pred_logits": pred_logits,
        }

        # Force non-empty, deterministic positive/negative windows so the per-(b, gt)
        # loop body is guaranteed to execute (a skipped loop would leave rc_txt with
        # no grad path at all, which is not what this test is checking).
        def fixed_windows(g_cxw, n_jitter, n_shift, jitter_iou, neg_iou, all_gt_xx=None):
            return (torch.tensor([[0.0, 0.3], [0.05, 0.35]]),
                    torch.tensor([[0.6, 0.9], [0.65, 0.95]]))

        criterion = self.criterion
        criterion._rc_make_geo_windows = fixed_windows
        try:
            loss = criterion.loss_region_contrast(outputs, targets, None)["loss_region_contrast"]
            loss.backward()
        finally:
            del criterion._rc_make_geo_windows

        self.assertIsNotNone(rc_vid.grad)
        self.assertGreater(rc_vid.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(rc_txt.grad)
        self.assertGreater(rc_txt.grad.abs().sum().item(), 0.0)
        self.assertIsNone(pred_spans.grad)
        self.assertIsNone(pred_logits.grad)

    # (f) finite loss on a normal random batch (real _rc_make_geo_windows, no mocking).
    def test_finite_loss_on_random_batch(self):
        torch.manual_seed(1)
        bs, L, dim, nq = 2, 10, 8, 5
        outputs = {
            "rc_vid": torch.randn(bs, L, dim, requires_grad=True),
            "rc_txt": torch.randn(bs, dim, requires_grad=True),
            "video_mask": torch.ones(bs, L),
            "pred_spans": torch.rand(bs, nq, 2),
            "pred_logits": torch.randn(bs, nq, 1),
        }
        targets = {"span_labels": [
            {"spans": torch.tensor([[0.3, 0.2]])},
            {"spans": torch.tensor([[0.5, 0.3], [0.2, 0.1]])},
        ]}
        loss = self.criterion.loss_region_contrast(outputs, targets, None)["loss_region_contrast"]
        self.assertTrue(torch.isfinite(loss).item())

    # Multi-window samples: another relevant window for the same text must not be
    # selected as an adversarial negative for the current GT.
    def test_adversarial_negatives_exclude_other_gt_windows(self):
        dim = 4
        pos_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        neg_vec = torch.tensor([0.0, 1.0, 0.0, 0.0])
        zero_vec = torch.zeros(dim)
        rc_vid = torch.stack([
            pos_vec, pos_vec, zero_vec, zero_vec, neg_vec,
            neg_vec, zero_vec, zero_vec, pos_vec, pos_vec,
        ]).unsqueeze(0)
        outputs = {
            "rc_vid": rc_vid,
            "rc_txt": pos_vec.unsqueeze(0),
            "video_mask": torch.ones(1, 10),
            # q0 and q1 are both valid GT regions; q2 is the only true wrong-region proposal.
            "pred_spans": torch.tensor([[[0.1, 0.2], [0.9, 0.2], [0.5, 0.1]]]),
            "pred_logits": torch.tensor([[[100.0], [90.0], [0.0]]]),
        }
        targets = {"span_labels": [{"spans": torch.tensor([[0.1, 0.2], [0.9, 0.2]])}]}

        def fixed_windows(g_cxw, n_jitter, n_shift, jitter_iou, neg_iou, all_gt_xx=None):
            if g_cxw[0].item() < 0.5:
                pos = torch.tensor([[0.0, 0.2]])
            else:
                pos = torch.tensor([[0.8, 1.0]])
            return pos, torch.empty(0, 2)

        criterion = self.criterion
        orig_n_adversarial = criterion.region_contrast_n_adversarial
        criterion._rc_make_geo_windows = fixed_windows
        criterion.region_contrast_n_adversarial = 1
        try:
            loss = criterion.loss_region_contrast(outputs, targets, None)["loss_region_contrast"]
        finally:
            del criterion._rc_make_geo_windows
            criterion.region_contrast_n_adversarial = orig_n_adversarial

        self.assertLess(loss.item(), 0.01)

    # Warmup: before region_contrast_warmup_epoch, the loss is an exact zero that
    # still keeps rc_vid in the autograd graph (graph-keeping zero, not a bare 0.0).
    def test_warmup_gives_graph_keeping_zero(self):
        _, criterion = build_model(_args(region_contrast_loss_coef=1.0, region_contrast_warmup_epoch=5))
        criterion.set_epoch(0)
        rc_vid = torch.randn(1, 4, 128, requires_grad=True)
        outputs = {
            "rc_vid": rc_vid, "rc_txt": torch.randn(1, 128, requires_grad=True),
            "video_mask": torch.ones(1, 4),
            "pred_spans": torch.rand(1, 3, 2), "pred_logits": torch.randn(1, 3, 1),
        }
        targets = {"span_labels": [{"spans": torch.tensor([[0.3, 0.2]])}]}
        loss = criterion.loss_region_contrast(outputs, targets, None)["loss_region_contrast"]
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(rc_vid.grad)


if __name__ == "__main__":
    unittest.main()
