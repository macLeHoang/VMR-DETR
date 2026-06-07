"""CPU tests for the unified proposal refinement and quality stage."""

import types
import unittest

import torch

from vmr_detr.modeling.model import (
    ProposalRefinementStage,
    SetCriterion,
    build_model,
)
from vmr_detr.ops.span_utils import span_cxw_to_xx, span_xx_to_cxw


def _make_stage(hidden_dim=16, inner_bins=3, boundary_samples=2,
                max_shift_clips=1, span_num_bins=8, shift_frac=0.0):
    return ProposalRefinementStage(
        hidden_dim=hidden_dim,
        stage2_dim=12,
        inner_bins=inner_bins,
        boundary_samples=boundary_samples,
        max_shift_clips=max_shift_clips,
        span_num_bins=span_num_bins,
        shift_frac=shift_frac,
    )


def _make_inputs(bsz=2, queries=5, hidden_dim=16, length=12, bins=8):
    hs = torch.randn(bsz, queries, hidden_dim)
    centers = torch.rand(bsz, queries) * 0.5 + 0.25
    widths = torch.rand(bsz, queries) * 0.2 + 0.1
    spans = torch.stack([centers, widths], dim=-1)
    logits = torch.randn(bsz, queries, 1)
    span_logits = torch.randn(bsz, queries, 2 * bins)
    memory = torch.randn(bsz, length, hidden_dim)
    mask = torch.ones(bsz, length)
    mask[1, length // 2:] = 0
    return hs, spans, logits, span_logits, memory, mask


class TestProposalRefinementStage(unittest.TestCase):
    def test_identity_and_neutral_quality_at_initialization(self):
        stage = _make_stage()
        hs, spans, logits, span_logits, memory, mask = _make_inputs()
        # Use wide spans (0.3) and a full mask so min-width clamp is a no-op
        # and identity holds exactly at zero-initialized heads.
        widths = torch.full_like(spans[..., 1], 0.3)
        centers = torch.full_like(spans[..., 0], 0.5)
        spans = torch.stack([centers, widths], dim=-1)
        mask = torch.ones_like(mask)
        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=False
        )

        self.assertTrue(torch.allclose(outputs["refined_spans"], spans, atol=1e-6))
        self.assertTrue(torch.allclose(
            outputs["refined_quality_logits"],
            torch.zeros_like(outputs["refined_quality_logits"]),
        ))
        expected_scores = logits.squeeze(-1).sigmoid() * 0.5
        self.assertTrue(torch.allclose(outputs["refined_scores"], expected_scores))

    def test_linear_sampler_matches_clip_centers(self):
        memory = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        positions = torch.tensor([[[0.125, 0.375, 0.625, 0.875]]])
        sampled = ProposalRefinementStage._sample_1d(
            memory, positions, torch.tensor([4.0])
        )
        self.assertTrue(torch.allclose(
            sampled[0, 0, :, 0], torch.arange(4, dtype=torch.float32), atol=1e-6
        ))

    def test_directional_pooling_preserves_temporal_order(self):
        stage = _make_stage(
            hidden_dim=1, inner_bins=2, boundary_samples=1, span_num_bins=None
        )
        memory = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        spans_xx = torch.tensor([[[0.25, 0.75]]])
        directional, inner = stage._pool_features(
            memory, spans_xx, torch.tensor([8.0])
        )
        start_before, start_after, end_before, end_after = directional[0, 0]
        self.assertLess(start_before.item(), start_after.item())
        self.assertLess(end_before.item(), end_after.item())
        self.assertLess(inner[0, 0, 0].item(), inner[0, 0, 1].item())

    def test_shift_is_bounded_and_boundaries_remain_ordered(self):
        stage = _make_stage(max_shift_clips=1)
        torch.nn.init.zeros_(stage.localization_head[-1].weight)
        stage.localization_head[-1].bias.data.copy_(torch.tensor([20.0, -20.0]))
        hs, spans, logits, span_logits, memory, mask = _make_inputs()
        # Use wide spans (>= 0.3) and a full mask so the min-width clamp does not
        # push boundaries beyond base_shift, keeping the bound assertion valid.
        widths = torch.full_like(spans[..., 1], 0.3)
        centers = torch.full_like(spans[..., 0], 0.5)
        spans = torch.stack([centers, widths], dim=-1)
        mask = torch.ones_like(mask)
        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=False
        )
        base_xx = span_cxw_to_xx(spans)
        refined_xx = span_cxw_to_xx(outputs["refined_spans"])
        valid_len = mask.sum(1)
        for batch_idx in range(len(valid_len)):
            max_shift = 1.0 / valid_len[batch_idx]
            self.assertLessEqual(
                (refined_xx[batch_idx] - base_xx[batch_idx]).abs().max().item(),
                max_shift.item() + 1e-6,
            )
        self.assertTrue((refined_xx[..., 0] < refined_xx[..., 1]).all())
        self.assertTrue((refined_xx >= 0).all())
        self.assertTrue((refined_xx <= 1).all())

    def test_detached_stage_blocks_stage1_gradients(self):
        stage = _make_stage()
        for head in (stage.localization_head, stage.quality_head):
            torch.nn.init.normal_(head[-1].weight, std=0.1)
        hs, spans, logits, span_logits, memory, mask = _make_inputs()
        hs.requires_grad_(True)
        spans.requires_grad_(True)
        memory.requires_grad_(True)
        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=True
        )
        (outputs["refined_spans"].sum()
         + outputs["refined_quality_logits"].sum()).backward()
        self.assertIsNone(hs.grad)
        self.assertIsNone(spans.grad)
        self.assertIsNone(memory.grad)

    def test_joint_stage_allows_localization_gradients(self):
        stage = _make_stage()
        torch.nn.init.normal_(stage.localization_head[-1].weight, std=0.1)
        hs, spans, logits, span_logits, memory, mask = _make_inputs()
        hs.requires_grad_(True)
        spans.requires_grad_(True)
        memory.requires_grad_(True)
        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=False
        )
        outputs["refined_spans"].sum().backward()
        self.assertGreater(hs.grad.abs().sum().item(), 0)
        self.assertGreater(spans.grad.abs().sum().item(), 0)
        self.assertGreater(memory.grad.abs().sum().item(), 0)

    def test_quality_coordinates_are_detached(self):
        stage = _make_stage()
        torch.nn.init.normal_(stage.quality_head[-1].weight, std=0.1)
        hs, spans, logits, span_logits, memory, mask = _make_inputs()
        spans.requires_grad_(True)
        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=False
        )
        grad = torch.autograd.grad(
            outputs["refined_quality_logits"].sum(),
            spans,
            allow_unused=True,
        )[0]
        self.assertTrue(grad is None or grad.abs().sum().item() == 0)

    def test_width_proportional_shift_scales_with_span(self):
        """Wider spans should receive a larger max per-boundary movement when shift_frac > 0."""
        stage = _make_stage(shift_frac=0.5)
        torch.nn.init.zeros_(stage.localization_head[-1].weight)
        # Bias saturates tanh: start pushed right (+1), end pushed left (-1)
        stage.localization_head[-1].bias.data.copy_(torch.tensor([20.0, -20.0]))

        hidden_dim = 16
        length = 12
        bsz = 1
        queries = 2
        hs = torch.randn(bsz, queries, hidden_dim)
        logits = torch.randn(bsz, queries, 1)
        span_logits = torch.randn(bsz, queries, 16)
        memory = torch.randn(bsz, length, hidden_dim)
        mask = torch.ones(bsz, length)

        # Wide span: width 0.5, narrow span: width 0.05, both centered at 0.5
        # spans shape: (bsz=1, queries=2, 2) — last dim is [center, width]
        spans = torch.tensor([[[0.5, 0.5], [0.5, 0.05]]])

        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=False
        )
        base_xx = span_cxw_to_xx(spans)
        refined_xx = span_cxw_to_xx(outputs["refined_spans"])
        movement = (refined_xx - base_xx).abs().max(dim=-1).values  # (B, Q)

        wide_movement = movement[0, 0].item()
        narrow_movement = movement[0, 1].item()
        self.assertGreater(
            wide_movement, narrow_movement,
            msg=f"Wide span movement ({wide_movement:.4f}) should exceed narrow ({narrow_movement:.4f})"
        )

    def test_min_width_enforced(self):
        """After boundary collapse, refined span should still have width >= 1/valid_len."""
        stage = _make_stage()
        torch.nn.init.zeros_(stage.localization_head[-1].weight)
        # Bias collapses the span: start pushed right, end pushed left
        stage.localization_head[-1].bias.data.copy_(torch.tensor([20.0, -20.0]))

        hidden_dim = 16
        length = 12
        bsz = 1
        queries = 3
        hs = torch.randn(bsz, queries, hidden_dim)
        logits = torch.randn(bsz, queries, 1)
        span_logits = torch.randn(bsz, queries, 16)
        memory = torch.randn(bsz, length, hidden_dim)
        mask = torch.ones(bsz, length)  # full mask -> valid_len = 12
        valid_len = mask.sum(1)  # tensor([12.])

        centers = torch.full((bsz, queries), 0.5)
        widths = torch.full((bsz, queries), 0.3)
        spans = torch.stack([centers, widths], dim=-1)

        outputs = stage(
            hs, spans, logits, span_logits, memory, mask, detach_stage1=False
        )
        refined_xx = span_cxw_to_xx(outputs["refined_spans"])
        refined_widths = refined_xx[..., 1] - refined_xx[..., 0]
        min_width = 1.0 / valid_len[0].item()

        self.assertTrue(
            (refined_widths >= min_width - 1e-6).all(),
            msg=f"min refined width {refined_widths.min().item():.6f} < min_width {min_width:.6f}"
        )
        self.assertTrue(
            (refined_xx[..., 0] < refined_xx[..., 1]).all(),
            msg="refined start must be strictly less than refined end"
        )


class TestStage2Loss(unittest.TestCase):
    @staticmethod
    def _criterion(epoch=11):
        criterion = object.__new__(SetCriterion)
        criterion.current_epoch = epoch
        criterion.stage2_start_epoch = 10
        criterion.stage2_positive_iou = 0.2
        criterion.max_v_l = 20
        criterion.vfl_alpha = 0.75
        criterion.vfl_gamma = 2.0
        return criterion

    def test_warmup_returns_graph_connected_zero(self):
        criterion = self._criterion(epoch=10)
        spans = torch.tensor([[[0.5, 0.4]]], requires_grad=True)
        quality = torch.zeros(1, 1, requires_grad=True)
        outputs = {
            "pred_spans": spans,
            "refined_spans": spans,
            "refined_quality_logits": quality,
        }
        targets = {"span_labels": [{"spans": torch.tensor([[0.5, 0.4]])}]}
        losses = criterion.loss_stage2(outputs, targets, None)
        total = sum(losses.values())
        self.assertEqual(total.item(), 0.0)
        total.backward()
        self.assertIsNotNone(quality.grad)

    def test_perfect_refinement_has_zero_localization_loss(self):
        criterion = self._criterion()
        gt = torch.tensor([[0.5, 0.4]])
        outputs = {
            "pred_spans": torch.tensor([[[0.5, 0.35]]]),
            "refined_spans": gt.unsqueeze(0).clone().requires_grad_(True),
            "refined_quality_logits": torch.zeros(1, 1, requires_grad=True),
        }
        targets = {"span_labels": [{"spans": gt}]}
        losses = criterion.loss_stage2(outputs, targets, None)
        self.assertAlmostEqual(losses["loss_stage2_boundary"].item(), 0.0, places=6)
        self.assertAlmostEqual(losses["loss_stage2_giou"].item(), 0.0, places=6)

    def test_quality_target_does_not_backpropagate_into_spans(self):
        criterion = self._criterion()
        base = torch.tensor([[[0.5, 0.4]]], requires_grad=True)
        refined = torch.tensor([[[0.52, 0.38]]], requires_grad=True)
        quality = torch.zeros(1, 1, requires_grad=True)
        outputs = {
            "pred_spans": base,
            "refined_spans": refined,
            "refined_quality_logits": quality,
        }
        targets = {"span_labels": [{"spans": torch.tensor([[0.5, 0.4]])}]}
        loss = criterion.loss_stage2(outputs, targets, None)["loss_stage2_quality"]
        loss.backward()
        self.assertIsNone(base.grad)
        self.assertIsNone(refined.grad)
        self.assertGreater(quality.grad.abs().sum().item(), 0)


class TestStage2ModelIntegration(unittest.TestCase):
    @staticmethod
    def _args(use_stage2):
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
            video_input_proj="linear",
            use_stage2=use_stage2, stage2_dim=16, stage2_inner_bins=2,
            stage2_boundary_samples=1, stage2_max_shift_clips=1.0, stage2_shift_frac=0.0,
            stage2_positive_iou=0.2, stage2_start_epoch=10,
            stage2_joint_epoch=30, stage2_boundary_loss_coef=0.5 if use_stage2 else 0.0,
            stage2_giou_loss_coef=0.5 if use_stage2 else 0.0,
            stage2_quality_loss_coef=1.0 if use_stage2 else 0.0,
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

    def test_disabled_stage_preserves_stage1_outputs(self):
        model, criterion = build_model(self._args(use_stage2=False))
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool),
                torch.randn(1, 12, 32), torch.ones(1, 12),
            )
        self.assertNotIn("refined_spans", outputs)
        self.assertNotIn("loss_stage2_quality", criterion.weight_dict)

    def test_enabled_stage_exposes_paired_outputs_and_losses(self):
        model, criterion = build_model(self._args(use_stage2=True))
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool),
                torch.randn(1, 12, 32), torch.ones(1, 12),
            )
        self.assertEqual(outputs["refined_spans"].shape, (1, 5, 2))
        self.assertEqual(outputs["refined_quality_logits"].shape, (1, 5))
        self.assertEqual(outputs["refined_scores"].shape, (1, 5))
        self.assertIn("loss_stage2_boundary", criterion.weight_dict)
        self.assertIn("loss_stage2_giou", criterion.weight_dict)
        self.assertIn("loss_stage2_quality", criterion.weight_dict)


if __name__ == "__main__":
    unittest.main()
