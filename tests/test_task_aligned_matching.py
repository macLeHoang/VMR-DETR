import unittest

import torch

from vmr_detr.modeling.matcher import HungarianMatcher, TaskAlignedMatcher
from vmr_detr.modeling.model import (
    SetCriterion,
    VMRDETR,
    _decode_fdr_cumulative_outputs,
    fdr_logits_to_spans,
    fdr_offset_support,
    init_temporal_queries,
)
from vmr_detr.ops.span_utils import span_xx_to_cxw


def _logits_from_fg_scores(scores):
    scores = torch.as_tensor(scores, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.logit(scores).unsqueeze(-1)


def _cxw(windows):
    return span_xx_to_cxw(torch.as_tensor(windows, dtype=torch.float32))


def _manual_vfl_loss(fg_logits, target_score, positive, alpha=0.75, gamma=2.0):
    src_score = torch.sigmoid(fg_logits)
    negative_weight = alpha * src_score.pow(gamma)
    weight = negative_weight * (~positive).to(fg_logits.dtype) + target_score
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        fg_logits, target_score, reduction="none"
    )
    return (bce * weight).mean()


class _DummyTransformer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.d_model = hidden_dim
        self.nhead = 1


class _FixedSpanEmbed(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, hs):
        return self.logits.to(device=hs.device, dtype=hs.dtype)


class _FixedFDRVMRDETR(VMRDETR):
    def __init__(self, refs, delta_logits):
        hidden_dim = 4
        super().__init__(
            transformer=_DummyTransformer(hidden_dim),
            position_embed=None,
            txt_position_embed=None,
            txt_dim=hidden_dim,
            vid_dim=hidden_dim,
            num_queries=refs.shape[2],
            input_dropout=0.0,
            aux_loss=True,
            max_v_l=75,
            span_loss_type="fdr",
            n_input_proj=1,
            fdr_num_bins=delta_logits.shape[-1] // 2,
            fdr_reg_scale=1.5,
            fdr_min_ref_width=1.0 / 75,
        )
        self.register_buffer("refs", refs)
        self.span_embed = _FixedSpanEmbed(delta_logits)

    def _run_text_video_transformer(self, src_vid, src_vid_mask, src_txt, src_txt_mask):
        bsz = src_vid.shape[0]
        n_layers = self.refs.shape[0]
        refs = self.refs.to(device=src_vid.device, dtype=src_vid.dtype).expand(-1, bsz, -1, -1)
        hs = torch.zeros(n_layers, bsz, self.num_queries, self.hidden_dim, device=src_vid.device)
        vid_mem = torch.zeros(bsz, src_vid.shape[1], self.hidden_dim, device=src_vid.device)
        memory_global = torch.zeros(bsz, self.hidden_dim, device=src_vid.device)
        txt_mem = torch.zeros(bsz, src_txt.shape[1], self.hidden_dim, device=src_vid.device)
        return hs, refs, vid_mem, memory_global, txt_mem


class TemporalQueryInitTest(unittest.TestCase):
    def test_temporal_query_helper_uses_multiscale_widths(self):
        refs = init_temporal_queries(7).sigmoid()

        expected_centers = torch.linspace(0.05, 0.95, 7)
        expected_widths = torch.tensor([0.1, 0.3, 0.6, 0.1, 0.3, 0.6, 0.1])

        self.assertEqual(refs.shape, (7, 2))
        self.assertTrue(torch.allclose(refs[:, 0], expected_centers, atol=1e-6))
        self.assertTrue(torch.allclose(refs[:, 1], expected_widths, atol=1e-6))

    def test_temporal_query_helper_builds_valid_custom_anchor_grid(self):
        refs = init_temporal_queries(9, anchor_widths=[0.1, 0.3, 0.6]).sigmoid()
        starts = refs[:, 0] - 0.5 * refs[:, 1]
        ends = refs[:, 0] + 0.5 * refs[:, 1]

        self.assertEqual(refs.shape, (9, 2))
        self.assertTrue(torch.all(starts >= -1e-6))
        self.assertTrue(torch.all(ends <= 1 + 1e-6))
        expected_widths = torch.tensor([0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6])
        self.assertTrue(torch.allclose(torch.sort(refs[:, 1]).values, expected_widths, atol=1e-6))

    def test_temporal_query_helper_handles_uneven_and_clamped_custom_widths(self):
        refs = init_temporal_queries(10, anchor_widths=[0.0, 0.1, 0.1, 2.0]).sigmoid()
        starts = refs[:, 0] - 0.5 * refs[:, 1]
        ends = refs[:, 0] + 0.5 * refs[:, 1]

        self.assertEqual(refs.shape, (10, 2))
        self.assertTrue(torch.all(starts >= -1e-6))
        self.assertTrue(torch.all(ends <= 1 + 1e-6))
        self.assertEqual(torch.unique(torch.round(refs[:, 1] * 1000)).numel(), 3)

    def test_model_can_initialize_query_refs_from_temporal_anchors(self):
        model = VMRDETR(
            transformer=_DummyTransformer(hidden_dim=4),
            position_embed=None,
            txt_position_embed=None,
            txt_dim=4,
            vid_dim=4,
            num_queries=4,
            input_dropout=0.0,
            max_v_l=75,
            n_input_proj=1,
            query_init="temporal_anchors",
        )

        expected_refs = init_temporal_queries(4).sigmoid()

        self.assertEqual(model.query_init, "temporal_anchors")
        self.assertTrue(torch.allclose(model.query_embed.weight.detach().sigmoid(), expected_refs, atol=1e-6))

    def test_model_can_initialize_query_refs_from_custom_temporal_anchors(self):
        model = VMRDETR(
            transformer=_DummyTransformer(hidden_dim=4),
            position_embed=None,
            txt_position_embed=None,
            txt_dim=4,
            vid_dim=4,
            num_queries=5,
            input_dropout=0.0,
            max_v_l=75,
            n_input_proj=1,
            query_init="temporal_anchors",
            query_anchor_widths="0.2,0.5",
        )

        expected_refs = init_temporal_queries(5, anchor_widths=[0.2, 0.5]).sigmoid()

        self.assertEqual(model.query_init, "temporal_anchors")
        self.assertTrue(torch.allclose(model.query_embed.weight.detach().sigmoid(), expected_refs, atol=1e-6))

    def test_invalid_query_init_raises(self):
        with self.assertRaisesRegex(ValueError, "query_init"):
            VMRDETR(
                transformer=_DummyTransformer(hidden_dim=4),
                position_embed=None,
                txt_position_embed=None,
                txt_dim=4,
                vid_dim=4,
                num_queries=4,
                input_dropout=0.0,
                max_v_l=75,
                n_input_proj=1,
                query_init="unsupported",
            )


class TaskAlignedMatcherTest(unittest.TestCase):
    def test_alignment_prefers_iou_over_high_class_score(self):
        matcher = TaskAlignedMatcher(topk=1, alpha=1.0, beta=6.0)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.99, 0.50, 0.60]).unsqueeze(0),
            "pred_spans": _cxw([[0.0, 0.2], [0.4, 0.6], [0.35, 0.65]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.4, 0.6]]))]}

        indices = matcher(outputs, targets)

        self.assertEqual(indices[0][0].tolist(), [1])
        self.assertEqual(indices[0][1].tolist(), [0])

    def test_topk_selects_multiple_queries_per_target(self):
        matcher = TaskAlignedMatcher(topk=2, alpha=1.0, beta=1.0)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.70, 0.10]).unsqueeze(0),
            "pred_spans": _cxw([[0.4, 0.6], [0.38, 0.62], [0.0, 0.1]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.4, 0.6]]))]}

        indices = matcher(outputs, targets)

        self.assertEqual(indices[0][0].tolist(), [0, 1])
        self.assertEqual(indices[0][1].tolist(), [0, 0])

    def test_duplicate_query_keeps_highest_alignment_target(self):
        matcher = TaskAlignedMatcher(topk=1, alpha=1.0, beta=1.0)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.90, 0.10]).unsqueeze(0),
            "pred_spans": _cxw([[0.2, 0.4], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.4], [0.25, 0.45]]))]}

        indices = matcher(outputs, targets)

        self.assertEqual(indices[0][0].tolist(), [0])
        self.assertEqual(indices[0][1].tolist(), [0])

    def test_empty_targets_return_empty_indices(self):
        matcher = TaskAlignedMatcher(topk=2)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.70]).unsqueeze(0),
            "pred_spans": _cxw([[0.4, 0.6], [0.38, 0.62]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=torch.empty(0, 2))]}

        indices = matcher(outputs, targets)

        self.assertEqual(indices[0][0].numel(), 0)
        self.assertEqual(indices[0][1].numel(), 0)

    def test_rejects_discrete_ce_spans(self):
        with self.assertRaisesRegex(ValueError, "span_loss_type"):
            TaskAlignedMatcher(span_loss_type="ce")

    def test_accepts_fdr_spans(self):
        matcher = TaskAlignedMatcher(span_loss_type="fdr", topk=1)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0),
            "pred_spans": _cxw([[0.4, 0.6]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.4, 0.6]]))]}

        indices = matcher(outputs, targets)

        self.assertEqual(indices[0][0].tolist(), [0])
        self.assertEqual(indices[0][1].tolist(), [0])


class TaskAlignedCriterionTest(unittest.TestCase):
    def _criterion(self, matcher, matching_type, **kwargs):
        return SetCriterion(
            matcher=matcher,
            weight_dict={"loss_span": 1.0, "loss_giou": 1.0, "loss_label": 1.0},
            eos_coef=0.1,
            losses=["spans", "labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
            matching_type=matching_type,
            tal_alpha=1.0,
            tal_beta=6.0,
            **kwargs,
        )

    def test_hungarian_mode_keeps_existing_loss_keys(self):
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.30]).unsqueeze(0),
            "pred_spans": _cxw([[0.4, 0.6], [0.0, 0.2]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.4, 0.6]]))]}
        criterion = self._criterion(HungarianMatcher(), "hungarian")

        losses = criterion(outputs, targets)

        self.assertIn("loss_label", losses)
        self.assertIn("loss_span", losses)
        self.assertIn("loss_giou", losses)
        self.assertTrue(torch.isfinite(losses["loss_label"]))

    def test_hungarian_quality_targets_use_softened_matched_iou_after_ramp(self):
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.30]).unsqueeze(0),
            "pred_spans": _cxw([[0.0, 0.6], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_warmup_epoch=0,
            quality_label_ramp_epoch=0,
        )
        criterion.set_epoch(1)

        quality = criterion._hungarian_quality_targets(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        self.assertTrue(torch.allclose(quality[0, 0], torch.tensor(0.75)))
        self.assertEqual(quality[0, 1].item(), 0.0)

    def test_hungarian_quality_targets_apply_iou_gamma(self):
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.30]).unsqueeze(0),
            "pred_spans": _cxw([[0.0, 0.6], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_iou_gamma=2.0,
            quality_label_warmup_epoch=0,
            quality_label_ramp_epoch=0,
        )
        criterion.set_epoch(1)

        quality = criterion._hungarian_quality_targets(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        self.assertTrue(torch.allclose(quality[0, 0], torch.tensor(0.625)))
        self.assertEqual(quality[0, 1].item(), 0.0)

    def test_hungarian_quality_target_is_one_for_perfect_match(self):
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.30]).unsqueeze(0),
            "pred_spans": _cxw([[0.2, 0.8], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_iou_gamma=2.0,
            quality_label_warmup_epoch=0,
            quality_label_ramp_epoch=0,
        )
        criterion.set_epoch(1)

        quality = criterion._hungarian_quality_targets(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        self.assertTrue(torch.allclose(quality[0, 0], torch.tensor(1.0)))

    def test_hungarian_quality_labels_keep_eos_weight_for_unmatched_queries(self):
        outputs = {
            "pred_logits": torch.zeros(1, 2, 1),
            "pred_spans": _cxw([[0.2, 0.8], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_warmup_epoch=0,
            quality_label_ramp_epoch=0,
        )
        criterion.eos_coef = 0.25
        criterion.set_epoch(1)

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        expected = torch.log(torch.tensor(2.0)) * (1.0 + criterion.eos_coef) / 2.0
        self.assertTrue(torch.allclose(losses["loss_label"], expected))

    def test_hungarian_quality_labels_detach_iou_targets_from_span_gradients(self):
        pred_logits = torch.zeros(1, 1, 1, requires_grad=True)
        pred_spans = torch.tensor([[[0.3, 0.6]]], requires_grad=True)
        outputs = {
            "pred_logits": pred_logits,
            "pred_spans": pred_spans,
        }
        targets = {"span_labels": [dict(spans=torch.tensor([[0.5, 0.6]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_warmup_epoch=0,
            quality_label_ramp_epoch=0,
        )
        criterion.set_epoch(1)

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )
        losses["loss_label"].backward()

        self.assertIsNotNone(pred_logits.grad)
        self.assertIsNone(pred_spans.grad)

    def test_hungarian_quality_labels_ramp_from_hard_positive_to_softened_iou(self):
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0),
            "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_warmup_epoch=10,
            quality_label_ramp_epoch=30,
        )
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        criterion.set_epoch(5)
        warmup_quality = criterion._hungarian_quality_targets(outputs, targets, indices)
        criterion.set_epoch(20)
        mid_ramp_quality = criterion._hungarian_quality_targets(outputs, targets, indices)
        criterion.set_epoch(30)
        final_quality = criterion._hungarian_quality_targets(outputs, targets, indices)

        self.assertTrue(torch.allclose(warmup_quality[0, 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(mid_ramp_quality[0, 0], torch.tensor(0.875)))
        self.assertTrue(torch.allclose(final_quality[0, 0], torch.tensor(0.75)))

    def test_hungarian_quality_labels_ramp_to_powered_iou(self):
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0),
            "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_iou_gamma=2.0,
            quality_label_warmup_epoch=10,
            quality_label_ramp_epoch=30,
        )
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        criterion.set_epoch(5)
        warmup_quality = criterion._hungarian_quality_targets(outputs, targets, indices)
        criterion.set_epoch(20)
        mid_ramp_quality = criterion._hungarian_quality_targets(outputs, targets, indices)
        criterion.set_epoch(30)
        final_quality = criterion._hungarian_quality_targets(outputs, targets, indices)

        self.assertTrue(torch.allclose(warmup_quality[0, 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(mid_ramp_quality[0, 0], torch.tensor(0.8125)))
        self.assertTrue(torch.allclose(final_quality[0, 0], torch.tensor(0.625)))

    def test_hungarian_quality_loss_rejects_non_positive_iou_gamma(self):
        with self.assertRaisesRegex(ValueError, "quality_label_iou_gamma"):
            self._criterion(
                HungarianMatcher(),
                "hungarian",
                quality_label_iou_gamma=0.0,
            )

    def test_hungarian_quality_loss_rejects_tal_matching(self):
        with self.assertRaisesRegex(ValueError, "label_loss_type"):
            self._criterion(
                TaskAlignedMatcher(),
                "tal",
                label_loss_type="quality",
            )

    def test_hungarian_quality_labels_mirror_main_loss_for_auxiliary_outputs(self):
        logits = _logits_from_fg_scores([0.80]).unsqueeze(0)
        outputs = {
            "pred_logits": logits,
            "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
            "aux_outputs": [
                {
                    "pred_logits": logits,
                    "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
                }
            ],
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(),
            weight_dict={"loss_label": 1.0, "loss_label_0": 1.0},
            eos_coef=0.1,
            losses=["labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
            matching_type="hungarian",
            label_loss_type="quality",
            quality_label_strength=0.5,
            quality_label_warmup_epoch=0,
            quality_label_ramp_epoch=0,
        )
        criterion.set_epoch(1)

        losses = criterion(outputs, targets)

        fg_logit = logits.squeeze(-1)
        expected_final = torch.nn.functional.binary_cross_entropy_with_logits(
            fg_logit,
            torch.full_like(fg_logit, 0.75),
        )
        expected_aux = expected_final
        self.assertTrue(torch.allclose(losses["loss_label"], expected_final))
        self.assertTrue(torch.allclose(losses["loss_label_0"], expected_aux))

    def test_vfl_label_loss_uses_iou_targets_and_focal_negative_weight(self):
        outputs = {
            "pred_logits": torch.zeros(1, 2, 1),
            "pred_spans": _cxw([[0.2, 0.8], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="vfl",
            vfl_alpha=0.75,
            vfl_gamma=2.0,
        )
        criterion.eos_coef = 0.25

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        fg_logit = outputs["pred_logits"].squeeze(-1)
        target_score = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[True, False]])
        expected = _manual_vfl_loss(
            fg_logit,
            target_score,
            positive,
            alpha=criterion.vfl_alpha,
            gamma=criterion.vfl_gamma,
        )
        self.assertTrue(torch.allclose(losses["loss_label"], expected))

    def test_vfl_label_loss_uses_raw_matched_iou_for_partial_overlap(self):
        outputs = {
            "pred_logits": torch.zeros(1, 1, 1),
            "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="vfl",
        )

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        fg_logit = outputs["pred_logits"].squeeze(-1)
        target_score = torch.tensor([[0.5]])
        positive = torch.tensor([[True]])
        expected = _manual_vfl_loss(
            fg_logit,
            target_score,
            positive,
            alpha=criterion.vfl_alpha,
            gamma=criterion.vfl_gamma,
        )
        self.assertTrue(torch.allclose(losses["loss_label"], expected))

    def test_vfl_label_loss_detaches_iou_targets_from_span_gradients(self):
        pred_logits = torch.zeros(1, 1, 1, requires_grad=True)
        pred_spans = torch.tensor([[[0.3, 0.6]]], requires_grad=True)
        outputs = {
            "pred_logits": pred_logits,
            "pred_spans": pred_spans,
        }
        targets = {"span_labels": [dict(spans=torch.tensor([[0.5, 0.6]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="vfl",
        )

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )
        losses["loss_label"].backward()

        self.assertIsNotNone(pred_logits.grad)
        self.assertIsNone(pred_spans.grad)

    def test_vfl_label_loss_backpropagates_through_negative_focal_weight(self):
        pred_logits = torch.tensor([[[0.0], [1.0]]], requires_grad=True)
        outputs = {
            "pred_logits": pred_logits,
            "pred_spans": _cxw([[0.2, 0.8], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="vfl",
            vfl_alpha=0.75,
            vfl_gamma=2.0,
        )

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )
        losses["loss_label"].backward()

        expected_logits = pred_logits.detach().clone().requires_grad_()
        expected_fg = expected_logits.squeeze(-1)
        target_score = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[True, False]])
        expected_loss = _manual_vfl_loss(
            expected_fg,
            target_score,
            positive,
            alpha=criterion.vfl_alpha,
            gamma=criterion.vfl_gamma,
        )
        expected_loss.backward()

        detached_logits = pred_logits.detach().clone().requires_grad_()
        detached_fg = detached_logits.squeeze(-1)
        detached_score = torch.sigmoid(detached_fg.detach())
        detached_weight = (
            criterion.vfl_alpha * detached_score.pow(criterion.vfl_gamma)
        ) * (~positive).to(detached_fg.dtype) + target_score
        detached_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            detached_fg, target_score, reduction="none"
        )
        detached_loss = (detached_bce * detached_weight).mean()
        detached_loss.backward()

        self.assertTrue(torch.allclose(pred_logits.grad, expected_logits.grad))
        self.assertFalse(
            torch.allclose(pred_logits.grad[0, 1, 0], detached_logits.grad[0, 1, 0])
        )

    def test_vfl_label_loss_handles_empty_matches_as_negatives(self):
        outputs = {
            "pred_logits": torch.zeros(1, 2, 1),
            "pred_spans": _cxw([[0.2, 0.8], [0.8, 1.0]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=torch.empty(0, 2))]}
        criterion = self._criterion(
            HungarianMatcher(),
            "hungarian",
            label_loss_type="vfl",
            vfl_alpha=0.75,
            vfl_gamma=2.0,
        )

        losses = criterion.loss_labels(
            outputs,
            targets,
            [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))],
        )

        fg_logit = outputs["pred_logits"].squeeze(-1)
        target_score = torch.zeros_like(fg_logit)
        positive = torch.zeros_like(fg_logit, dtype=torch.bool)
        expected = _manual_vfl_loss(
            fg_logit,
            target_score,
            positive,
            alpha=criterion.vfl_alpha,
            gamma=criterion.vfl_gamma,
        )
        self.assertTrue(torch.allclose(losses["loss_label"], expected))

    def test_vfl_label_loss_mirrors_main_loss_for_auxiliary_outputs(self):
        logits = _logits_from_fg_scores([0.80]).unsqueeze(0)
        outputs = {
            "pred_logits": logits,
            "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
            "aux_outputs": [
                {
                    "pred_logits": logits,
                    "pred_spans": _cxw([[0.0, 0.6]]).unsqueeze(0),
                }
            ],
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.8]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(),
            weight_dict={"loss_label": 1.0, "loss_label_0": 1.0},
            eos_coef=0.1,
            losses=["labels"],
            temperature=0.07,
            span_loss_type="l1",
            max_v_l=75,
            matching_type="hungarian",
            label_loss_type="vfl",
        )

        losses = criterion(outputs, targets)

        fg_logit = logits.squeeze(-1)
        target_score = torch.full_like(fg_logit, 0.5)
        expected_final = _manual_vfl_loss(
            fg_logit,
            target_score,
            torch.ones_like(fg_logit, dtype=torch.bool),
            alpha=criterion.vfl_alpha,
            gamma=criterion.vfl_gamma,
        )
        expected_aux = expected_final
        self.assertTrue(torch.allclose(losses["loss_label"], expected_final))
        self.assertTrue(torch.allclose(losses["loss_label_0"], expected_aux))

    def test_vfl_label_loss_rejects_tal_matching(self):
        with self.assertRaisesRegex(ValueError, "label_loss_type"):
            self._criterion(
                TaskAlignedMatcher(),
                "tal",
                label_loss_type="vfl",
            )

    def test_vfl_label_loss_rejects_discrete_ce_spans(self):
        with self.assertRaisesRegex(ValueError, "label_loss_type"):
            SetCriterion(
                matcher=HungarianMatcher(span_loss_type="ce"),
                weight_dict={"loss_label": 1.0},
                eos_coef=0.1,
                losses=["labels"],
                temperature=0.07,
                span_loss_type="ce",
                max_v_l=75,
                label_loss_type="vfl",
            )

    def test_tal_mode_returns_finite_losses_and_backpropagates(self):
        pred_logits = _logits_from_fg_scores([0.80, 0.70, 0.20]).unsqueeze(0).requires_grad_()
        pred_spans = _cxw([[0.4, 0.6], [0.35, 0.65], [0.0, 0.2]]).unsqueeze(0).requires_grad_()
        outputs = {
            "pred_logits": pred_logits,
            "pred_spans": pred_spans,
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.4, 0.6]]))]}
        criterion = self._criterion(TaskAlignedMatcher(topk=2), "tal")

        losses = criterion(outputs, targets)
        total_loss = losses["loss_label"] + losses["loss_span"] + losses["loss_giou"]
        total_loss.backward()

        self.assertTrue(torch.isfinite(total_loss))
        self.assertIsNotNone(pred_logits.grad)
        self.assertIsNotNone(pred_spans.grad)

    def test_tal_criterion_rejects_discrete_ce_spans(self):
        with self.assertRaisesRegex(ValueError, "TAL matching requires"):
            SetCriterion(
                matcher=TaskAlignedMatcher(),
                weight_dict={"loss_span": 1.0, "loss_giou": 1.0, "loss_label": 1.0},
                eos_coef=0.1,
                losses=["spans", "labels"],
                temperature=0.07,
                span_loss_type="ce",
                max_v_l=75,
                matching_type="tal",
            )

class TemporalFDRTest(unittest.TestCase):
    def test_support_is_symmetric_sorted_and_dense_near_zero(self):
        support = fdr_offset_support(32, 1.5)

        self.assertEqual(support.numel(), 32)
        self.assertTrue(torch.all(support[1:] > support[:-1]))
        self.assertTrue(torch.allclose(support, -support.flip(0), atol=1e-6))
        center_gap = support[16] - support[15]
        edge_gap = support[-1] - support[-2]
        self.assertLess(center_gap, edge_gap)

    def test_zero_logits_decode_to_reference_span(self):
        refs = _cxw([[0.25, 0.75]]).unsqueeze(0)
        logits = torch.zeros(1, 1, 64)

        pred = fdr_logits_to_spans(logits, refs, 32, 1.5, 1.0 / 75)

        self.assertTrue(torch.allclose(pred, refs, atol=1e-6))

    def test_fdr_span_head_initializes_to_zero_residual_logits(self):
        model = VMRDETR(
            transformer=_DummyTransformer(hidden_dim=4),
            position_embed=None,
            txt_position_embed=None,
            txt_dim=4,
            vid_dim=4,
            num_queries=2,
            input_dropout=0.0,
            max_v_l=75,
            span_loss_type="fdr",
            n_input_proj=1,
            fdr_num_bins=32,
            fdr_min_ref_width=0.05,
        )

        span_logits = model.span_embed(torch.randn(3, 1, 2, 4))

        self.assertEqual(model.class_embed.out_features, 1)
        self.assertTrue(torch.allclose(span_logits, torch.zeros_like(span_logits)))

    def test_boundary_offsets_expand_reference_span(self):
        refs = _cxw([[0.4, 0.6]]).unsqueeze(0)
        logits = torch.zeros(1, 1, 2, 32)
        logits[..., 0, 0] = 20.0
        logits[..., 1, -1] = 20.0

        pred = fdr_logits_to_spans(logits.reshape(1, 1, 64), refs, 32, 1.5, 1.0 / 75)
        pred_xx = pred[..., 0] - pred[..., 1] * 0.5, pred[..., 0] + pred[..., 1] * 0.5

        self.assertLess(pred_xx[0].item(), 0.4)
        self.assertGreater(pred_xx[1].item(), 0.6)

    def test_cumulative_fdr_outputs_use_initial_reference_for_all_layers(self):
        fdr_num_bins = 32
        refs = _cxw([[0.20, 0.40], [0.55, 0.75], [0.05, 0.25]])[:, None, None, :]
        delta_logits = torch.zeros(3, 1, 1, 2 * fdr_num_bins)
        delta_logits[0, 0, 0, 0] = 8.0
        delta_logits[1, 0, 0, fdr_num_bins - 1] = 4.0
        delta_logits[2, 0, 0, fdr_num_bins + fdr_num_bins // 2] = 6.0

        span_logits, spans, span_refs = _decode_fdr_cumulative_outputs(
            delta_logits,
            refs,
            fdr_num_bins,
            1.5,
            1.0 / 75,
        )

        expected_logits = torch.cumsum(delta_logits, dim=0)
        initial_refs = refs[:1].expand_as(refs)
        expected_spans = fdr_logits_to_spans(expected_logits, initial_refs, fdr_num_bins, 1.5, 1.0 / 75)
        current_ref_spans = fdr_logits_to_spans(expected_logits, refs, fdr_num_bins, 1.5, 1.0 / 75)

        self.assertTrue(torch.allclose(span_logits, expected_logits))
        self.assertTrue(torch.allclose(spans, expected_spans))
        self.assertTrue(torch.allclose(span_refs, initial_refs))
        self.assertFalse(torch.allclose(spans[1], current_ref_spans[1]))

    def test_fdr_forward_exports_initial_reference_for_aux_outputs(self):
        fdr_num_bins = 32
        refs = _cxw([[0.20, 0.40], [0.55, 0.75], [0.05, 0.25]])[:, None, None, :]
        delta_logits = torch.zeros(3, 1, 1, 2 * fdr_num_bins)
        delta_logits[0, 0, 0, 0] = 8.0
        delta_logits[1, 0, 0, fdr_num_bins - 1] = 4.0
        delta_logits[2, 0, 0, fdr_num_bins + fdr_num_bins // 2] = 6.0
        model = _FixedFDRVMRDETR(refs, delta_logits)

        outputs = model(
            src_txt=torch.zeros(1, 1, 4),
            src_txt_mask=torch.ones(1, 1),
            src_vid=torch.zeros(1, 2, 4),
            src_vid_mask=torch.ones(1, 2),
        )

        expected_logits = torch.cumsum(delta_logits, dim=0)
        initial_refs = refs[:1].expand_as(refs)

        self.assertTrue(torch.allclose(outputs["pred_span_logits"], expected_logits[-1]))
        self.assertTrue(torch.allclose(outputs["pred_span_refs"], initial_refs[-1]))
        for i, aux_output in enumerate(outputs["aux_outputs"]):
            self.assertTrue(torch.allclose(aux_output["pred_span_logits"], expected_logits[i]))
            self.assertTrue(torch.allclose(aux_output["pred_span_refs"], initial_refs[i]))

    def test_fdr_criterion_returns_finite_losses_and_backpropagates(self):
        fdr_num_bins = 32
        refs = _cxw([[0.3, 0.7]]).unsqueeze(0)
        pred_logits = torch.zeros(1, 1, 2 * fdr_num_bins, requires_grad=True)
        pred_spans = fdr_logits_to_spans(pred_logits, refs, fdr_num_bins, 1.5, 1.0 / 75)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0).requires_grad_(),
            "pred_spans": pred_spans,
            "pred_span_logits": pred_logits,
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.28, 0.72]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_fgl": 1.0, "loss_giou": 1.0, "loss_label": 1.0},
            eos_coef=0.1,
            losses=["spans", "labels"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
            fdr_num_bins=fdr_num_bins,
            fdr_reg_scale=1.5,
            fdr_min_ref_width=1.0 / 75,
        )

        losses = criterion(outputs, targets)
        total_loss = losses["loss_fgl"] + losses["loss_giou"] + losses["loss_label"]
        total_loss.backward()

        self.assertIn("loss_fgl", losses)
        self.assertIn("loss_giou", losses)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertIsNotNone(pred_logits.grad)

    def test_log_width_loss_is_zero_when_widths_match(self):
        pred_spans = _cxw([[0.2, 0.4]]).unsqueeze(0).requires_grad_()
        outputs = {
            "pred_spans": pred_spans,
            "pred_span_logits": torch.zeros(1, 1, 64),
            "pred_span_refs": _cxw([[0.2, 0.4]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.4]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_width": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
            width_loss_type="log",
        )

        losses = criterion.loss_spans(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))])

        self.assertIn("loss_width", losses)
        self.assertTrue(torch.allclose(losses["loss_width"], torch.tensor(0.0)))

    def test_log_width_loss_penalizes_overlong_predictions(self):
        pred_spans = _cxw([[0.1, 0.5]]).unsqueeze(0).requires_grad_()
        outputs = {
            "pred_spans": pred_spans,
            "pred_span_logits": torch.zeros(1, 1, 64),
            "pred_span_refs": _cxw([[0.1, 0.5]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.4]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_width": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
            width_loss_type="log",
        )

        losses = criterion.loss_spans(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))])

        self.assertGreater(losses["loss_width"].item(), 0.0)

    def test_span_xx_loss_penalizes_start_end_boundary_errors(self):
        pred_spans = _cxw([[0.1, 0.5]]).unsqueeze(0).requires_grad_()
        outputs = {
            "pred_spans": pred_spans,
            "pred_span_logits": torch.zeros(1, 1, 64),
            "pred_span_refs": _cxw([[0.1, 0.5]]).unsqueeze(0),
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.4]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_span_xx": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        losses = criterion.loss_spans(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))])
        losses["loss_span_xx"].backward()

        self.assertIn("loss_span_xx", losses)
        self.assertTrue(torch.allclose(losses["loss_span_xx"], torch.tensor(0.1)))
        self.assertIsNotNone(pred_spans.grad)

    def test_fdr_empty_targets_return_zero_finite_losses(self):
        refs = _cxw([[0.3, 0.7]]).unsqueeze(0)
        pred_logits = torch.zeros(1, 1, 64)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0),
            "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
            "pred_span_logits": pred_logits,
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=torch.empty(0, 2))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_fgl": 1.0, "loss_giou": 1.0, "loss_label": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        losses = criterion(outputs, targets)

        self.assertEqual(losses["loss_fgl"].item(), 0.0)
        self.assertEqual(losses["loss_giou"].item(), 0.0)

    def test_fdr_empty_targets_return_zero_width_loss_when_enabled(self):
        refs = _cxw([[0.3, 0.7]]).unsqueeze(0)
        pred_logits = torch.zeros(1, 1, 64)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0),
            "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
            "pred_span_logits": pred_logits,
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=torch.empty(0, 2))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_fgl": 1.0, "loss_giou": 1.0, "loss_width": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
            width_loss_type="log",
        )

        losses = criterion(outputs, targets)

        self.assertEqual(losses["loss_width"].item(), 0.0)

    def test_fdr_empty_targets_return_zero_span_xx_loss_when_enabled(self):
        refs = _cxw([[0.3, 0.7]]).unsqueeze(0)
        pred_logits = torch.zeros(1, 1, 64)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80]).unsqueeze(0),
            "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
            "pred_span_logits": pred_logits,
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=torch.empty(0, 2))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_fgl": 1.0, "loss_giou": 1.0, "loss_span_xx": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        losses = criterion(outputs, targets)

        self.assertEqual(losses["loss_span_xx"].item(), 0.0)

    def test_width_loss_is_repeated_for_auxiliary_outputs(self):
        refs = _cxw([[0.2, 0.6], [0.1, 0.5]]).unsqueeze(0)
        pred_logits = torch.zeros(1, 2, 64)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.20]).unsqueeze(0),
            "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
            "pred_span_logits": pred_logits,
            "pred_span_refs": refs,
            "aux_outputs": [
                {
                    "pred_logits": _logits_from_fg_scores([0.70, 0.30]).unsqueeze(0),
                    "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
                    "pred_span_logits": pred_logits,
                    "pred_span_refs": refs,
                },
                {
                    "pred_logits": _logits_from_fg_scores([0.60, 0.40]).unsqueeze(0),
                    "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
                    "pred_span_logits": pred_logits,
                    "pred_span_refs": refs,
                },
            ],
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.25, 0.55]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={
                "loss_fgl": 1.0,
                "loss_giou": 1.0,
                "loss_width": 1.0,
                "loss_width_0": 1.0,
                "loss_width_1": 1.0,
                "loss_span_xx": 1.0,
                "loss_span_xx_0": 1.0,
                "loss_span_xx_1": 1.0,
            },
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
            width_loss_type="log",
        )

        losses = criterion(outputs, targets)

        self.assertIn("loss_width", losses)
        self.assertIn("loss_width_0", losses)
        self.assertIn("loss_width_1", losses)
        self.assertIn("loss_span_xx", losses)
        self.assertIn("loss_span_xx_0", losses)
        self.assertIn("loss_span_xx_1", losses)

    def test_go_lsd_global_indices_merge_and_resolve_duplicates(self):
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_golsd": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )
        final_indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        aux_indices = [
            [(torch.tensor([0, 2]), torch.tensor([1, 0]))],
            [(torch.tensor([0, 2]), torch.tensor([1, 0]))],
        ]

        indices_go = criterion._get_global_optimal_indices(final_indices, aux_indices)

        self.assertEqual(indices_go[0][0].tolist(), [0, 1, 2])
        self.assertEqual(indices_go[0][1].tolist(), [1, 1, 0])

    def test_go_lsd_global_indices_handle_empty_targets(self):
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_golsd": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )
        empty_indices = [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))]

        indices_go = criterion._get_global_optimal_indices(empty_indices, [empty_indices])

        self.assertEqual(indices_go[0][0].numel(), 0)
        self.assertEqual(indices_go[0][1].numel(), 0)

    def test_go_lsd_loss_is_zero_for_identical_teacher_logits(self):
        refs = _cxw([[0.2, 0.6], [0.6, 0.8]]).unsqueeze(0)
        logits = torch.randn(1, 2, 64)
        spans = fdr_logits_to_spans(logits, refs, 32, 1.5, 1.0 / 75)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.20]).unsqueeze(0),
            "pred_spans": spans,
            "pred_span_logits": logits,
            "pred_span_refs": refs,
        }
        aux_outputs = {
            "pred_logits": _logits_from_fg_scores([0.70, 0.30]).unsqueeze(0),
            "pred_spans": spans,
            "pred_span_logits": logits.clone().requires_grad_(),
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.6]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_golsd": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        loss = criterion._loss_go_lsd(aux_outputs, outputs, targets, [(torch.tensor([0]), torch.tensor([0]))])

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0), atol=1e-6))

    def test_go_lsd_loss_is_positive_and_detaches_teacher(self):
        refs = _cxw([[0.2, 0.6], [0.6, 0.8]]).unsqueeze(0)
        teacher_logits = torch.zeros(1, 2, 64, requires_grad=True)
        student_logits = torch.zeros(1, 2, 64, requires_grad=True)
        student_logits.data[:, :, 0] = 3.0
        spans = fdr_logits_to_spans(teacher_logits.detach(), refs, 32, 1.5, 1.0 / 75)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.40]).unsqueeze(0),
            "pred_spans": spans,
            "pred_span_logits": teacher_logits,
            "pred_span_refs": refs,
        }
        aux_outputs = {
            "pred_logits": _logits_from_fg_scores([0.70, 0.30]).unsqueeze(0),
            "pred_spans": spans,
            "pred_span_logits": student_logits,
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.6]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_golsd": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        loss = criterion._loss_go_lsd(aux_outputs, outputs, targets, [(torch.tensor([0]), torch.tensor([0]))])
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(student_logits.grad)
        self.assertIsNone(teacher_logits.grad)

    def test_go_lsd_matched_and_unmatched_queries_can_contribute(self):
        refs = _cxw([[0.2, 0.6], [0.6, 0.8]]).unsqueeze(0)
        teacher_logits = torch.zeros(1, 2, 64)
        spans = fdr_logits_to_spans(teacher_logits, refs, 32, 1.5, 1.0 / 75)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.90]).unsqueeze(0),
            "pred_spans": spans,
            "pred_span_logits": teacher_logits,
            "pred_span_refs": refs,
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.6]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_golsd": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        matched_student = teacher_logits.clone()
        matched_student[:, 0, 0] = 4.0
        unmatched_student = teacher_logits.clone()
        unmatched_student[:, 1, 0] = 4.0
        matched_loss = criterion._loss_go_lsd(
            {**outputs, "pred_span_logits": matched_student, "pred_spans": spans},
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )
        unmatched_loss = criterion._loss_go_lsd(
            {**outputs, "pred_span_logits": unmatched_student, "pred_spans": spans},
            outputs,
            targets,
            [(torch.tensor([0]), torch.tensor([0]))],
        )

        self.assertGreater(matched_loss.item(), 0.0)
        self.assertGreater(unmatched_loss.item(), 0.0)

    def test_go_lsd_forward_emits_aux_losses_when_enabled(self):
        refs = _cxw([[0.2, 0.6], [0.6, 0.8]]).unsqueeze(0)
        teacher_logits = torch.zeros(1, 2, 64)
        aux_logits_0 = teacher_logits.clone()
        aux_logits_1 = teacher_logits.clone()
        aux_logits_0[:, :, 0] = 2.0
        aux_logits_1[:, :, 1] = 2.0
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.20]).unsqueeze(0),
            "pred_spans": fdr_logits_to_spans(teacher_logits, refs, 32, 1.5, 1.0 / 75),
            "pred_span_logits": teacher_logits,
            "pred_span_refs": refs,
            "aux_outputs": [
                {
                    "pred_logits": _logits_from_fg_scores([0.70, 0.30]).unsqueeze(0),
                    "pred_spans": fdr_logits_to_spans(aux_logits_0, refs, 32, 1.5, 1.0 / 75),
                    "pred_span_logits": aux_logits_0,
                    "pred_span_refs": refs,
                },
                {
                    "pred_logits": _logits_from_fg_scores([0.60, 0.40]).unsqueeze(0),
                    "pred_spans": fdr_logits_to_spans(aux_logits_1, refs, 32, 1.5, 1.0 / 75),
                    "pred_span_logits": aux_logits_1,
                    "pred_span_refs": refs,
                },
            ],
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.6]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={
                "loss_fgl": 1.0,
                "loss_giou": 1.0,
                "loss_golsd": 1.0,
                "loss_golsd_0": 1.0,
                "loss_golsd_1": 1.0,
            },
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        losses = criterion(outputs, targets)

        self.assertIn("loss_golsd_0", losses)
        self.assertIn("loss_golsd_1", losses)
        self.assertTrue(torch.isfinite(losses["loss_golsd_0"]))
        self.assertTrue(torch.isfinite(losses["loss_golsd_1"]))

    def test_go_lsd_forward_omits_aux_losses_when_disabled(self):
        refs = _cxw([[0.2, 0.6], [0.6, 0.8]]).unsqueeze(0)
        pred_logits = torch.zeros(1, 2, 64)
        outputs = {
            "pred_logits": _logits_from_fg_scores([0.80, 0.20]).unsqueeze(0),
            "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
            "pred_span_logits": pred_logits,
            "pred_span_refs": refs,
            "aux_outputs": [
                {
                    "pred_logits": _logits_from_fg_scores([0.70, 0.30]).unsqueeze(0),
                    "pred_spans": fdr_logits_to_spans(pred_logits, refs, 32, 1.5, 1.0 / 75),
                    "pred_span_logits": pred_logits,
                    "pred_span_refs": refs,
                }
            ],
        }
        targets = {"span_labels": [dict(spans=_cxw([[0.2, 0.6]]))]}
        criterion = SetCriterion(
            matcher=HungarianMatcher(span_loss_type="fdr"),
            weight_dict={"loss_fgl": 1.0, "loss_giou": 1.0},
            eos_coef=0.1,
            losses=["spans"],
            temperature=0.07,
            span_loss_type="fdr",
            max_v_l=75,
        )

        losses = criterion(outputs, targets)

        self.assertNotIn("loss_golsd_0", losses)

    def test_go_lsd_rejects_non_fdr_spans(self):
        with self.assertRaisesRegex(ValueError, "GO-LSD"):
            SetCriterion(
                matcher=HungarianMatcher(span_loss_type="l1"),
                weight_dict={"loss_golsd": 1.0},
                eos_coef=0.1,
                losses=["spans"],
                temperature=0.07,
                span_loss_type="l1",
                max_v_l=75,
            )

    def test_span_xx_loss_rejects_discrete_ce_spans(self):
        with self.assertRaisesRegex(ValueError, "loss_span_xx"):
            SetCriterion(
                matcher=HungarianMatcher(span_loss_type="ce"),
                weight_dict={"loss_span_xx": 1.0},
                eos_coef=0.1,
                losses=["spans"],
                temperature=0.07,
                span_loss_type="ce",
                max_v_l=75,
            )


if __name__ == "__main__":
    unittest.main()
