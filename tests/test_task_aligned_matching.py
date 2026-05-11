import unittest

import torch

from vmr_detr.modeling.matcher import HungarianMatcher, TaskAlignedMatcher
from vmr_detr.modeling.model import SetCriterion
from vmr_detr.ops.span_utils import span_xx_to_cxw


def _logits_from_fg_scores(scores):
    scores = torch.as_tensor(scores, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    fg_logits = torch.logit(scores)
    bg_logits = torch.zeros_like(fg_logits)
    return torch.stack([fg_logits, bg_logits], dim=-1)


def _cxw(windows):
    return span_xx_to_cxw(torch.as_tensor(windows, dtype=torch.float32))


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


class TaskAlignedCriterionTest(unittest.TestCase):
    def _criterion(self, matcher, matching_type):
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


if __name__ == "__main__":
    unittest.main()
