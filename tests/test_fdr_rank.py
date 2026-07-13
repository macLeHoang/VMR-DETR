"""Focused tests for the opt-in FDR-Rank proposal reranker."""

import unittest

import torch

from vmr_detr.modeling.model import SetCriterion
from vmr_detr.modeling.ranking import (
    FDRRanker,
    fdr_distribution_features,
    pairwise_temporal_features,
    threshold_aware_gain,
)


class TestFDRRankFeatures(unittest.TestCase):
    def test_fdr_posteriors_and_statistics_are_finite(self):
        logits = torch.randn(2, 4, 8)
        probabilities, statistics = fdr_distribution_features(logits, 4)
        self.assertEqual(probabilities.shape, logits.shape)
        self.assertEqual(statistics.shape, (2, 4, 10))
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(torch.isfinite(statistics).all())
        self.assertTrue(torch.allclose(
            probabilities.reshape(2, 4, 2, 4).sum(-1),
            torch.ones(2, 4, 2),
            atol=1e-6,
        ))

    def test_pairwise_features_are_symmetric(self):
        spans = torch.tensor([[[0.2, 0.2], [0.5, 0.3], [0.8, 0.1]]])
        features = pairwise_temporal_features(spans)
        self.assertTrue(torch.allclose(features, features.transpose(1, 2)))
        self.assertTrue(torch.allclose(features[0, :, :, 0].diag(), torch.ones(3)))

    def test_threshold_gain_is_monotonic(self):
        quality = torch.tensor([[0.1, 0.5, 0.9]])
        gain = threshold_aware_gain(quality)
        self.assertLess(gain[0, 0].item(), gain[0, 1].item())
        self.assertLess(gain[0, 1].item(), gain[0, 2].item())

    def test_ranker_is_query_permutation_equivariant(self):
        torch.manual_seed(7)
        ranker = FDRRanker(
            hidden_dim=8,
            fdr_num_bins=4,
            rank_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        ranker.eval()
        hs = torch.randn(2, 5, 8)
        spans = torch.rand(2, 5, 2)
        spans[..., 1] = spans[..., 1] * 0.4 + 0.1
        logits = torch.randn(2, 5, 1)
        quality = torch.randn(2, 5)
        fdr_logits = torch.randn(2, 5, 8)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        with torch.no_grad():
            original = ranker(hs, spans, logits, quality, fdr_logits)
            permuted = ranker(
                hs[:, permutation],
                spans[:, permutation],
                logits[:, permutation],
                quality[:, permutation],
                fdr_logits[:, permutation],
            )
        self.assertTrue(torch.allclose(permuted, original[:, permutation], atol=1e-6))


class TestFDRRankLoss(unittest.TestCase):
    @staticmethod
    def _criterion(epoch):
        criterion = object.__new__(SetCriterion)
        criterion.current_epoch = epoch
        criterion.rank_start_epoch = 20
        criterion.rank_ramp_epochs = 10
        criterion.rank_gain_tau = 0.05
        criterion.rank_target_tau = 0.1
        criterion.rank_score_tau = 0.2
        return criterion

    @staticmethod
    def _outputs():
        refined = torch.tensor(
            [[[0.5, 0.2], [0.5, 0.5], [0.2, 0.1]]], requires_grad=True
        )
        return {
            "pred_spans": refined.detach().clone(),
            "refined_spans": refined,
            "pred_rank_logits": torch.zeros(1, 3, requires_grad=True),
        }

    def test_rank_loss_is_zero_before_warmup(self):
        criterion = self._criterion(epoch=19)
        outputs = self._outputs()
        losses = criterion.loss_fdr_rank(
            outputs,
            {"span_labels": [{"spans": torch.tensor([[0.5, 0.2]])}]},
        )
        self.assertEqual(losses["loss_rank_listwise"].item(), 0.0)
        self.assertEqual(losses["loss_rank_quality"].item(), 0.0)

    def test_rank_loss_uses_detached_iou_targets(self):
        criterion = self._criterion(epoch=30)
        outputs = self._outputs()
        losses = criterion.loss_fdr_rank(
            outputs,
            {"span_labels": [{"spans": torch.tensor([[0.5, 0.2]])}]},
        )
        total = losses["loss_rank_listwise"] + losses["loss_rank_quality"]
        total.backward()
        self.assertIsNone(outputs["refined_spans"].grad)
        self.assertGreater(outputs["pred_rank_logits"].grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
