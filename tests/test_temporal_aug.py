"""CPU tests for train-only temporal-crop data augmentation."""

import os
import random
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

from vmr_detr.data.start_end_dataset import StartEndDataset
from vmr_detr.data.start_end_dataset_audio import StartEndDataset_audio
from vmr_detr.config.options import BaseOptions
from vmr_detr.cli.train_utils import has_nonuniform_weights


def _make_dataset(prob=1.0, use_tef=False):
    dataset = StartEndDataset.__new__(StartEndDataset)
    dataset.dset_name = "charades_sta"
    dataset.data = [{
        "qid": 1,
        "vid": "video",
        "duration": 20.0,
        "relevant_windows": [[6.0, 12.0]],
    }]
    dataset.use_video = True
    dataset.use_tef = use_tef
    dataset.load_labels = True
    dataset.max_windows = 5
    dataset.span_loss_type = "l1"
    dataset.clip_len = 2
    dataset.max_v_l = 10
    dataset.data_path = "train.jsonl"
    dataset.temporal_aug_prob = prob
    dataset.temporal_aug_min_keep = 0.5
    dataset.context_extend_prob = 0.0
    dataset.context_extend_max_frac = 1.0
    dataset.temporal_mask_prob = 0.0
    dataset.temporal_mask_n = 0
    dataset.temporal_mask_max_len = 0
    dataset.feat_noise_prob = 0.0
    dataset.feat_noise_std = 0.0
    dataset.multi_moment_prob = 0.0
    dataset._aug_enabled = True
    dataset.intra_video_hard_neg_ratio = 1.0
    dataset.intra_video_hardneg_iou_thd = 0.1
    dataset.emit_hardneg_labels = True
    dataset.vid2windows = {"video": [[[0.0, 2.0]]]}
    dataset._get_query_feat_by_qid = lambda qid: torch.arange(6).reshape(3, 2).float()
    dataset._get_video_feat_by_vid = lambda vid, duration: torch.arange(30).reshape(10, 3).float()
    return dataset


def _make_qv_dataset(prob=0.0, use_tef=False):
    dataset = _make_dataset(prob=prob, use_tef=use_tef)
    dataset.dset_name = "hl"
    dataset.data = [{
        "qid": 1,
        "vid": "video",
        "duration": 20.0,
        "relevant_windows": [[6.0, 12.0]],
        "relevant_clip_ids": [1, 3, 4, 6, 8],
        "saliency_scores": [
            [1, 1, 1],
            [4, 4, 4],
            [3, 3, 3],
            [2, 2, 2],
            [0, 0, 0],
        ],
    }]
    dataset.intra_video_hard_neg_ratio = 0.0
    dataset.emit_hardneg_labels = False
    dataset.vid2windows = {"video": [[[6.0, 12.0]]]}
    return dataset


def _crop(dataset, video_feat=None):
    if video_feat is None:
        video_feat = torch.arange(30).reshape(10, 3).float()
    with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
            patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 8]):
        return dataset._temporal_crop(video_feat, [[6.0, 12.0]], 20.0, 10)


class TestTemporalAug(unittest.TestCase):
    def test_prob_zero_is_byte_identical(self):
        dataset = _make_dataset(prob=0.0)
        baseline = _make_dataset(prob=0.0)
        baseline._temporal_crop = lambda feat, windows, duration, ctx_l: (
            feat, windows, duration, ctx_l, False, 0.0)

        random.seed(7)
        expected = baseline[0]
        random.seed(7)
        actual = dataset[0]

        self.assertEqual(actual["meta"], expected["meta"])
        self.assertEqual(actual["model_inputs"].keys(), expected["model_inputs"].keys())
        for key, expected_value in expected["model_inputs"].items():
            actual_value = actual["model_inputs"][key]
            if torch.is_tensor(expected_value):
                self.assertEqual(actual_value.numpy().tobytes(), expected_value.numpy().tobytes())
            elif hasattr(expected_value, "tobytes"):
                self.assertEqual(actual_value.tobytes(), expected_value.tobytes())
            else:
                self.assertEqual(actual_value, expected_value)

    def test_span_labels_stay_normalized_after_crop(self):
        dataset = _make_dataset()
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 8]):
            output = dataset[0]

        spans = output["model_inputs"]["span_labels"]
        self.assertTrue(torch.all(spans >= 0))
        self.assertTrue(torch.all(spans <= 1))

    def test_gt_moment_clip_set_is_preserved(self):
        dataset = _make_dataset()
        video_feat = torch.arange(10).reshape(10, 1).float()
        feat, windows, duration, ctx_l, cropped, offset_delta = _crop(dataset, video_feat)

        original_clips = set(range(3, 6))
        shifted_clips = set(range(int(windows[0][0] / 2), int(windows[0][1] / 2)))
        mapped_clips = {clip + int(offset_delta / 2) for clip in shifted_clips}
        self.assertTrue(cropped)
        self.assertEqual(offset_delta, 4.0)
        self.assertEqual(mapped_clips, original_clips)
        self.assertEqual(feat.squeeze(1).tolist(), list(range(2, 8)))
        self.assertEqual((duration, ctx_l), (12.0, 6))

    def test_context_length_shrinks_after_crop(self):
        dataset = _make_dataset()
        feat, _, _, ctx_l, cropped, _ = _crop(dataset)

        self.assertTrue(cropped)
        self.assertEqual(ctx_l, len(feat))
        self.assertLess(ctx_l, 10)

    def test_tef_length_matches_video_after_crop(self):
        dataset = _make_dataset(use_tef=True)
        random.seed(0)
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 8]):
            output = dataset[0]

        video_feat = output["model_inputs"]["video_feat"]
        self.assertEqual(len(video_feat), 6)
        self.assertEqual(video_feat.shape, (6, 5))

    def test_context_extend_skips_same_video_sources(self):
        dataset = _make_dataset()
        dataset.context_extend_prob = 1.0
        dataset.max_v_l = 12
        dataset.data.append({
            "qid": 2,
            "vid": "video",
            "duration": 20.0,
            "relevant_windows": [[0.0, 2.0]],
        })

        video_feat = torch.arange(30).reshape(10, 3).float()
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", return_value=2), \
                patch("vmr_detr.data.start_end_dataset.random.choice") as choice_mock:
            feat, windows, duration, ctx_l, extended, offset_delta = dataset._context_extend(
                video_feat, [[6.0, 12.0]], 20.0, 10, 0, "video")

        choice_mock.assert_not_called()
        self.assertFalse(extended)
        self.assertEqual(offset_delta, 0.0)
        self.assertTrue(torch.equal(feat, video_feat))
        self.assertEqual(windows, [[6.0, 12.0]])
        self.assertEqual((duration, ctx_l), (20.0, 10))

    def test_context_extend_uses_different_video_and_reports_offset(self):
        dataset = _make_dataset()
        dataset.context_extend_prob = 1.0
        dataset.max_v_l = 12
        dataset.data.append({
            "qid": 2,
            "vid": "other",
            "duration": 8.0,
            "relevant_windows": [[0.0, 2.0]],
        })

        def get_video_feat(vid, duration):
            if vid == "other":
                return torch.full((4, 3), 100.0)
            return torch.arange(30).reshape(10, 3).float()

        dataset._get_video_feat_by_vid = get_video_feat
        video_feat = torch.arange(30).reshape(10, 3).float()
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 1]), \
                patch("vmr_detr.data.start_end_dataset.random.choice", return_value=1):
            feat, windows, duration, ctx_l, extended, offset_delta = dataset._context_extend(
                video_feat, [[6.0, 12.0]], 20.0, 10, 0, "video")

        self.assertTrue(extended)
        self.assertEqual(offset_delta, -2.0)
        self.assertEqual(windows, [[8.0, 14.0]])
        self.assertEqual((duration, ctx_l), (24.0, 12))
        self.assertTrue(torch.equal(feat[0], torch.full((3,), 100.0)))
        self.assertTrue(torch.equal(feat[1:11], video_feat))

    def test_temporal_mask_leaves_gt_intact_and_masks_context(self):
        dataset = _make_dataset()
        dataset.temporal_mask_prob = 1.0
        dataset.temporal_mask_n = 1
        dataset.temporal_mask_max_len = 2
        video_feat = torch.arange(1, 31).reshape(10, 3).float()
        windows = [[6.0, 12.0]]
        ctx_l = 10

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.choice", return_value=(0, 2)):
            masked = dataset._temporal_mask(video_feat, windows, 20.0, ctx_l)

        self.assertEqual(windows, [[6.0, 12.0]])
        self.assertEqual(ctx_l, 10)
        self.assertTrue(torch.equal(masked[3:6], video_feat[3:6]))
        self.assertTrue(torch.equal(masked[0:2], torch.zeros_like(masked[0:2])))
        non_gt = list(range(0, 3)) + list(range(6, 10))
        zeroed = [idx for idx in non_gt if torch.equal(masked[idx], torch.zeros_like(masked[idx]))]
        self.assertGreaterEqual(len(zeroed), 1)

    def test_feature_noise_preserves_shape_and_respects_probability(self):
        dataset = _make_dataset()
        dataset.feat_noise_prob = 1.0
        dataset.feat_noise_std = 0.02
        video_feat = torch.ones(4, 3)

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.torch.randn_like",
                      return_value=torch.ones_like(video_feat)):
            noisy = dataset._feature_noise(video_feat)

        self.assertEqual(noisy.shape, video_feat.shape)
        self.assertFalse(torch.equal(noisy, video_feat))
        self.assertTrue(torch.allclose(noisy, video_feat + 0.02))

        dataset.feat_noise_prob = 0.0
        self.assertTrue(torch.equal(dataset._feature_noise(video_feat), video_feat))

    def test_temporal_mask_remains_zero_when_feature_noise_also_applies(self):
        dataset = _make_dataset(prob=0.0)
        dataset.temporal_mask_prob = 1.0
        dataset.temporal_mask_n = 1
        dataset.temporal_mask_max_len = 2
        dataset.feat_noise_prob = 1.0
        dataset.feat_noise_std = 0.02

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.choice", return_value=(0, 2)), \
                patch("vmr_detr.data.start_end_dataset.torch.randn_like",
                      side_effect=lambda feat: torch.ones_like(feat)):
            output = dataset[0]

        video_feat = output["model_inputs"]["video_feat"]
        self.assertTrue(torch.equal(video_feat[0:2], torch.zeros_like(video_feat[0:2])))
        self.assertTrue(torch.allclose(
            video_feat[3:6],
            torch.arange(9, 18).reshape(3, 3).float() + 0.02,
        ))

    def test_text_dropout_getter_invokes_row_drop(self):
        dataset = StartEndDataset.__new__(StartEndDataset)
        dataset.dset_name = "charades_sta"
        dataset.q_feat_dirs = []
        dataset.q_feat_type = "last_hidden_state"
        dataset.max_q_l = 32
        dataset.normalize_t = False
        dataset.txt_drop_ratio = 0.5
        dataset._aug_enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.q_feat_dirs = [tmpdir]
            np.savez(
                os.path.join(tmpdir, "qid1.npz"),
                last_hidden_state=np.ones((6, 4), dtype=np.float32),
            )
            with patch("vmr_detr.data.start_end_dataset.np.random.choice",
                       return_value=np.array([0, 2, 5])), \
                    patch.object(dataset, "random_drop_rows",
                                 wraps=dataset.random_drop_rows) as drop_mock:
                q_feat = dataset._get_query_feat_by_qid(1)

        drop_mock.assert_called_once()
        zero_rows = (q_feat.abs().sum(dim=1) == 0).nonzero(as_tuple=False).flatten().tolist()
        self.assertEqual(zero_rows, [0, 2, 5])

    def test_multi_moment_paste_adds_hard_negative_without_new_positive(self):
        dataset = _make_dataset()
        dataset.multi_moment_prob = 1.0
        dataset.max_v_l = 12
        dataset.data.append({
            "qid": 2,
            "vid": "other",
            "duration": 8.0,
            "relevant_windows": [[2.0, 8.0]],
        })
        video_feat = torch.arange(30).reshape(10, 3).float()
        other_feat = torch.arange(100, 124).reshape(8, 3).float()

        def get_video_feat(vid, duration):
            if vid == "other":
                return other_feat
            return video_feat

        dataset._get_video_feat_by_vid = get_video_feat
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.choice", return_value=1), \
                patch("vmr_detr.data.start_end_dataset.random.randint", return_value=1):
            feat, windows, duration, ctx_l, pasted, offset_delta = dataset._multi_moment_paste(
                video_feat, [[6.0, 12.0]], 20.0, 10, 0, "video")

        self.assertTrue(pasted)
        self.assertEqual(ctx_l, 12)
        self.assertLessEqual(ctx_l, dataset.max_v_l)
        self.assertEqual(duration, 24.0)
        self.assertEqual(windows, [[8.0, 14.0]])
        self.assertEqual(len(windows), 1)
        self.assertEqual(offset_delta, -2.0)
        self.assertTrue(torch.equal(feat[0], other_feat[2]))
        self.assertTrue(torch.equal(feat[1:11], video_feat))
        self.assertTrue(torch.equal(feat[11], other_feat[3]))

    def test_get_span_labels_does_not_mutate_input_windows(self):
        dataset = _make_dataset()
        dataset.max_windows = 1
        windows = [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]

        def reverse_in_place(items):
            items.reverse()

        with patch("vmr_detr.data.start_end_dataset.random.shuffle", side_effect=reverse_in_place):
            spans = dataset.get_span_labels(windows, ctx_l=3, duration=6.0)

        self.assertEqual(windows, [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
        self.assertEqual(spans.shape, (1, 2))

    def test_unlabeled_sample_does_not_require_relevant_windows(self):
        dataset = _make_dataset()
        dataset.load_labels = False
        dataset.data = [{
            "qid": 1,
            "vid": "video",
            "duration": 20.0,
        }]

        output = dataset[0]

        self.assertEqual(output["meta"], dataset.data[0])
        self.assertEqual(set(output["model_inputs"].keys()), {"query_feat", "video_feat"})
        self.assertEqual(output["model_inputs"]["video_feat"].shape, (10, 3))

    def test_unlabeled_init_does_not_require_relevant_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.jsonl")
            with open(data_path, "w") as f:
                f.write('{"qid": 1, "vid": "video", "duration": 20.0}\n')

            dataset = StartEndDataset(
                dset_name="charades_sta",
                data_path=data_path,
                v_feat_dirs=[],
                q_feat_dir=[],
                load_labels=False,
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dict(dataset.vid2windows), {})


class TestQVHighlightsAug(unittest.TestCase):
    def test_qv_prob_zero_is_byte_identical(self):
        dataset = _make_qv_dataset(prob=0.0)
        baseline = _make_qv_dataset(prob=0.0)
        baseline._temporal_crop = lambda feat, windows, duration, ctx_l: (
            feat, windows, duration, ctx_l, False, 0.0)

        random.seed(7)
        expected = baseline[0]
        random.seed(7)
        actual = dataset[0]

        self.assertEqual(actual["meta"], expected["meta"])
        self.assertEqual(actual["model_inputs"].keys(), expected["model_inputs"].keys())
        for key, expected_value in expected["model_inputs"].items():
            actual_value = actual["model_inputs"][key]
            if torch.is_tensor(expected_value):
                self.assertEqual(actual_value.numpy().tobytes(), expected_value.numpy().tobytes())
            elif hasattr(expected_value, "tobytes"):
                self.assertEqual(actual_value.tobytes(), expected_value.tobytes())
            else:
                self.assertEqual(actual_value, expected_value)

    def test_qv_crop_remaps_saliency_clip_ids_and_scores(self):
        dataset = _make_qv_dataset(prob=1.0)

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 8]):
            output = dataset[0]

        video_feat = output["model_inputs"]["video_feat"]
        score_array = output["model_inputs"]["saliency_all_labels"]
        spans = output["model_inputs"]["span_labels"]

        self.assertEqual(len(video_feat), 6)
        self.assertTrue(torch.equal(video_feat, torch.arange(6, 24).reshape(6, 3).float()))
        self.assertEqual(len(score_array), 6)
        self.assertEqual(score_array.tolist(), [0.0, 12.0, 9.0, 0.0, 6.0, 0.0])
        self.assertTrue(torch.allclose(spans, torch.tensor([[(2.0 + 8.0) / 24.0, 6.0 / 12.0]])))
        self.assertEqual(dataset.data[0]["relevant_clip_ids"], [1, 3, 4, 6, 8])

    def test_qv_crop_skips_when_no_saliency_labels_remain(self):
        dataset = _make_qv_dataset(prob=1.0)
        dataset.data[0]["relevant_clip_ids"] = [0, 1, 8, 9]
        dataset.data[0]["saliency_scores"] = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ]

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 8]):
            output = dataset[0]

        self.assertEqual(output["model_inputs"]["video_feat"].shape, (10, 3))
        self.assertEqual(len(output["model_inputs"]["saliency_all_labels"]), 10)
        self.assertEqual(
            output["model_inputs"]["saliency_all_labels"].tolist(),
            [3.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 12.0],
        )

    def test_qv_context_extend_shifts_saliency_clip_ids(self):
        dataset = _make_qv_dataset(prob=0.0)
        dataset.context_extend_prob = 1.0
        dataset.max_v_l = 12
        dataset.data.append({
            "qid": 2,
            "vid": "other",
            "duration": 8.0,
            "relevant_windows": [[0.0, 2.0]],
            "relevant_clip_ids": [0],
            "saliency_scores": [[1, 1, 1]],
        })

        def get_video_feat(vid, duration):
            if vid == "other":
                return torch.full((4, 3), 100.0)
            return torch.arange(30).reshape(10, 3).float()

        dataset._get_video_feat_by_vid = get_video_feat
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", side_effect=[2, 1]), \
                patch("vmr_detr.data.start_end_dataset.random.choice", return_value=1):
            output = dataset[0]

        score_array = output["model_inputs"]["saliency_all_labels"]
        self.assertEqual(output["model_inputs"]["video_feat"].shape, (12, 3))
        self.assertEqual(score_array.tolist(), [0.0, 0.0, 3.0, 0.0, 12.0, 9.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0])

    def test_qv_multi_moment_paste_shifts_saliency_clip_ids(self):
        dataset = _make_qv_dataset(prob=0.0)
        dataset.multi_moment_prob = 1.0
        dataset.max_v_l = 12
        dataset.data.append({
            "qid": 2,
            "vid": "other",
            "duration": 8.0,
            "relevant_windows": [[2.0, 8.0]],
            "relevant_clip_ids": [1],
            "saliency_scores": [[1, 1, 1]],
        })
        video_feat = torch.arange(30).reshape(10, 3).float()
        other_feat = torch.arange(100, 124).reshape(8, 3).float()

        def get_video_feat(vid, duration):
            if vid == "other":
                return other_feat
            return video_feat

        dataset._get_video_feat_by_vid = get_video_feat
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.choice", return_value=1), \
                patch("vmr_detr.data.start_end_dataset.random.randint", return_value=1):
            output = dataset[0]

        video_out = output["model_inputs"]["video_feat"]
        score_array = output["model_inputs"]["saliency_all_labels"]
        self.assertEqual(video_out.shape, (12, 3))
        self.assertTrue(torch.equal(video_out[0], other_feat[2]))
        self.assertTrue(torch.equal(video_out[1:11], video_feat))
        self.assertEqual(score_array.tolist(), [0.0, 0.0, 3.0, 0.0, 12.0, 9.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0])

    def test_qv_position_jitter_remaps_saliency_clip_ids(self):
        dataset = _make_qv_dataset(prob=0.0)
        dataset.position_jitter_prob = 1.0
        dataset.position_jitter_context_sec = 0.0

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.randint", return_value=7):
            output = dataset[0]

        video_feat = output["model_inputs"]["video_feat"]
        score_array = output["model_inputs"]["saliency_all_labels"]
        self.assertTrue(torch.equal(video_feat[7:10], torch.arange(9, 18).reshape(3, 3).float()))
        self.assertEqual(score_array.tolist(), [0.0, 3.0, 0.0, 6.0, 0.0, 0.0, 0.0, 12.0, 9.0, 0.0])

    def test_qv_temporal_mask_protects_saliency_labeled_clips(self):
        dataset = _make_qv_dataset(prob=0.0)
        dataset.temporal_mask_prob = 1.0
        dataset.temporal_mask_n = 1
        dataset.temporal_mask_max_len = 2

        def choose_span(valid_spans):
            self.assertNotIn((0, 2), valid_spans)
            self.assertNotIn((6, 7), valid_spans)
            self.assertNotIn((8, 10), valid_spans)
            self.assertIn((7, 8), valid_spans)
            return (7, 8)

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.random.choice", side_effect=choose_span):
            output = dataset[0]

        video_feat = output["model_inputs"]["video_feat"]
        self.assertTrue(torch.equal(video_feat[1], torch.tensor([3.0, 4.0, 5.0])))
        self.assertTrue(torch.equal(video_feat[3:6], torch.arange(9, 18).reshape(3, 3).float()))
        self.assertTrue(torch.equal(video_feat[7], torch.zeros_like(video_feat[7])))

    def test_qv_val_path_rejects_nonzero_augmentation(self):
        with self.assertRaises(AssertionError):
            StartEndDataset(
                dset_name="hl",
                data_path="highlight_val_release.jsonl",
                v_feat_dirs=[],
                q_feat_dir=[],
                temporal_aug_prob=0.1,
            )


class TestAugStopEpoch(unittest.TestCase):
    def _make_aug_dataset(self, aug_stop_epoch):
        dataset = _make_dataset(prob=0.8)
        dataset.context_extend_prob = 0.6
        dataset.temporal_mask_prob = 1.0
        dataset.temporal_mask_n = 1
        dataset.temporal_mask_max_len = 2
        dataset.feat_noise_prob = 1.0
        dataset.feat_noise_std = 0.02
        dataset.multi_moment_prob = 1.0
        dataset.aug_stop_epoch = max(0, int(aug_stop_epoch))  # mirror __init__ clamping
        dataset._base_temporal_aug_prob = dataset.temporal_aug_prob
        dataset._base_context_extend_prob = dataset.context_extend_prob
        return dataset

    def test_set_epoch_disables_aug_after_cutoff(self):
        ds = self._make_aug_dataset(aug_stop_epoch=2)

        # At the cutoff epoch itself (epoch == aug_stop_epoch), probs unchanged
        ds.set_epoch(2)
        self.assertGreater(ds.temporal_aug_prob, 0.0)
        self.assertGreater(ds.context_extend_prob, 0.0)

        # One epoch past the cutoff, both probs zeroed
        ds.set_epoch(3)
        self.assertEqual(ds.temporal_aug_prob, 0.0)
        self.assertEqual(ds.context_extend_prob, 0.0)
        self.assertFalse(ds._aug_enabled)

    def test_set_epoch_zero_means_always_on(self):
        ds = self._make_aug_dataset(aug_stop_epoch=0)
        original_aug_prob = ds.temporal_aug_prob
        original_ctx_prob = ds.context_extend_prob

        ds.set_epoch(999)
        self.assertEqual(ds.temporal_aug_prob, original_aug_prob)
        self.assertEqual(ds.context_extend_prob, original_ctx_prob)
        self.assertTrue(ds._aug_enabled)

    def test_set_epoch_negative_means_always_on(self):
        ds = self._make_aug_dataset(aug_stop_epoch=-1)
        # negative clamped to 0 in __init__ / _make_aug_dataset
        self.assertEqual(ds.aug_stop_epoch, 0)

        ds.set_epoch(999)
        self.assertGreater(ds.temporal_aug_prob, 0.0)
        self.assertGreater(ds.context_extend_prob, 0.0)
        self.assertTrue(ds._aug_enabled)

    def test_set_epoch_rederives_from_epoch(self):
        ds = self._make_aug_dataset(aug_stop_epoch=2)
        original_aug_prob = ds.temporal_aug_prob
        original_ctx_prob = ds.context_extend_prob

        # past the cutoff — probs zeroed
        ds.set_epoch(3)
        self.assertEqual(ds.temporal_aug_prob, 0.0)
        self.assertEqual(ds.context_extend_prob, 0.0)

        # back at/before the cutoff — probs restored (stateless re-derivation)
        ds.set_epoch(2)
        self.assertEqual(ds.temporal_aug_prob, original_aug_prob)
        self.assertEqual(ds.context_extend_prob, original_ctx_prob)
        self.assertTrue(ds._aug_enabled)

    def test_cutoff_makes_new_augmentations_noops(self):
        ds = self._make_aug_dataset(aug_stop_epoch=2)
        ds.data.append({
            "qid": 2,
            "vid": "other",
            "duration": 8.0,
            "relevant_windows": [[2.0, 8.0]],
        })
        ds.max_v_l = 12
        video_feat = torch.arange(30).reshape(10, 3).float()
        other_feat = torch.arange(100, 124).reshape(8, 3).float()

        def get_video_feat(vid, duration):
            if vid == "other":
                return other_feat
            return video_feat

        ds._get_video_feat_by_vid = get_video_feat
        ds.set_epoch(3)
        self.assertFalse(ds._aug_enabled)

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
                patch("vmr_detr.data.start_end_dataset.torch.randn_like",
                      return_value=torch.ones_like(video_feat)):
            self.assertTrue(torch.equal(ds._feature_noise(video_feat), video_feat))
            self.assertTrue(torch.equal(
                ds._temporal_mask(video_feat, [[6.0, 12.0]], 20.0, 10),
                video_feat,
            ))
            feat, windows, duration, ctx_l, pasted, offset_delta = ds._multi_moment_paste(
                video_feat, [[6.0, 12.0]], 20.0, 10, 0, "video")

        self.assertFalse(pasted)
        self.assertEqual(offset_delta, 0.0)
        self.assertEqual(windows, [[6.0, 12.0]])
        self.assertEqual((duration, ctx_l), (20.0, 10))
        self.assertTrue(torch.equal(feat, video_feat))

        text_ds = StartEndDataset.__new__(StartEndDataset)
        text_ds.dset_name = "charades_sta"
        text_ds.q_feat_dirs = []
        text_ds.q_feat_type = "last_hidden_state"
        text_ds.max_q_l = 32
        text_ds.normalize_t = False
        text_ds.txt_drop_ratio = 0.5
        text_ds.temporal_aug_prob = 0.0
        text_ds.context_extend_prob = 0.0
        text_ds._base_temporal_aug_prob = 0.0
        text_ds._base_context_extend_prob = 0.0
        text_ds.aug_stop_epoch = 2
        text_ds.set_epoch(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            text_ds.q_feat_dirs = [tmpdir]
            np.savez(
                os.path.join(tmpdir, "qid1.npz"),
                last_hidden_state=np.ones((6, 4), dtype=np.float32),
            )
            with patch.object(text_ds, "random_drop_rows",
                              side_effect=AssertionError("dropout should be gated off")):
                q_feat = text_ds._get_query_feat_by_qid(1)

        self.assertTrue(torch.equal(q_feat, torch.ones(6, 4)))

    def test_audio_text_dropout_respects_aug_stop_epoch(self):
        dataset = StartEndDataset_audio.__new__(StartEndDataset_audio)
        dataset.dset_name = "charades_sta"
        dataset.q_feat_dir = None
        dataset.q_feat_type = "last_hidden_state"
        dataset.max_q_l = 32
        dataset.normalize_t = False
        dataset.txt_drop_ratio = 0.5
        dataset.aug_stop_epoch = 2
        dataset._aug_enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset.q_feat_dir = tmpdir
            np.savez(
                os.path.join(tmpdir, "qid1.npz"),
                last_hidden_state=np.ones((6, 4), dtype=np.float32),
            )
            with patch("vmr_detr.data.start_end_dataset_audio.np.random.choice",
                       return_value=np.array([0, 2, 5])):
                before_cutoff = dataset._get_query_feat_by_qid(1)

            dataset.set_epoch(3)
            with patch.object(dataset, "random_drop_rows",
                              side_effect=AssertionError("dropout should be gated off")):
                after_cutoff = dataset._get_query_feat_by_qid(1)

        zero_rows = (before_cutoff.abs().sum(dim=1) == 0).nonzero(as_tuple=False).flatten().tolist()
        self.assertEqual(zero_rows, [0, 2, 5])
        self.assertTrue(torch.equal(after_cutoff, torch.ones(6, 4)))


class TestLengthBalance(unittest.TestCase):
    def test_tvsum_returns_no_balance_weights(self):
        dataset = StartEndDataset.__new__(StartEndDataset)
        dataset.dset_name = "tvsum"
        dataset.data = [{"vid": "a"}, {"vid": "b"}]

        self.assertIsNone(dataset.get_balance_weights())

    def test_balance_weights_are_inverse_frequency_by_annotation_width(self):
        dataset = StartEndDataset.__new__(StartEndDataset)
        dataset.dset_name = "charades_sta"
        dataset.length_balance_bins = (0.1, 0.25)
        dataset.data = [
            {"duration": 100.0, "relevant_windows": [[0.0, 5.0]]},
            {"duration": 100.0, "relevant_windows": [[10.0, 15.0]]},
            {"duration": 100.0, "relevant_windows": [[0.0, 50.0]]},
        ]

        weights = dataset.get_balance_weights()

        self.assertEqual(weights, [0.5, 0.5, 1.0])
        self.assertTrue(has_nonuniform_weights(weights))

    def test_uniform_weights_are_not_sampler_worthy(self):
        self.assertFalse(has_nonuniform_weights(None))
        self.assertFalse(has_nonuniform_weights([]))
        self.assertFalse(has_nonuniform_weights([1.0, 1.0, 1.0]))
        self.assertTrue(has_nonuniform_weights([1.0, 1.0, 2.0]))


class TestAugOptions(unittest.TestCase):
    def _valid_opt(self):
        return SimpleNamespace(
            temporal_aug_prob=0.5,
            temporal_aug_min_keep=0.5,
            context_extend_prob=0.5,
            context_extend_max_frac=1.0,
            temporal_mask_prob=0.5,
            temporal_mask_n=1,
            temporal_mask_max_len=2,
            feat_noise_prob=0.5,
            feat_noise_std=0.02,
            multi_moment_prob=0.5,
            aug_stop_epoch=30,
            position_jitter_prob=0.0,
            position_jitter_context_sec=2.0,
            position_jitter_max_shift_frac=0.0,
        )

    def test_validate_aug_options_rejects_invalid_ranges(self):
        invalid_cases = [
            ("temporal_aug_prob", 1.5),
            ("context_extend_prob", -0.1),
            ("temporal_mask_prob", 1.5),
            ("feat_noise_prob", -0.1),
            ("multi_moment_prob", 1.5),
            ("temporal_aug_min_keep", 0.0),
            ("temporal_aug_min_keep", 1.5),
            ("context_extend_max_frac", -0.1),
            ("temporal_mask_n", -1),
            ("temporal_mask_max_len", -1),
            ("feat_noise_std", -0.01),
            ("aug_stop_epoch", -1),
            ("position_jitter_prob", 1.5),
            ("position_jitter_prob", -0.1),
            ("position_jitter_context_sec", -1.0),
            ("position_jitter_max_shift_frac", -0.1),
        ]
        for name, value in invalid_cases:
            opt = self._valid_opt()
            setattr(opt, name, value)
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    BaseOptions._validate_aug_options(opt)

    def test_resume_all_restores_saved_aug_options(self):
        saved_opt = SimpleNamespace(
            temporal_aug_prob=0.8,
            temporal_aug_min_keep=0.4,
            context_extend_prob=0.7,
            context_extend_max_frac=1.5,
            temporal_mask_prob=0.6,
            temporal_mask_n=2,
            temporal_mask_max_len=4,
            feat_noise_prob=0.3,
            feat_noise_std=0.04,
            multi_moment_prob=0.2,
            aug_stop_epoch=12,
            length_balance=True,
            length_balance_bins=[0.05, 0.2],
        )
        opt = SimpleNamespace(
            resume_all=True,
            temporal_aug_prob=0.0,
            temporal_aug_min_keep=0.5,
            context_extend_prob=0.0,
            context_extend_max_frac=1.0,
            temporal_mask_prob=0.0,
            temporal_mask_n=0,
            temporal_mask_max_len=0,
            feat_noise_prob=0.0,
            feat_noise_std=0.0,
            multi_moment_prob=0.0,
            aug_stop_epoch=0,
            length_balance=False,
            length_balance_bins=[0.1, 0.25],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model.ckpt")
            torch.save({"opt": saved_opt}, ckpt_path)
            opt.resume = ckpt_path

            BaseOptions._restore_train_resume_data_options(opt)

        self.assertEqual(opt.temporal_aug_prob, 0.8)
        self.assertEqual(opt.temporal_aug_min_keep, 0.4)
        self.assertEqual(opt.context_extend_prob, 0.7)
        self.assertEqual(opt.context_extend_max_frac, 1.5)
        self.assertEqual(opt.temporal_mask_prob, 0.6)
        self.assertEqual(opt.temporal_mask_n, 2)
        self.assertEqual(opt.temporal_mask_max_len, 4)
        self.assertEqual(opt.feat_noise_prob, 0.3)
        self.assertEqual(opt.feat_noise_std, 0.04)
        self.assertEqual(opt.multi_moment_prob, 0.2)
        self.assertEqual(opt.aug_stop_epoch, 12)
        self.assertTrue(opt.length_balance)
        self.assertEqual(opt.length_balance_bins, [0.05, 0.2])


class TestPositionJitter(unittest.TestCase):
    def _make_jitter_dataset(self, context_sec=0.0, max_shift_frac=0.0):
        dataset = _make_dataset(prob=0.0)
        dataset.position_jitter_prob = 1.0
        dataset.position_jitter_context_sec = context_sec
        dataset.position_jitter_max_shift_frac = max_shift_frac
        return dataset

    def test_relocation_clips_match_original_moment(self):
        """After jitter (no context reserve), clips at the new window equal the original moment clips."""
        ds = self._make_jitter_dataset(context_sec=0.0)
        video_feat = torch.arange(30).reshape(10, 3).float()
        windows = [[6.0, 12.0]]
        # GT clips: floor(6/2)=3, ceil(12/2)=6, m=3
        # r=0, core=[3,6), M=3
        # rest_idx=[0,1,2,6,7,8,9] (7 elements), hi=10-3=7
        # new_core_st=7 → perm=[0,1,2,6,7,8,9,3,4,5]
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
             patch("vmr_detr.data.start_end_dataset.random.randint", return_value=7):
            new_feat, new_windows, perm, jittered = ds._position_jitter(
                video_feat, windows, 20.0, 10)

        self.assertTrue(jittered)
        self.assertEqual(len(new_windows), 1)
        self.assertEqual(len(new_feat), 10)
        # new GT clips 7-9 → seconds [14.0, 20.0]
        self.assertAlmostEqual(new_windows[0][0], 14.0)
        self.assertAlmostEqual(new_windows[0][1], 20.0)
        # clips at new moment position must equal original moment clips
        self.assertTrue(torch.equal(new_feat[7:10], video_feat[3:6]))
        # perm is a valid permutation
        self.assertEqual(sorted(perm), list(range(10)))
        # new_feat == old_feat[perm]
        self.assertTrue(torch.equal(new_feat, video_feat[torch.tensor(perm, dtype=torch.long)]))

    def test_context_reserve_neighbors_travel_with_moment(self):
        """With context_sec=2.0 (r=1), left and right neighbor clips stay adjacent to moment."""
        ds = self._make_jitter_dataset(context_sec=2.0)
        video_feat = torch.arange(30).reshape(10, 3).float()
        windows = [[6.0, 12.0]]
        # r=1, core_st=2, core_ed=7, M=5, core_idx=[2,3,4,5,6]
        # rest_idx=[0,1,7,8,9] (5 elements), hi=10-5=5
        # new_core_st=5 → perm=[0,1,7,8,9,2,3,4,5,6]
        # core placed at positions 5..9: clip2(left)=pos5, moment(3-5)=pos6-8, clip6(right)=pos9
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
             patch("vmr_detr.data.start_end_dataset.random.randint", return_value=5):
            new_feat, new_windows, perm, jittered = ds._position_jitter(
                video_feat, windows, 20.0, 10)

        self.assertTrue(jittered)
        # left neighbor (old clip 2) sits immediately before the moment
        self.assertTrue(torch.equal(new_feat[5], video_feat[2]))
        # moment clips unchanged
        self.assertTrue(torch.equal(new_feat[6:9], video_feat[3:6]))
        # right neighbor (old clip 6) sits immediately after the moment
        self.assertTrue(torch.equal(new_feat[9], video_feat[6]))

    def test_jitter_span_labels_fractional(self):
        ds = self._make_jitter_dataset(context_sec=0.0)
        ds.clip_len = 2
        ds.data[0]["duration"] = 30.0
        ds.data[0]["relevant_windows"] = [[6.5, 11.5]]
        ds.intra_video_hard_neg_ratio = 0.0
        ds.emit_hardneg_labels = False

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
             patch("vmr_detr.data.start_end_dataset.random.randint", return_value=6):
            output = ds[0]

        spans = output["model_inputs"]["span_labels"]
        delta_sec = (6 - 2) * (30.0 / 10)
        expected_st = 6.5 + delta_sec
        expected_ed = 11.5 + delta_sec
        expected = torch.tensor([[
            (expected_st + expected_ed) / (2.0 * 30.0),
            (expected_ed - expected_st) / 30.0,
        ]])
        snapped = torch.tensor([[(18.0 + 24.0) / (2.0 * 30.0), (24.0 - 18.0) / 30.0]])

        self.assertTrue(torch.all(spans >= 0))
        self.assertTrue(torch.all(spans <= 1))
        self.assertTrue(torch.allclose(spans, expected, atol=1e-6))
        self.assertFalse(torch.allclose(spans, snapped, atol=1e-6))

    def test_jitter_window_preserves_subclip_offset(self):
        ds = self._make_jitter_dataset(context_sec=0.0)
        video_feat = torch.arange(30).reshape(10, 3).float()
        windows = [[6.5, 11.5]]

        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0), \
             patch("vmr_detr.data.start_end_dataset.random.randint", return_value=6):
            new_feat, new_windows, perm, jittered = ds._position_jitter(
                video_feat, windows, 20.0, 10)

        self.assertTrue(jittered)
        self.assertAlmostEqual(new_windows[0][0], 12.5)
        self.assertAlmostEqual(new_windows[0][1], 17.5)
        self.assertNotEqual(new_windows, [[12.0, 18.0]])
        self.assertTrue(torch.equal(new_feat[6:9], video_feat[3:6]))
        self.assertEqual(sorted(perm), list(range(10)))

    def test_noop_multiple_windows(self):
        """len(windows) != 1 causes position jitter to skip."""
        ds = self._make_jitter_dataset()
        video_feat = torch.arange(30).reshape(10, 3).float()
        new_feat, new_windows, perm, jittered = ds._position_jitter(
            video_feat, [[0.0, 6.0], [8.0, 12.0]], 20.0, 10)
        self.assertFalse(jittered)
        self.assertIsNone(perm)
        self.assertTrue(torch.equal(new_feat, video_feat))
        self.assertEqual(new_windows, [[0.0, 6.0], [8.0, 12.0]])

    def test_noop_aug_disabled(self):
        """_aug_enabled=False remains the manual global augmentation gate."""
        ds = self._make_jitter_dataset()
        ds._aug_enabled = False
        video_feat = torch.arange(30).reshape(10, 3).float()
        new_feat, new_windows, perm, jittered = ds._position_jitter(
            video_feat, [[6.0, 12.0]], 20.0, 10)
        self.assertFalse(jittered)
        self.assertIsNone(perm)
        self.assertTrue(torch.equal(new_feat, video_feat))

    def test_noop_core_covers_entire_video(self):
        """Huge context_sec expands core to cover all clips — no-op (nowhere to move)."""
        ds = self._make_jitter_dataset(context_sec=100.0)
        video_feat = torch.arange(30).reshape(10, 3).float()
        # r=50 → core_st=0, core_ed=10, M=10 >= ctx_l=10 → early return
        with patch("vmr_detr.data.start_end_dataset.random.random", return_value=0.0):
            new_feat, new_windows, perm, jittered = ds._position_jitter(
                video_feat, [[6.0, 12.0]], 20.0, 10)
        self.assertFalse(jittered)
        self.assertIsNone(perm)
        self.assertTrue(torch.equal(new_feat, video_feat))

    def test_inv_perm_label_remap_logic(self):
        """Directly verify the inv_perm/score-gather label-remapping arithmetic."""
        # perm from test_relocation: [0,1,2,6,7,8,9,3,4,5]
        # old GT at clips 3-5; after jitter, GT at positions 7-9
        perm = [0, 1, 2, 6, 7, 8, 9, 3, 4, 5]
        ctx_l = 10
        inv = [0] * ctx_l
        for _i, _p in enumerate(perm):
            inv[_p] = _i

        # old GT clips (3,4,5) must map to new positions (7,8,9)
        self.assertEqual(inv[3], 7)
        self.assertEqual(inv[4], 8)
        self.assertEqual(inv[5], 9)

        # score_l: 1s at old GT clips 3-5; after gather, 1s must be at new positions 7-9
        score_l = np.zeros(ctx_l)
        score_l[3:6] = 1.0
        new_score = score_l[perm]
        np.testing.assert_array_equal(new_score[7:10], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(new_score[:7], [0.0] * 7)

        # pos indices remapped: [3, 5] → [7, 9]
        pos_l = [3, 5]
        new_pos = [inv[i] for i in pos_l]
        self.assertEqual(new_pos, [7, 9])

        # hard-neg index at old clip 0 stays at new position 0 (perm[0]=0 → inv[0]=0)
        hardneg = [0]
        new_hardneg = [inv[i] for i in hardneg]
        self.assertEqual(new_hardneg, [0])


if __name__ == "__main__":
    unittest.main()
