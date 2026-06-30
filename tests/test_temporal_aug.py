"""CPU tests for train-only temporal-crop data augmentation."""

import os
import random
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

from vmr_detr.data.start_end_dataset import StartEndDataset
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
    dataset.intra_video_hard_neg_ratio = 1.0
    dataset.intra_video_hardneg_iou_thd = 0.1
    dataset.emit_hardneg_labels = True
    dataset.vid2windows = {"video": [[[0.0, 2.0]]]}
    dataset._get_query_feat_by_qid = lambda qid: torch.arange(6).reshape(3, 2).float()
    dataset._get_video_feat_by_vid = lambda vid, duration: torch.arange(30).reshape(10, 3).float()
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


class TestAugStopEpoch(unittest.TestCase):
    def _make_aug_dataset(self, aug_stop_epoch):
        dataset = _make_dataset(prob=0.8)
        dataset.context_extend_prob = 0.6
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

    def test_set_epoch_zero_means_always_on(self):
        ds = self._make_aug_dataset(aug_stop_epoch=0)
        original_aug_prob = ds.temporal_aug_prob
        original_ctx_prob = ds.context_extend_prob

        ds.set_epoch(999)
        self.assertEqual(ds.temporal_aug_prob, original_aug_prob)
        self.assertEqual(ds.context_extend_prob, original_ctx_prob)

    def test_set_epoch_negative_means_always_on(self):
        ds = self._make_aug_dataset(aug_stop_epoch=-1)
        # negative clamped to 0 in __init__ / _make_aug_dataset
        self.assertEqual(ds.aug_stop_epoch, 0)

        ds.set_epoch(999)
        self.assertGreater(ds.temporal_aug_prob, 0.0)
        self.assertGreater(ds.context_extend_prob, 0.0)

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
            aug_stop_epoch=30,
        )

    def test_validate_aug_options_rejects_invalid_ranges(self):
        invalid_cases = [
            ("temporal_aug_prob", 1.5),
            ("context_extend_prob", -0.1),
            ("temporal_aug_min_keep", 0.0),
            ("temporal_aug_min_keep", 1.5),
            ("context_extend_max_frac", -0.1),
            ("aug_stop_epoch", -1),
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
        self.assertEqual(opt.aug_stop_epoch, 12)
        self.assertTrue(opt.length_balance)
        self.assertEqual(opt.length_balance_bins, [0.05, 0.2])


if __name__ == "__main__":
    unittest.main()
