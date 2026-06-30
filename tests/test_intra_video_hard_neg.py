import unittest
import random
import sys
import types

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pandas"] = types.ModuleType("pandas")

from vmr_detr.data.start_end_dataset import StartEndDataset


def _make_dataset(vid2windows, iou_thd=0.1, hard_neg_ratio=0.0, emit_hardneg=False):
    """Instantiate StartEndDataset without loading any features or data files."""
    obj = object.__new__(StartEndDataset)
    obj.vid2windows = vid2windows
    obj.intra_video_hardneg_iou_thd = iou_thd
    obj.intra_video_hard_neg_ratio = hard_neg_ratio
    obj.emit_hardneg_labels = emit_hardneg
    obj.dset_name = "charades_sta"
    return obj


class TestSecondsIou(unittest.TestCase):
    def test_identical_windows(self):
        iou = StartEndDataset._seconds_iou([0.0, 10.0], [0.0, 10.0])
        self.assertAlmostEqual(iou, 1.0)

    def test_disjoint_windows(self):
        iou = StartEndDataset._seconds_iou([0.0, 5.0], [5.0, 10.0])
        self.assertAlmostEqual(iou, 0.0)

    def test_partial_overlap(self):
        # [0, 4] and [2, 6]: intersection=2, union=6
        iou = StartEndDataset._seconds_iou([0.0, 4.0], [2.0, 6.0])
        self.assertAlmostEqual(iou, 2.0 / 6.0)

    def test_zero_length_window(self):
        iou = StartEndDataset._seconds_iou([3.0, 3.0], [0.0, 10.0])
        self.assertAlmostEqual(iou, 0.0)

    def test_contained_window(self):
        # [2, 4] inside [0, 10]: intersection=2, union=10
        iou = StartEndDataset._seconds_iou([2.0, 4.0], [0.0, 10.0])
        self.assertAlmostEqual(iou, 2.0 / 10.0)


class TestIntraVideoHardPool(unittest.TestCase):
    def _make_multi_moment_dataset(self):
        """Video with two moments: [0,4] and [6,10], ctx_l=10, clip_len=1.0."""
        vid2windows = {
            "vid_A": [
                [[0.0, 4.0]],  # moment 1: clips 0-4
                [[6.0, 10.0]],  # moment 2: clips 6-10
            ]
        }
        return _make_dataset(vid2windows, iou_thd=0.1)

    def test_hard_pool_clips_inside_other_moments(self):
        ds = self._make_multi_moment_dataset()
        # Current GT: clips 0-3 (seconds [0,4]); hard pool should include clips 6-9
        hard_pool = ds._intra_video_hard_pool(
            vid="vid_A",
            cur_window=[0.0, 4.0],
            gt_st=0,
            gt_ed=3,
            ctx_l=10,
            clip_len=1.0,
        )
        self.assertTrue(len(hard_pool) > 0, "Hard pool should be non-empty")
        # All returned clips must be outside the current GT [0, 3]
        for ci in hard_pool:
            self.assertNotIn(ci, range(0, 4), f"Clip {ci} is within current GT")
        # Clips 6-10 (min(ctx_l, int(10/1)+1)=10) should be present
        for ci in range(6, 10):
            self.assertIn(ci, hard_pool, f"Clip {ci} from other moment should be in pool")

    def test_exact_boundary_window_does_not_include_next_clip(self):
        ds = self._make_multi_moment_dataset()
        hard_pool = ds._intra_video_hard_pool(
            vid="vid_A",
            cur_window=[0.0, 4.0],
            gt_st=0,
            gt_ed=3,
            ctx_l=12,
            clip_len=1.0,
        )
        self.assertIn(9, hard_pool)
        self.assertNotIn(10, hard_pool)

    def test_hard_pool_excludes_current_gt_clips(self):
        ds = self._make_multi_moment_dataset()
        # Current GT: clips 6-9 (seconds [6,10])
        hard_pool = ds._intra_video_hard_pool(
            vid="vid_A",
            cur_window=[6.0, 10.0],
            gt_st=6,
            gt_ed=9,
            ctx_l=10,
            clip_len=1.0,
        )
        for ci in hard_pool:
            self.assertNotIn(ci, range(6, 10), f"Clip {ci} is within current GT")

    def test_hard_pool_iou_guard_excludes_similar_moments(self):
        # Two moments with high overlap — the second should be filtered out
        vid2windows = {
            "vid_B": [
                [[0.0, 5.0]],  # moment 1: clips 0-4
                [[1.0, 6.0]],  # moment 2: IoU with moment 1 is high
            ]
        }
        ds = _make_dataset(vid2windows, iou_thd=0.5)
        # IoU([0,5],[1,6]) = intersection 4 / union 6 ≈ 0.667 > 0.5 → should be excluded
        hard_pool = ds._intra_video_hard_pool(
            vid="vid_B",
            cur_window=[0.0, 5.0],
            gt_st=0,
            gt_ed=4,
            ctx_l=10,
            clip_len=1.0,
        )
        # The second moment overlaps enough to be filtered by iou_thd=0.5
        self.assertEqual(hard_pool, [], "High-IoU moment should be filtered out")

    def test_single_moment_video_returns_empty(self):
        vid2windows = {
            "vid_C": [
                [[2.0, 8.0]],  # only one moment
            ]
        }
        ds = _make_dataset(vid2windows, iou_thd=0.1)
        hard_pool = ds._intra_video_hard_pool(
            vid="vid_C",
            cur_window=[2.0, 8.0],
            gt_st=2,
            gt_ed=7,
            ctx_l=10,
            clip_len=1.0,
        )
        self.assertEqual(hard_pool, [], "Single-moment video should yield empty hard pool")


class TestGetSaliencyLabelsBackwardCompat(unittest.TestCase):
    """With ratio=0, emit=False, function must return 4-tuple with hardneg=None
    and neg_clip_indices must all be outside GT."""

    def test_default_off_returns_none_hardneg(self):
        vid2windows = {
            "vid_D": [
                [[0.0, 5.0]],
                [[6.0, 10.0]],
            ]
        }
        ds = _make_dataset(vid2windows, iou_thd=0.1, hard_neg_ratio=0.0, emit_hardneg=False)
        pos, neg, score_array, hardneg = ds.get_saliency_labels_sub_as_query(
            gt_window=[0.0, 5.0],
            duration=10.0,
            ctx_l=10,
            vid="vid_D",
            cur_window=[0.0, 5.0],
        )
        self.assertIsNone(hardneg, "hardneg must be None when emit_hardneg_labels=False")
        # All neg clips should be outside GT (gt_st=0, gt_ed=4 for [0,5] / clip_len=1)
        for ci in neg:
            self.assertNotIn(ci, range(0, 5), f"Neg clip {ci} is within GT")

    def test_no_vid_arg_returns_none_hardneg(self):
        vid2windows = {}
        ds = _make_dataset(vid2windows, iou_thd=0.1, hard_neg_ratio=0.0, emit_hardneg=False)
        pos, neg, score_array, hardneg = ds.get_saliency_labels_sub_as_query(
            gt_window=[2.0, 6.0],
            duration=10.0,
            ctx_l=10,
        )
        self.assertIsNone(hardneg)


class TestLevelBEmitHardneg(unittest.TestCase):
    def test_emit_hardneg_labels_non_empty_pool(self):
        vid2windows = {
            "vid_E": [
                [[0.0, 4.0]],
                [[6.0, 10.0]],
            ]
        }
        ds = _make_dataset(vid2windows, iou_thd=0.1, hard_neg_ratio=0.0, emit_hardneg=True)
        pos, neg, score_array, hardneg = ds.get_saliency_labels_sub_as_query(
            gt_window=[0.0, 4.0],
            duration=10.0,
            ctx_l=10,
            vid="vid_E",
            cur_window=[0.0, 4.0],
        )
        self.assertIsNotNone(hardneg, "hardneg must be set when emit_hardneg_labels=True")
        self.assertEqual(len(hardneg), 2)
        # Each hardneg clip should come from the other moment (clips 6-9)
        for ci in hardneg:
            self.assertIn(ci, range(6, 10), f"Hardneg clip {ci} not from other moment")

    def test_emit_hardneg_labels_empty_pool_fallback(self):
        """When hard pool is empty, hardneg should fall back to neg clips."""
        vid2windows = {
            "vid_F": [
                [[0.0, 10.0]],  # single moment covers everything
            ]
        }
        ds = _make_dataset(vid2windows, iou_thd=0.1, hard_neg_ratio=0.0, emit_hardneg=True)
        pos, neg, score_array, hardneg = ds.get_saliency_labels_sub_as_query(
            gt_window=[0.0, 10.0],
            duration=10.0,
            ctx_l=10,
            vid="vid_F",
            cur_window=[0.0, 10.0],
        )
        self.assertIsNotNone(hardneg)
        self.assertEqual(len(hardneg), 2)


if __name__ == "__main__":
    unittest.main()
