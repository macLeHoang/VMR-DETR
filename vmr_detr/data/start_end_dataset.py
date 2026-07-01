import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from collections import defaultdict
from math import ceil, floor
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from vmr_detr.ops.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None, v_feat_len_mode="min",
                 intra_video_hard_neg_ratio=0.0, intra_video_hardneg_iou_thd=0.1,
                 emit_hardneg_labels=False,
                 temporal_aug_prob=0.0, temporal_aug_min_keep=0.5,
                 context_extend_prob=0.0, context_extend_max_frac=1.0,
                 aug_stop_epoch=0,
                 length_balance_bins=(0.1, 0.25),
                 temporal_mask_prob=0.0, temporal_mask_n=0,
                 temporal_mask_max_len=0, feat_noise_prob=0.0,
                 feat_noise_std=0.0, multi_moment_prob=0.0,
                 position_jitter_prob=0.0, position_jitter_context_sec=2.0,
                 position_jitter_max_shift_frac=0.0):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dirs = q_feat_dir \
            if isinstance(q_feat_dir, list) else [q_feat_dir]
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.v_feat_len_mode = v_feat_len_mode
        self.intra_video_hard_neg_ratio = intra_video_hard_neg_ratio
        self.intra_video_hardneg_iou_thd = intra_video_hardneg_iou_thd
        self.emit_hardneg_labels = emit_hardneg_labels
        self.temporal_aug_prob = temporal_aug_prob
        self.temporal_aug_min_keep = temporal_aug_min_keep
        self.context_extend_prob = context_extend_prob
        self.context_extend_max_frac = context_extend_max_frac
        self.aug_stop_epoch = max(0, int(aug_stop_epoch))
        self.temporal_mask_prob = temporal_mask_prob
        self.temporal_mask_n = temporal_mask_n
        self.temporal_mask_max_len = temporal_mask_max_len
        self.feat_noise_prob = feat_noise_prob
        self.feat_noise_std = feat_noise_std
        self.multi_moment_prob = multi_moment_prob
        self.position_jitter_prob = position_jitter_prob
        self.position_jitter_context_sec = position_jitter_context_sec
        self.position_jitter_max_shift_frac = position_jitter_max_shift_frac
        self._aug_enabled = True
        self._base_temporal_aug_prob = self.temporal_aug_prob
        self._base_context_extend_prob = self.context_extend_prob
        self.length_balance_bins = length_balance_bins
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0
            assert temporal_aug_prob == 0 and context_extend_prob == 0 \
                and temporal_mask_prob == 0 and feat_noise_prob == 0 \
                and multi_moment_prob == 0 and position_jitter_prob == 0, \
                "augmentation must be disabled for val/test"

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES
        assert self.v_feat_len_mode in ("min", "max", "time_grid")

        # data
        self.data = self.load_data()

        # build per-video window index for intra-video hard negatives
        self.vid2windows = defaultdict(list)
        for d in self.data:
            if "relevant_windows" in d:
                self.vid2windows[d["vid"]].append(d["relevant_windows"])

        # load specific domain data for tvsum dataset
        if self.dset_name == 'tvsum':
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data
        

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            rng = random.Random(42)
            idx = list(range(len(datalist)))
            rng.shuffle(idx)
            datalist = [datalist[i] for i in idx[:n_examples]]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    @staticmethod
    def _seconds_iou(a, b):
        """1D temporal IoU of two [st, ed] second-windows."""
        inter_st = max(a[0], b[0])
        inter_ed = min(a[1], b[1])
        inter = max(0.0, inter_ed - inter_st)
        union = max(0.0, a[1] - a[0]) + max(0.0, b[1] - b[0]) - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _intra_video_hard_pool(self, vid, cur_window, gt_st, gt_ed, ctx_l, clip_len,
                                crop_offset_sec=0.0):
        """Return a list of clip indices covered by OTHER moments of the same video.

        Excludes clips within the current GT range [gt_st, gt_ed] and excludes
        moments whose seconds-IoU with cur_window exceeds self.intra_video_hardneg_iou_thd.
        When crop_offset_sec != 0.0, other-moment windows are mapped into the
        cropped/extended timeline before converting to clip indices.
        """
        hard_clips = set()
        for windows_list in self.vid2windows[vid]:
            for w in windows_list:
                # false-negative guard: skip moments too similar to the current GT
                if self._seconds_iou(w, cur_window) > self.intra_video_hardneg_iou_thd:
                    continue
                w_mapped = [w[0] - crop_offset_sec, w[1] - crop_offset_sec]
                lo = max(0.0, w_mapped[0])
                hi = min(ctx_l * clip_len, w_mapped[1])
                if hi <= lo:
                    continue
                w_st = max(0, int(floor(lo / clip_len)))
                w_ed = min(ctx_l, int(ceil(hi / clip_len)))
                for ci in range(w_st, w_ed):
                    if ci < gt_st or ci > gt_ed:
                        hard_clips.add(ci)
        return list(hard_clips)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)

        # Local copies so augmentation never mutates meta. Unlabeled test
        # splits do not carry relevant_windows and must stay clean.
        windows = [list(w) for w in meta["relevant_windows"]] if self.load_labels else None
        duration = meta.get("duration")
        timeline_offset_sec = 0.0

        _cropped = False   # FIX 1: always defined before conditional block
        _extended = False  # FIX 1: always defined before conditional block
        _composed = False
        _jittered = False
        _jitter_perm = None
        sal_windows = None
        sal_offset = 0.0

        if self.use_video:
            video_feat = self._get_video_feat_by_vid(meta["vid"], meta.get("duration"))  # (Lv, Dv)
            ctx_l = len(video_feat)

            # Apply temporal augmentations before TEF only for label paths
            # that derive saliency supervision from remapped relevant_windows.
            aug_label_safe = (
                self.dset_name in ('charades_sta', 'tacos', 'activitynet', 'nlq')
                or 'subs_train' in str(getattr(self, 'data_path', ''))
            )
            if self.load_labels and windows is not None and self.dset_name != 'tvsum' and aug_label_safe:
                video_feat, windows, duration, ctx_l, _cropped, crop_offset_delta = self._temporal_crop(
                    video_feat, windows, duration, ctx_l)
                timeline_offset_sec += crop_offset_delta
                video_feat, windows, duration, ctx_l, _composed, compose_offset_delta = self._multi_moment_paste(
                    video_feat, windows, duration, ctx_l, index, meta["vid"])
                timeline_offset_sec += compose_offset_delta
                if not _composed:
                    video_feat, windows, duration, ctx_l, _extended, extend_offset_delta = self._context_extend(
                        video_feat, windows, duration, ctx_l, index, meta["vid"])
                    timeline_offset_sec += extend_offset_delta
                sal_windows = [list(w) for w in windows]
                sal_offset = timeline_offset_sec
                if 'subs_train' not in str(getattr(self, 'data_path', '')):
                    video_feat, windows, _jitter_perm, _jittered = self._position_jitter(
                        video_feat, windows, duration, ctx_l)

            video_feat = self._feature_noise(video_feat)
            if self.load_labels and windows is not None and self.dset_name != 'tvsum':
                video_feat = self._temporal_mask(video_feat, windows, duration, ctx_l)

            model_inputs["video_feat"] = video_feat
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            if self.dset_name == 'tvsum':
                model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                meta_label = meta['label']
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
            else:
                if windows is None:
                    windows = [list(w) for w in meta["relevant_windows"]]
                aug_applied = bool(_cropped or _extended or _composed or _jittered)
                model_inputs["span_labels"] = self.get_span_labels(  # (#windows, 2)
                    windows, ctx_l, duration=(duration if aug_applied else None))

                if self.dset_name in ['charades_sta', 'tacos', 'activitynet']:  ## charades, tacos, nlq
                    pos_l, neg_l, score_l, hardneg = self.get_saliency_labels_sub_as_query(
                        (sal_windows[0] if sal_windows is not None else windows[0]), duration, ctx_l,
                        vid=meta["vid"], cur_window=meta["relevant_windows"][0],
                        crop_offset_sec=(sal_offset if sal_windows is not None else timeline_offset_sec))
                    if _jittered and _jitter_perm is not None:
                        inv = [0] * ctx_l
                        for _i, _p in enumerate(_jitter_perm):
                            inv[_p] = _i
                        pos_l = [inv[i] for i in pos_l]
                        neg_l = [inv[i] for i in neg_l]
                        if hardneg is not None:
                            hardneg = [inv[i] for i in hardneg]
                        score_l = score_l[_jitter_perm]
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = pos_l, neg_l, score_l
                    if hardneg is not None:
                        model_inputs["saliency_hardneg_labels"] = hardneg
                elif self.dset_name in ['nlq']:
                    pos_l, neg_l, score_l, hardneg = self.get_saliency_labels_sub_as_query(
                        (sal_windows[0] if sal_windows is not None else windows[0]), duration, ctx_l, 2,
                        vid=meta["vid"], cur_window=meta["relevant_windows"][0],
                        crop_offset_sec=(sal_offset if sal_windows is not None else timeline_offset_sec))
                    if _jittered and _jitter_perm is not None:
                        inv = [0] * ctx_l
                        for _i, _p in enumerate(_jitter_perm):
                            inv[_p] = _i
                        pos_l = [inv[i] for i in pos_l]
                        neg_l = [inv[i] for i in neg_l]
                        if hardneg is not None:
                            hardneg = [inv[i] for i in hardneg]
                        score_l = score_l[_jitter_perm]
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = pos_l, neg_l, score_l
                    if hardneg is not None:
                        model_inputs["saliency_hardneg_labels"] = hardneg
                elif "subs_train" not in self.data_path:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs[
                        "saliency_all_labels"], hardneg = \
                        self.get_saliency_labels_sub_as_query(windows[0], duration, ctx_l,
                                                              vid=meta["vid"], cur_window=meta["relevant_windows"][0],
                                                              crop_offset_sec=timeline_offset_sec)  # only one gt
                    if hardneg is not None:
                        model_inputs["saliency_hardneg_labels"] = hardneg

        return dict(meta=meta, model_inputs=model_inputs)

    # ------------------------------------------------------------------
    # Temporal data-augmentation helpers
    # ------------------------------------------------------------------

    def set_epoch(self, epoch):
        """One-way-style switch re-derived from the epoch each call: disable
        temporal augmentation once we pass aug_stop_epoch (1-indexed,
        inclusive), and keep it at the configured strength otherwise.
        aug_stop_epoch <= 0 means augmentation stays on for the whole run."""
        self._aug_enabled = (self.aug_stop_epoch <= 0) or (epoch <= self.aug_stop_epoch)
        if self.aug_stop_epoch > 0 and epoch > self.aug_stop_epoch:
            self.temporal_aug_prob = 0.0
            self.context_extend_prob = 0.0
        else:
            self.temporal_aug_prob = self._base_temporal_aug_prob
            self.context_extend_prob = self._base_context_extend_prob

    def _feature_noise(self, feat):
        """Add small Gaussian noise to normalized video features."""
        feat_noise_prob = getattr(self, 'feat_noise_prob', 0.0)
        feat_noise_std = getattr(self, 'feat_noise_std', 0.0)
        if (not getattr(self, "_aug_enabled", True)
                or feat_noise_prob <= 0
                or feat_noise_std <= 0
                or random.random() >= feat_noise_prob):
            return feat
        return feat + torch.randn_like(feat) * feat_noise_std

    def _temporal_mask(self, feat, windows, duration, ctx_l):
        """Zero random non-GT clip spans without changing geometry."""
        temporal_mask_prob = getattr(self, 'temporal_mask_prob', 0.0)
        temporal_mask_n = int(getattr(self, 'temporal_mask_n', 0))
        temporal_mask_max_len = int(getattr(self, 'temporal_mask_max_len', 0))
        if (not getattr(self, "_aug_enabled", True)
                or temporal_mask_prob <= 0
                or temporal_mask_n <= 0
                or temporal_mask_max_len <= 0
                or duration is None
                or ctx_l <= 0
                or random.random() >= temporal_mask_prob):
            return feat

        clip_len = duration / ctx_l
        if clip_len <= 0:
            return feat

        gt_st = max(0, int(floor(min(w[0] for w in windows) / clip_len)))
        gt_ed = min(ctx_l, int(ceil(max(w[1] for w in windows) / clip_len)))
        max_len = min(temporal_mask_max_len, ctx_l)
        valid_spans = [
            (start, start + span_len)
            for span_len in range(1, max_len + 1)
            for start in range(0, ctx_l - span_len + 1)
            if start + span_len <= gt_st or start >= gt_ed
        ]
        if not valid_spans:
            return feat

        masked = feat.clone()
        for _ in range(temporal_mask_n):
            start, end = random.choice(valid_spans)
            masked[start:end] = 0
        return masked

    def _temporal_crop(self, video_feat, windows, duration, ctx_l):
        """Randomly crop a temporal subwindow that fully contains the GT moment.

        Returns (feat, windows, duration, ctx_l, cropped_bool, offset_delta_sec).
        All values are unchanged when the augmentation is disabled or skipped.
        """
        if (not self.use_video
                or not getattr(self, "_aug_enabled", True)
                or self.temporal_aug_prob <= 0
                or random.random() >= self.temporal_aug_prob):
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        clip_len = duration / ctx_l

        # GT clip span that MUST be fully retained.
        gt_st = max(0, int(min(w[0] for w in windows) / clip_len))
        gt_ed = min(ctx_l, int(ceil(max(w[1] for w in windows) / clip_len)))

        min_len = max(gt_ed - gt_st, int(round(self.temporal_aug_min_keep * ctx_l)))

        # Sample start (call randint first, then end — test depends on this order).
        start = random.randint(0, gt_st)
        end_lo = max(gt_ed, start + min_len)
        if end_lo > ctx_l:
            # Cannot satisfy constraints; skip augmentation for this sample.
            return (video_feat, windows, duration, ctx_l, False, 0.0)
        end = random.randint(end_lo, ctx_l)

        new_feat = video_feat[start:end]
        offset_sec = start * clip_len
        new_windows = [
            [max(0.0, w[0] - offset_sec), min((end - start) * clip_len, w[1] - offset_sec)]
            for w in windows
        ]
        new_ctx_l = end - start
        new_duration = new_ctx_l * clip_len

        return (new_feat, new_windows, new_duration, new_ctx_l, True, offset_sec)

    def _context_extend(self, video_feat, windows, duration, ctx_l, index, current_vid):
        """Pad context from a random other sample to reduce normalised GT width.

        Returns (feat, windows, duration, ctx_l, extended_bool, offset_delta_sec).
        All values are unchanged when the augmentation is disabled or skipped.
        """
        context_extend_prob = getattr(self, 'context_extend_prob', 0.0)
        if (not self.use_video
                or not getattr(self, "_aug_enabled", True)
                or context_extend_prob <= 0
                or random.random() >= context_extend_prob):
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        clip_len = duration / ctx_l
        max_total = int(getattr(self, 'max_v_l', ctx_l))
        available = max(0, max_total - ctx_l)
        if available <= 0:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        context_extend_max_frac = getattr(self, 'context_extend_max_frac', 1.0)
        max_extra = min(available, int(round(context_extend_max_frac * ctx_l)))
        if max_extra < 1:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        extra = random.randint(1, max_extra)

        # Source extra clips from a different video so pasted context cannot
        # contain same-video annotated positives that labels treat as negatives.
        candidate_indices = [
            i for i, item in enumerate(self.data)
            if i != index and item.get("vid") != current_vid
        ]
        if not candidate_indices:
            return (video_feat, windows, duration, ctx_l, False, 0.0)
        other_index = random.choice(candidate_indices)
        other_meta = self.data[other_index]
        try:
            other_feat = self._get_video_feat_by_vid(other_meta["vid"], other_meta.get("duration"))
        except Exception:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        # Guard feature-dim mismatch.
        if len(other_feat) == 0 or other_feat.shape[-1] != video_feat.shape[-1]:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        # Tile to reach `extra` clips if the other video is too short.
        if len(other_feat) < extra:
            n_repeats = (extra + len(other_feat) - 1) // len(other_feat)
            other_feat = other_feat.repeat(n_repeats, 1)
        other_feat = other_feat[:extra]

        # Random left/right split.
        left_extra = random.randint(0, extra)
        right_extra = extra - left_extra

        parts = []
        if left_extra > 0:
            parts.append(other_feat[:left_extra])
        parts.append(video_feat)
        if right_extra > 0:
            parts.append(other_feat[left_extra:])

        new_feat = torch.cat(parts, dim=0)
        offset = left_extra * clip_len
        new_ctx_l = len(new_feat)
        new_duration = new_ctx_l * clip_len
        new_windows = [
            [max(0.0, w[0] + offset), min(new_duration, w[1] + offset)]
            for w in windows
        ]

        return (new_feat, new_windows, new_duration, new_ctx_l, True, -offset)

    def _multi_moment_paste(self, video_feat, windows, duration, ctx_l, index, current_vid):
        """Paste another video's GT region as hard negative context.

        The pasted moment is intentionally excluded from `windows`; only the
        original moment remains positive after shifting by any left-side paste.
        """
        multi_moment_prob = getattr(self, 'multi_moment_prob', 0.0)
        if (not getattr(self, "_aug_enabled", True)
                or not self.use_video
                or multi_moment_prob <= 0
                or random.random() >= multi_moment_prob):
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        if duration is None or ctx_l <= 0:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        clip_len = duration / ctx_l
        if clip_len <= 0:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        max_total = int(getattr(self, 'max_v_l', ctx_l))
        available = max(0, max_total - ctx_l)
        if available <= 0:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        candidate_indices = [
            i for i, item in enumerate(self.data)
            if i != index and item.get("vid") != current_vid and item.get("relevant_windows")
        ]
        if not candidate_indices:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        other_index = random.choice(candidate_indices)
        other_meta = self.data[other_index]
        try:
            other_feat = self._get_video_feat_by_vid(other_meta["vid"], other_meta.get("duration"))
        except Exception:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        if len(other_feat) == 0 or other_feat.shape[-1] != video_feat.shape[-1]:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        other_duration = other_meta.get("duration")
        if other_duration is None:
            other_clip_len = getattr(self, "clip_len", clip_len)
        else:
            other_clip_len = float(other_duration) / len(other_feat)
        if other_clip_len <= 0:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        other_window = other_meta["relevant_windows"][0]
        src_st = max(0, int(floor(other_window[0] / other_clip_len)))
        src_ed = min(len(other_feat), int(ceil(other_window[1] / other_clip_len)))
        if src_ed <= src_st:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        pasted_feat = other_feat[src_st:src_ed][:available]
        extra = len(pasted_feat)
        if extra <= 0:
            return (video_feat, windows, duration, ctx_l, False, 0.0)

        left_extra = random.randint(0, extra)
        right_extra = extra - left_extra

        parts = []
        if left_extra > 0:
            parts.append(pasted_feat[:left_extra])
        parts.append(video_feat)
        if right_extra > 0:
            parts.append(pasted_feat[left_extra:])

        new_feat = torch.cat(parts, dim=0)
        offset = left_extra * clip_len
        new_ctx_l = len(new_feat)
        new_duration = new_ctx_l * clip_len
        new_windows = [
            [max(0.0, w[0] + offset), min(new_duration, w[1] + offset)]
            for w in windows
        ]

        return (new_feat, new_windows, new_duration, new_ctx_l, True, -offset)

    def _position_jitter(self, video_feat, windows, duration, ctx_l):
        """Relocate the GT moment (+ ±context reserve) to a new position via a pure
        clip permutation. Returns (feat, new_windows, perm, jittered_bool). Length-preserving."""
        prob = getattr(self, 'position_jitter_prob', 0.0)
        if (not self.use_video
                or not getattr(self, "_aug_enabled", True)
                or prob <= 0
                or duration is None or ctx_l <= 0
                or len(windows) != 1
                or random.random() >= prob):
            return video_feat, windows, None, False
        clip_len = duration / ctx_l
        if clip_len <= 0:
            return video_feat, windows, None, False
        gt_st = max(0, int(floor(windows[0][0] / clip_len)))
        gt_ed = min(ctx_l, int(ceil(windows[0][1] / clip_len)))
        m = gt_ed - gt_st
        if m <= 0:
            return video_feat, windows, None, False
        r = int(round(getattr(self, 'position_jitter_context_sec', 2.0) / clip_len))
        core_st = max(0, gt_st - r)
        core_ed = min(ctx_l, gt_ed + r)
        M = core_ed - core_st
        if M >= ctx_l:
            return video_feat, windows, None, False
        rest_idx = list(range(0, core_st)) + list(range(core_ed, ctx_l))
        core_idx = list(range(core_st, core_ed))
        max_shift = getattr(self, 'position_jitter_max_shift_frac', 0.0)
        if max_shift and max_shift > 0:
            lo = max(0, core_st - int(round(max_shift * ctx_l)))
            hi = min(ctx_l - M, core_st + int(round(max_shift * ctx_l)))
        else:
            lo, hi = 0, ctx_l - M
        if hi < lo:
            return video_feat, windows, None, False
        new_core_st = random.randint(lo, hi)
        if new_core_st == core_st:
            return video_feat, windows, None, False
        perm = rest_idx[:new_core_st] + core_idx + rest_idx[new_core_st:]
        new_feat = video_feat[torch.as_tensor(perm, dtype=torch.long)]
        delta_sec = (new_core_st - core_st) * clip_len
        new_windows = [[
            max(0.0, min(float(duration), windows[0][0] + delta_sec)),
            max(0.0, min(float(duration), windows[0][1] + delta_sec)),
        ]]
        return new_feat, new_windows, perm, True

    # ------------------------------------------------------------------
    # Length-balanced sampling weight
    # ------------------------------------------------------------------

    def get_balance_weights(self):
        """Return per-sample inverse-frequency weights based on normalised GT width.

        Bins are defined by self.length_balance_bins edges (default (0.1, 0.25)):
            bin 0 (short):  w_norm < 0.1
            bin 1 (medium): 0.1 <= w_norm < 0.25
            bin 2 (long):   w_norm >= 0.25
        weight_i = total / (num_bins * count_in_bin_of_i).
        Samples whose metadata is missing or has zero duration get weight 1.0.
        Return None for datasets where length balancing should be skipped.
        """
        if self.dset_name == 'tvsum':
            return None

        bin_edges = sorted(getattr(self, 'length_balance_bins', (0.1, 0.25)))
        num_bins = len(bin_edges) + 1

        bin_indices = []
        for meta in self.data:
            try:
                ws = meta["relevant_windows"]
                dur = float(meta["duration"])
                if dur <= 0:
                    bin_indices.append(-1)
                    continue
                w_norm = sum((w[1] - w[0]) for w in ws) / (len(ws) * dur)
                b = 0
                for edge in bin_edges:
                    if w_norm >= edge:
                        b += 1
                bin_indices.append(b)
            except (KeyError, IndexError, ZeroDivisionError, TypeError, ValueError):
                bin_indices.append(-1)

        total = len(self.data)
        bin_counts = [0] * num_bins
        for b in bin_indices:
            if 0 <= b < num_bins:
                bin_counts[b] += 1

        weights = []
        for b in bin_indices:
            if b < 0 or bin_counts[b] == 0:
                weights.append(1.0)
            else:
                weights.append(total / (num_bins * bin_counts[b]))
        return weights


    def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2, vid=None,
                                          cur_window=None, crop_offset_sec=0.0):
        clip_len = duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            if self.dset_name == 'nlq':
                pos_clip_indices = [gt_st] * 2
            else:
                pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))

        # build hard pool only when needed
        hard_pool = []
        need_hard_pool = (vid is not None and
                          (self.intra_video_hard_neg_ratio > 0 or self.emit_hardneg_labels))
        if need_hard_pool:
            _cur_window = cur_window if cur_window is not None else gt_window
            hard_pool = self._intra_video_hard_pool(vid, _cur_window, gt_st, gt_ed, ctx_l, clip_len,
                                                     crop_offset_sec=crop_offset_sec)

        # Level A: mix hard negatives into the existing neg_clip_indices slots.
        # When Level A is inactive, preserve the exact original sampling (without
        # replacement) so default behavior is bit-for-bit unchanged.
        if self.intra_video_hard_neg_ratio > 0 and hard_pool:
            neg_clip_indices = []
            for _ in range(max_n):
                if random.random() < self.intra_video_hard_neg_ratio:
                    neg_clip_indices.append(random.choice(hard_pool))
                elif neg_pool:
                    neg_clip_indices.append(random.choice(neg_pool))
                else:
                    neg_clip_indices.append(pos_clip_indices[0] if pos_clip_indices else 0)
        else:
            try:
                neg_clip_indices = random.sample(neg_pool, k=max_n)
            except:
                neg_clip_indices = pos_clip_indices

        # For charades_sta
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed + 1] = 1

        # Level B: dedicated hard-neg clip labels
        hardneg_clip_indices = None
        if self.emit_hardneg_labels:
            if hard_pool:
                hardneg_clip_indices = [random.choice(hard_pool) for _ in range(max_n)]
            else:
                hardneg_clip_indices = list(neg_clip_indices)

        return pos_clip_indices, neg_clip_indices, score_array, hardneg_clip_indices
        

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # FIX 2: clamp all ids to [0, ctx_l-1]; score_array stays exactly length ctx_l
        rel = [min(max(int(c), 0), ctx_l - 1) for c in rel_clip_ids]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel)):
            score_array[rel[idx]] = agg_scores[idx]

        # indices in the whole video
        hard_pos_clip_indices = [rel[idx] for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [rel[idx] for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l, duration=None):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        duration: when provided (after temporal augmentation), use it as the normalisation denominator
            so that span values remain correct after crop/extend.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        windows = [list(w) for w in windows]
        if len(windows) > self.max_windows:
            selected_windows = list(windows)
            random.shuffle(selected_windows)
            windows = selected_windows[:self.max_windows]
        if self.span_loss_type in ("l1", "dfl", "fdr"):
            denom = duration if duration is not None else (ctx_l * self.clip_len)
            windows = torch.Tensor(windows) / denom  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            eff = (duration / ctx_l) if duration is not None else self.clip_len
            windows = torch.Tensor([
                [int(w[0] / eff), min(int(w[1] / eff), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows
    
    def _get_query_feat_by_qid(self, qid, aug_id=0):
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_list = []
        q_feat_type = self.q_feat_type
        for _feat_dir in self.q_feat_dirs:
            if self.dset_name == "tvsum":
                q_feat_path = join(_feat_dir, f"{qid}{aug}.npz")
                q_feat_type = 'token'
            elif self.dset_name in ['youtube_uni', 'nlq', 'tacos']:
                q_feat_path = join(_feat_dir, f"{qid}{aug}.npz")
            else:
                q_feat_path = join(_feat_dir, f"qid{qid}{aug}.npz")

            q_feat = np.load(q_feat_path)[q_feat_type].astype(np.float32)

            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]

            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            q_feat_list.append(q_feat)
        # some features are slightly longer than the others
        min_len = min([len(e) for e in q_feat_list])
        q_feat_list = [e[:min_len] for e in q_feat_list]
        q_feat = np.concatenate(q_feat_list, axis=1)
        if self.txt_drop_ratio > 0 and getattr(self, "_aug_enabled", True):
            q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (Lv, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _aligned_video_length_from_lengths(self, lengths):
        if not lengths:
            return 0
        if self.v_feat_len_mode == "max":
            return max(lengths)
        return min(lengths)

    def _canonical_video_length(self, duration, fallback_lengths=None):
        if duration is not None and self.clip_len > 0:
            length = int(round(max(float(duration), 0.0) / self.clip_len))
            return max(min(length, int(self.max_v_l)), 1)
        if fallback_lengths:
            return max(min(max(fallback_lengths), int(self.max_v_l)), 1)
        return 1

    @staticmethod
    def _uniform_time_centers(length, duration):
        if length <= 0:
            return np.zeros((0,), dtype=np.float32)
        duration = max(float(duration), 1e-6)
        return ((np.arange(length, dtype=np.float32) + 0.5) / float(length)) * duration

    def _get_stream_time_centers(self, feat_npz, feat_len, duration):
        if feat_len <= 0:
            return np.zeros((0,), dtype=np.float32)
        duration = max(float(duration), 1e-6)
        if "timestamps" in feat_npz:
            ts = np.asarray(feat_npz["timestamps"], dtype=np.float32)
            if ts.ndim == 2 and ts.shape[1] >= 2:
                centers = 0.5 * (ts[:, 0] + ts[:, 1])
            else:
                centers = ts.reshape(-1)
            if len(centers) == feat_len:
                return np.clip(centers, 0.0, duration)
        return self._uniform_time_centers(feat_len, duration)

    @staticmethod
    def _resample_by_index(feat, target_len):
        if len(feat) == 0:
            return np.zeros((target_len, feat.shape[1]), dtype=np.float32)
        if len(feat) == target_len:
            return feat
        if len(feat) == 1:
            return np.repeat(feat, target_len, axis=0)
        src_x = np.linspace(0.0, 1.0, num=len(feat), dtype=np.float32)
        tgt_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
        out = np.empty((target_len, feat.shape[1]), dtype=np.float32)
        for d in range(feat.shape[1]):
            out[:, d] = np.interp(tgt_x, src_x, feat[:, d], left=feat[0, d], right=feat[-1, d])
        return out

    @staticmethod
    def _resample_by_time(feat, src_t, target_t):
        if len(feat) == 0:
            return np.zeros((len(target_t), feat.shape[1]), dtype=np.float32)
        if len(feat) == len(target_t) and np.allclose(src_t, target_t):
            return feat
        if len(feat) == 1:
            return np.repeat(feat, len(target_t), axis=0)
        src_t = np.asarray(src_t, dtype=np.float32)
        target_t = np.asarray(target_t, dtype=np.float32)
        src_t = np.maximum.accumulate(src_t)
        src_t = src_t + np.linspace(0.0, 1e-6, num=len(src_t), dtype=np.float32)
        out = np.empty((len(target_t), feat.shape[1]), dtype=np.float32)
        for d in range(feat.shape[1]):
            out[:, d] = np.interp(target_t, src_t, feat[:, d], left=feat[0, d], right=feat[-1, d])
        return out

    def _get_video_feat_by_vid(self, vid, duration=None):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        else:
            v_feat_list = []
            raw_lengths = []
            time_centers = []
            duration_val = float(duration) if duration is not None else float(self.max_v_l * self.clip_len)
            duration_val = max(duration_val, 1e-6)
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                feat_npz = np.load(_feat_path)
                _feat = feat_npz["features"].astype(np.float32)
                raw_lengths.append(len(_feat))
                v_feat_list.append(_feat)
                time_centers.append(self._get_stream_time_centers(feat_npz, len(_feat), duration_val))

            if self.v_feat_len_mode == "time_grid":
                target_len = self._canonical_video_length(duration, fallback_lengths=raw_lengths)
                target_t = self._uniform_time_centers(target_len, duration_val)
                v_feat_list = [
                    self._resample_by_time(src_feat, src_t, target_t)
                    for src_feat, src_t in zip(v_feat_list, time_centers)
                ]
            else:
                target_len = self._aligned_video_length_from_lengths(raw_lengths)
                target_len = max(min(int(target_len), int(self.max_v_l)), 1)
                if self.v_feat_len_mode == "max":
                    v_feat_list = [self._resample_by_index(e, target_len) for e in v_feat_list]
                else:
                    v_feat_list = [e[:target_len] for e in v_feat_list]

            if self.normalize_v:
                v_feat_list = [l2_normalize_np_array(e) for e in v_feat_list]

            v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)



def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels", "saliency_hardneg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            # print(pad_data, mask_data)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_hardneg_labels" in batched_model_inputs:
        targets["saliency_hardneg_labels"] = batched_model_inputs["saliency_hardneg_labels"].to(
            device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
