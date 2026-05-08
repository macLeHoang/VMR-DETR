#!/usr/bin/env python3
"""Analyze R1@1 versus oracle R1@K for moment retrieval predictions."""

import argparse
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path


DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)
DEFAULT_KS = (1, 5, 10)
SPLIT_RANGES = OrderedDict(
    [
        ("full", (0.0, 150.0)),
        ("short", (0.0, 10.0)),
        ("middle", (10.0, 30.0)),
        ("long", (30.0, 150.0)),
    ]
)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def temporal_iou(window_a, window_b):
    a_start, a_end = float(window_a[0]), float(window_a[1])
    b_start, b_end = float(window_b[0]), float(window_b[1])
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def window_length(window):
    return float(window[1]) - float(window[0])


def windows_for_split(gt_row, split_name):
    windows = gt_row.get("relevant_windows", [])
    if split_name == "full":
        return windows

    min_len, max_len = SPLIT_RANGES[split_name]
    return [w for w in windows if min_len < window_length(w) <= max_len]


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    low = math.floor(k)
    high = math.ceil(k)
    if low == high:
        return values[low]
    return values[low] * (high - k) + values[high] * (k - low)


def summarize_values(values):
    if not values:
        return {}
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 50),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
    }


def best_iou_at_k(pred_windows, gt_windows, k):
    best_iou = 0.0
    best_rank = None
    best_pred = None
    best_gt = None
    for rank, pred in enumerate(pred_windows[:k], start=1):
        for gt in gt_windows:
            iou = temporal_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_rank = rank
                best_pred = pred
                best_gt = gt
    return best_iou, best_rank, best_pred, best_gt


def first_correct_rank(pred_windows, gt_windows, threshold, max_k):
    for rank, pred in enumerate(pred_windows[:max_k], start=1):
        if max((temporal_iou(pred, gt) for gt in gt_windows), default=0.0) >= threshold:
            return rank
    return None


def build_items(pred_rows, gt_rows):
    pred_by_qid = {row["qid"]: row for row in pred_rows}
    gt_by_qid = {row["qid"]: row for row in gt_rows}
    qids = sorted(set(pred_by_qid).intersection(gt_by_qid))
    items = []

    for qid in qids:
        pred_row = pred_by_qid[qid]
        pred_windows = sorted(
            pred_row.get("pred_relevant_windows", []),
            key=lambda window: float(window[2]) if len(window) > 2 else 0.0,
            reverse=True,
        )
        items.append(
            {
                "qid": qid,
                "pred": pred_row,
                "gt": gt_by_qid[qid],
                "pred_windows": pred_windows,
            }
        )

    return items, pred_by_qid, gt_by_qid


def analyze_split(items, split_name, thresholds, ks, include_candidate_diagnostics=False):
    split_rows = []
    max_k = max(ks)
    for item in items:
        gt_windows = windows_for_split(item["gt"], split_name)
        if not gt_windows:
            continue
        row = {"item": item, "gt_windows": gt_windows}
        for k in ks:
            row[f"iou@{k}"] = best_iou_at_k(item["pred_windows"], gt_windows, k)[0]
        split_rows.append(row)

    result = {
        "n": len(split_rows),
        "r1": OrderedDict(),
        "miou": OrderedDict(),
        "first_correct_rank": OrderedDict(),
        "length_diagnostics": {},
    }
    if not split_rows:
        return result

    for threshold in thresholds:
        threshold_key = f"{threshold:.2f}"
        result["r1"][threshold_key] = OrderedDict()
        for k in ks:
            hits = sum(row[f"iou@{k}"] >= threshold for row in split_rows)
            result["r1"][threshold_key][f"oracle@{k}"] = 100.0 * hits / len(split_rows)

        rank_counter = Counter()
        for row in split_rows:
            rank = first_correct_rank(
                row["item"]["pred_windows"], row["gt_windows"], threshold, max_k
            )
            if rank is None:
                rank_counter[f"none_top{max_k}"] += 1
            elif rank == 1:
                rank_counter["rank1"] += 1
            elif rank <= 5:
                rank_counter["rank2-5"] += 1
            else:
                rank_counter[f"rank6-{max_k}"] += 1
        result["first_correct_rank"][threshold_key] = {
            name: {
                "count": rank_counter.get(name, 0),
                "pct": 100.0 * rank_counter.get(name, 0) / len(split_rows),
            }
            for name in ["rank1", "rank2-5", f"rank6-{max_k}", f"none_top{max_k}"]
        }

    for k in ks:
        ious = [row[f"iou@{k}"] for row in split_rows]
        result["miou"][f"oracle@{k}"] = {
            "mean": 100.0 * sum(ious) / len(ious),
            "median": 100.0 * percentile(ious, 50),
        }

    length_diagnostics = compute_length_diagnostics(split_rows)
    result["length_diagnostics"] = length_diagnostics
    if include_candidate_diagnostics:
        result["candidate_diagnostics"] = compute_candidate_diagnostics(split_rows, max(ks))
    return result


def score_of_window(window):
    return float(window[2]) if len(window) > 2 else 0.0


def rank_counter_summary(ranks, max_k):
    counter = Counter()
    for rank in ranks:
        if rank is None:
            counter[f"none_top{max_k}"] += 1
        elif rank == 1:
            counter["rank1"] += 1
        elif rank <= 5:
            counter["rank2-5"] += 1
        else:
            counter[f"rank6-{max_k}"] += 1
    total = len(ranks)
    return {
        name: {
            "count": counter.get(name, 0),
            "pct": 100.0 * counter.get(name, 0) / total if total else 0.0,
        }
        for name in ["rank1", "rank2-5", f"rank6-{max_k}", f"none_top{max_k}"]
    }


def compute_candidate_diagnostics(split_rows, max_k):
    top1_ious = []
    best_ious = []
    iou_gains = []
    best_ranks = []
    score_gaps = []
    top1_lengths = []
    best_lengths = []
    top1_gt_ratios = []
    best_gt_ratios = []

    for row in split_rows:
        pred_windows = row["item"]["pred_windows"]
        if not pred_windows:
            continue
        top1 = pred_windows[0]
        top1_iou, _, _, top1_gt = best_iou_at_k([top1], row["gt_windows"], 1)
        best_iou, best_rank, best_pred, best_gt = best_iou_at_k(
            pred_windows, row["gt_windows"], max_k
        )
        if best_pred is None or best_gt is None:
            continue

        top1_len = max(0.0, window_length(top1))
        best_len = max(0.0, window_length(best_pred))
        top1_gt_len = max(1e-9, window_length(top1_gt if top1_gt is not None else best_gt))
        best_gt_len = max(1e-9, window_length(best_gt))

        top1_ious.append(top1_iou)
        best_ious.append(best_iou)
        iou_gains.append(best_iou - top1_iou)
        best_ranks.append(best_rank)
        score_gaps.append(score_of_window(top1) - score_of_window(best_pred))
        top1_lengths.append(top1_len)
        best_lengths.append(best_len)
        top1_gt_ratios.append(top1_len / top1_gt_len)
        best_gt_ratios.append(best_len / best_gt_len)

    return {
        "best_iou_rank": rank_counter_summary(best_ranks, max_k),
        "top1_iou": summarize_values(top1_ious),
        f"best_iou_top{max_k}": summarize_values(best_ious),
        f"iou_gain_top{max_k}_minus_top1": summarize_values(iou_gains),
        "top1_score_minus_best_iou_score": summarize_values(score_gaps),
        "top1_length": summarize_values(top1_lengths),
        "best_iou_length": summarize_values(best_lengths),
        "top1_length_ratio_to_gt": summarize_values(top1_gt_ratios),
        "best_iou_length_ratio_to_gt": summarize_values(best_gt_ratios),
    }


def compute_length_diagnostics(split_rows):
    gt_lengths = []
    top1_lengths = []
    length_ratios = []
    length_diffs = []
    center_errors = []
    starts_at_zero = 0
    too_long = 0
    too_short = 0

    for row in split_rows:
        pred_windows = row["item"]["pred_windows"]
        if not pred_windows:
            continue
        top1 = pred_windows[0]
        gt = max(row["gt_windows"], key=lambda window: temporal_iou(top1, window))
        pred_len = max(0.0, window_length(top1))
        gt_len = max(1e-9, window_length(gt))
        pred_center = (float(top1[0]) + float(top1[1])) / 2.0
        gt_center = (float(gt[0]) + float(gt[1])) / 2.0

        gt_lengths.append(gt_len)
        top1_lengths.append(pred_len)
        length_ratios.append(pred_len / gt_len)
        length_diffs.append(pred_len - gt_len)
        center_errors.append(pred_center - gt_center)
        starts_at_zero += int(abs(float(top1[0])) < 1e-9)
        too_long += int(pred_len > 1.5 * gt_len)
        too_short += int(pred_len < 0.67 * gt_len)

    n = len(top1_lengths)
    if n == 0:
        return {}
    abs_center_errors = [abs(value) for value in center_errors]
    return {
        "gt_length": summarize_values(gt_lengths),
        "top1_length": summarize_values(top1_lengths),
        "top1_minus_gt_length": summarize_values(length_diffs),
        "top1_length_ratio": summarize_values(length_ratios),
        "center_error": summarize_values(center_errors),
        "abs_center_error": summarize_values(abs_center_errors),
        "top1_starts_at_zero_pct": 100.0 * starts_at_zero / n,
        "top1_too_long_gt_1_5x_pct": 100.0 * too_long / n,
        "top1_too_short_lt_0_67x_pct": 100.0 * too_short / n,
    }


def analyze_rows(
    pred_rows,
    gt_rows,
    thresholds,
    ks,
    pred_path="<memory>",
    gt_path="<memory>",
    include_candidate_diagnostics=False,
):
    items, pred_by_qid, gt_by_qid = build_items(pred_rows, gt_rows)
    if not items:
        raise ValueError(
            "No overlapping qids between prediction and ground-truth files: "
            f"{pred_path} vs {gt_path}"
        )

    result = {
        "metadata": {
            "pred_path": str(pred_path),
            "gt_path": str(gt_path),
            "prediction_count": len(pred_rows),
            "ground_truth_count": len(gt_rows),
            "matched_count": len(items),
            "unmatched_prediction_count": len(set(pred_by_qid) - set(gt_by_qid)),
            "unmatched_ground_truth_count": len(set(gt_by_qid) - set(pred_by_qid)),
            "thresholds": list(thresholds),
            "ks": list(ks),
        },
        "splits": OrderedDict(),
    }

    for split_name in SPLIT_RANGES:
        result["splits"][split_name] = analyze_split(
            items,
            split_name,
            thresholds,
            ks,
            include_candidate_diagnostics=include_candidate_diagnostics,
        )
        result["splits"][split_name]["pct"] = (
            100.0 * result["splits"][split_name]["n"] / len(items)
        )
    return result


def analyze(pred_path, gt_path, thresholds, ks, include_candidate_diagnostics=False):
    pred_rows = load_jsonl(pred_path)
    gt_rows = load_jsonl(gt_path)
    return analyze_rows(
        pred_rows,
        gt_rows,
        thresholds,
        ks,
        pred_path=pred_path,
        gt_path=gt_path,
        include_candidate_diagnostics=include_candidate_diagnostics,
    )


def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def print_summary(result):
    meta = result["metadata"]
    print(
        "counts "
        f"gt={meta['ground_truth_count']} pred={meta['prediction_count']} "
        f"matched={meta['matched_count']} "
        f"unmatched_gt={meta['unmatched_ground_truth_count']} "
        f"unmatched_pred={meta['unmatched_prediction_count']}"
    )
    print()

    for split_name, split in result["splits"].items():
        print(f"[{split_name}] n={split['n']} ({fmt(split['pct'])}%)")
        if split["n"] == 0:
            print()
            continue
        for threshold, values in split["r1"].items():
            cells = [f"th={threshold}"]
            for key, value in values.items():
                label = "R1@1" if key == "oracle@1" else key
                cells.append(f"{label}={fmt(value)}")
            if "oracle@1" in values and "oracle@10" in values:
                cells.append(f"gain@10={fmt(values['oracle@10'] - values['oracle@1'])}")
            print("  " + " ".join(cells))

        length_diag = split.get("length_diagnostics", {})
        if length_diag:
            gt_mean = length_diag["gt_length"]["mean"]
            pred_mean = length_diag["top1_length"]["mean"]
            ratio_med = length_diag["top1_length_ratio"]["median"]
            print(
                "  lengths "
                f"gt_mean={fmt(gt_mean)} top1_mean={fmt(pred_mean)} "
                f"ratio_median={fmt(ratio_med)} "
                f"too_long={fmt(length_diag['top1_too_long_gt_1_5x_pct'])}% "
                f"too_short={fmt(length_diag['top1_too_short_lt_0_67x_pct'])}%"
            )
        candidate_diag = split.get("candidate_diagnostics")
        if candidate_diag:
            gain_key = next(
                (key for key in candidate_diag if key.startswith("iou_gain_top") and key.endswith("_minus_top1")),
                None,
            )
            gain = candidate_diag.get(gain_key, {}) if gain_key else {}
            score_gap = candidate_diag.get("top1_score_minus_best_iou_score", {})
            rank1 = candidate_diag["best_iou_rank"]["rank1"]["pct"]
            rank2_5 = candidate_diag["best_iou_rank"]["rank2-5"]["pct"]
            print(
                "  candidates "
                f"best_iou_rank1={fmt(rank1)}% "
                f"best_iou_rank2-5={fmt(rank2_5)}% "
                f"mean_iou_gain={fmt(100.0 * gain.get('mean', 0.0))} "
                f"median_score_gap={fmt(score_gap.get('median'))}"
            )
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute R1@1 and oracle R1@K diagnostics for VMR predictions."
    )
    parser.add_argument("--pred-path", default="best_charades_sta_val_preds.jsonl")
    parser.add_argument("--gt-path", default="test.jsonl")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--candidate-diagnostics", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = tuple(args.thresholds)
    ks = tuple(sorted(set(args.ks)))
    result = analyze(
        Path(args.pred_path),
        Path(args.gt_path),
        thresholds,
        ks,
        include_candidate_diagnostics=args.candidate_diagnostics,
    )
    print_summary(result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
