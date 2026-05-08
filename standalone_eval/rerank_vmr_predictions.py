#!/usr/bin/env python3
"""Offline reranking experiments for VMR-DETR prediction JSONL files."""

import argparse
import copy
import json
import math
from pathlib import Path

from analyze_oracle_r1 import (
    DEFAULT_KS,
    DEFAULT_THRESHOLDS,
    analyze_rows,
    load_jsonl,
    print_summary,
)


VARIANTS = (
    "baseline",
    "saliency_mean",
    "saliency_contrast",
    "saliency_topk",
    "saliency_trimmed",
    "length_aware",
)


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def safe_mean(values, default=0.0):
    return sum(values) / len(values) if values else default


def safe_logit(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def zscores(values):
    if not values:
        return []
    mean_value = safe_mean(values)
    variance = safe_mean([(value - mean_value) ** 2 for value in values])
    std = math.sqrt(max(variance, 0.0))
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean_value) / std for value in values]


def minmax_scores(values):
    if not values:
        return []
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def median(values):
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def window_length(window):
    return max(0.0, float(window[1]) - float(window[0]))


def saliency_indices(window, saliency_len, clip_length):
    if saliency_len <= 0:
        return 0, 0
    start = max(0, int(math.floor(float(window[0]) / clip_length)))
    end = min(saliency_len, int(math.ceil(float(window[1]) / clip_length)))
    if end <= start:
        end = min(saliency_len, start + 1)
    return start, end


def saliency_values_for_window(saliency_scores, window, clip_length):
    start, end = saliency_indices(window, len(saliency_scores), clip_length)
    return saliency_scores[start:end]


def context_saliency_values(saliency_scores, window, clip_length, context_ratio):
    if not saliency_scores:
        return []
    length = max(clip_length, window_length(window))
    context_len = length * context_ratio
    left = [float(window[0]) - context_len, float(window[0])]
    right = [float(window[1]), float(window[1]) + context_len]
    return (
        saliency_values_for_window(saliency_scores, left, clip_length)
        + saliency_values_for_window(saliency_scores, right, clip_length)
    )


def topk_mean(values, ratio):
    if not values:
        return 0.0
    k = max(1, int(math.ceil(len(values) * ratio)))
    return safe_mean(sorted(values, reverse=True)[:k])


def trimmed_mean(values, trim_ratio):
    if not values:
        return 0.0
    values = sorted(values)
    trim_n = int(math.floor(len(values) * trim_ratio))
    if trim_n == 0 or 2 * trim_n >= len(values):
        return safe_mean(values)
    return safe_mean(values[trim_n:-trim_n])


def candidate_components(row, candidates, clip_length, context_ratio, topk_ratio, trim_ratio):
    saliency_scores = [float(value) for value in row.get("pred_saliency_scores", [])]
    global_saliency_mean = safe_mean(saliency_scores)
    components = {
        "confidence": [],
        "confidence_logit": [],
        "saliency_mean": [],
        "saliency_topk": [],
        "saliency_trimmed": [],
        "saliency_contrast": [],
        "length": [],
    }

    for window in candidates:
        confidence = float(window[2]) if len(window) > 2 else 0.0
        inside = saliency_values_for_window(saliency_scores, window, clip_length)
        outside = context_saliency_values(saliency_scores, window, clip_length, context_ratio)
        inside_mean = safe_mean(inside, default=global_saliency_mean)
        outside_mean = safe_mean(outside, default=global_saliency_mean)

        components["confidence"].append(confidence)
        components["confidence_logit"].append(safe_logit(confidence))
        components["saliency_mean"].append(inside_mean)
        components["saliency_topk"].append(topk_mean(inside, topk_ratio))
        components["saliency_trimmed"].append(trimmed_mean(inside, trim_ratio))
        components["saliency_contrast"].append(inside_mean - outside_mean)
        components["length"].append(window_length(window))

    return components


def final_scores_for_variant(variant, components):
    if variant == "baseline":
        return components["confidence"]

    z_conf = zscores(components["confidence_logit"])
    z_mean = zscores(components["saliency_mean"])
    z_topk = zscores(components["saliency_topk"])
    z_trimmed = zscores(components["saliency_trimmed"])
    z_contrast = zscores(components["saliency_contrast"])
    z_length = zscores(components["length"])

    scores = []
    contrast_median = median(components["saliency_contrast"])
    for idx in range(len(components["confidence"])):
        if variant == "saliency_mean":
            score = z_conf[idx] + z_mean[idx]
        elif variant == "saliency_contrast":
            score = z_conf[idx] + 0.5 * z_mean[idx] + z_contrast[idx]
        elif variant == "saliency_topk":
            score = z_conf[idx] + z_topk[idx]
        elif variant == "saliency_trimmed":
            score = z_conf[idx] + z_trimmed[idx]
        elif variant == "length_aware":
            weak_support = components["saliency_contrast"][idx] <= contrast_median
            long_weak_penalty = max(0.0, z_length[idx]) if weak_support else 0.0
            score = z_conf[idx] + 0.5 * z_mean[idx] + z_contrast[idx] - 0.25 * long_weak_penalty
        else:
            raise ValueError(f"Unknown reranking variant: {variant}")
        scores.append(score)
    return scores


def replacement_scores(final_scores, original_candidates, score_output):
    if score_output == "original":
        return [float(window[2]) if len(window) > 2 else 0.0 for window in original_candidates]
    if score_output == "rank":
        n = len(final_scores)
        return [1.0 - (rank / (n + 1.0)) for rank in range(1, n + 1)]
    if score_output == "rerank":
        return minmax_scores(final_scores)
    raise ValueError(f"Unknown score output mode: {score_output}")


def rerank_row(
    row,
    variant,
    clip_length,
    context_ratio,
    topk_ratio,
    trim_ratio,
    max_candidates,
    score_output,
):
    out = copy.deepcopy(row)
    pred_windows = sorted(
        out.get("pred_relevant_windows", []),
        key=lambda window: float(window[2]) if len(window) > 2 else 0.0,
        reverse=True,
    )
    candidates = pred_windows[:max_candidates]
    tail = pred_windows[max_candidates:]
    if not candidates:
        return out

    components = candidate_components(
        out, candidates, clip_length, context_ratio, topk_ratio, trim_ratio
    )
    final_scores = final_scores_for_variant(variant, components)
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (final_scores[pair[0]], -pair[0]),
        reverse=True,
    )
    ordered_scores = replacement_scores(
        [final_scores[idx] for idx, _ in ranked],
        [window for _, window in ranked],
        score_output,
    )

    reranked = []
    for score, (_, window) in zip(ordered_scores, ranked):
        reranked.append([float(window[0]), float(window[1]), float(f"{score:.6f}")])
    for window in tail:
        reranked.append([float(window[0]), float(window[1]), float(window[2])])

    out["pred_relevant_windows"] = reranked
    return out


def rerank_rows(rows, variant, args):
    return [
        rerank_row(
            row,
            variant=variant,
            clip_length=args.clip_length,
            context_ratio=args.context_ratio,
            topk_ratio=args.topk_ratio,
            trim_ratio=args.trim_ratio,
            max_candidates=args.max_candidates,
            score_output=args.score_output,
        )
        for row in rows
    ]


def output_path_for_variant(pred_path, output_dir, output_path, variant, variant_count):
    if output_path is not None:
        if variant_count > 1:
            raise ValueError("--output-path can only be used with a single variant")
        return Path(output_path)
    pred_path = Path(pred_path)
    return Path(output_dir) / f"{pred_path.stem}_rerank_{variant}{pred_path.suffix}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerank VMR prediction windows with saliency-aware scoring variants."
    )
    parser.add_argument("--pred-path", default="best_charades_sta_val_preds.jsonl")
    parser.add_argument("--gt-path", default="test.jsonl")
    parser.add_argument(
        "--variant",
        default="all",
        choices=("all",) + VARIANTS,
        help="Reranking variant to run. Use all to run every preset.",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Evaluate variants without writing JSONL files.")
    parser.add_argument("--clip-length", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--context-ratio", type=float, default=1.0)
    parser.add_argument("--topk-ratio", type=float, default=0.25)
    parser.add_argument("--trim-ratio", type=float, default=0.2)
    parser.add_argument("--score-output", choices=("rerank", "rank", "original"), default="rerank")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    return parser.parse_args()


def main():
    args = parse_args()
    pred_rows = load_jsonl(args.pred_path)
    gt_rows = load_jsonl(args.gt_path) if args.gt_path else None
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    metrics = {}

    for variant in variants:
        print(f"=== variant: {variant} ===")
        reranked_rows = rerank_rows(pred_rows, variant, args)
        if not args.dry_run:
            out_path = output_path_for_variant(
                args.pred_path, args.output_dir, args.output_path, variant, len(variants)
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(reranked_rows, out_path)
            print(f"saved {out_path}")
        if gt_rows is not None:
            result = analyze_rows(
                reranked_rows,
                gt_rows,
                thresholds=tuple(args.thresholds),
                ks=tuple(sorted(set(args.ks))),
                pred_path=f"{args.pred_path}::{variant}",
                gt_path=args.gt_path,
            )
            metrics[variant] = result
            print_summary(result)

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
