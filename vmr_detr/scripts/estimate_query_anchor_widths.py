"""Estimate normalized temporal-anchor widths from a VMR train JSONL file."""

import argparse
import json

import numpy as np


def _parse_quantiles(raw_quantiles):
    quantiles = [float(value.strip()) for value in raw_quantiles.split(",") if value.strip()]
    if not quantiles:
        raise ValueError("At least one quantile is required.")
    for quantile in quantiles:
        if quantile < 0 or quantile > 1:
            raise ValueError("Quantiles must be in [0, 1].")
    return quantiles


def iter_normalized_widths(jsonl_path, min_width):
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            duration = float(item.get("duration", 0.0))
            if duration <= 0:
                continue
            for window in item.get("relevant_windows", []):
                if len(window) < 2:
                    continue
                start, end = float(window[0]), float(window[1])
                width = max(0.0, end - start) / duration
                yield float(np.clip(width, min_width, 1.0))


def estimate_anchor_widths(jsonl_path, quantiles, min_width):
    widths = np.asarray(list(iter_normalized_widths(jsonl_path, min_width)), dtype=np.float32)
    if widths.size == 0:
        raise ValueError("No valid relevant_windows with positive duration were found.")
    return widths, np.quantile(widths, quantiles)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate --query_anchor_widths from normalized training-window width quantiles."
    )
    parser.add_argument("train_jsonl", help="Training JSONL file containing duration and relevant_windows.")
    parser.add_argument("--quantiles", default="0.2,0.5,0.8",
                        help="Comma-separated quantiles in [0, 1].")
    parser.add_argument("--min_width", default=1e-3, type=float,
                        help="Minimum normalized width before quantile estimation.")
    parser.add_argument("--precision", default=4, type=int,
                        help="Decimal places used in the printed CLI value.")
    args = parser.parse_args()

    if args.min_width <= 0 or args.min_width > 1:
        raise ValueError("--min_width must be in (0, 1].")
    if args.precision < 0:
        raise ValueError("--precision must be >= 0.")

    quantiles = _parse_quantiles(args.quantiles)
    widths, anchors = estimate_anchor_widths(args.train_jsonl, quantiles, args.min_width)
    anchor_text = ",".join(f"{width:.{args.precision}f}" for width in anchors)

    print(f"num_widths: {widths.size}")
    print(f"quantiles: {','.join(str(quantile) for quantile in quantiles)}")
    print(f"query_anchor_widths: {anchor_text}")
    print(f"--query_anchor_widths {anchor_text}")


if __name__ == "__main__":
    main()
