"""Shared fine-grained distributional regression utilities."""

import math

import torch
import torch.nn.functional as F

from vmr_detr.ops.span_utils import span_cxw_to_xx, span_xx_to_cxw


def fdr_num_logits(fdr_num_bins):
    """Number of logits per FDR boundary.

    `fdr_num_bins` follows D-FINE's reg_max convention: bins are indexed
    0..reg_max, so the distribution has reg_max + 1 classes.
    """
    return int(fdr_num_bins) + 1


def fdr_offset_support(num_bins, reg_scale, device=None, dtype=None):
    """Symmetric non-uniform FDR offset support with reg_max + 1 bins."""
    if num_bins < 1:
        raise ValueError("num_bins/reg_max must be >= 1.")
    if reg_scale <= 0:
        raise ValueError("reg_scale must be > 0.")
    base = torch.linspace(-1.0, 1.0, fdr_num_logits(num_bins), device=device, dtype=dtype)
    return base.sign() * base.abs().pow(float(reg_scale))


def fdr_logits_to_spans(span_logits, reference_spans, fdr_num_bins,
                        fdr_reg_scale, fdr_min_ref_width):
    """Decode residual start/end offset distributions around references."""
    if fdr_min_ref_width <= 0:
        raise ValueError("fdr_min_ref_width must be > 0.")
    num_logits = fdr_num_logits(fdr_num_bins)
    logits = span_logits.reshape(*span_logits.shape[:-1], 2, num_logits)
    probabilities = F.softmax(logits, dim=-1)
    support = fdr_offset_support(
        fdr_num_bins,
        fdr_reg_scale,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    offsets = (
        probabilities
        * support.view(*([1] * (probabilities.dim() - 1)), num_logits)
    ).sum(dim=-1)

    reference_xx = span_cxw_to_xx(reference_spans).clamp(0, 1)
    reference_width = (
        reference_xx[..., 1] - reference_xx[..., 0]
    ).clamp(min=fdr_min_ref_width)
    predicted_start = reference_xx[..., 0] + offsets[..., 0] * reference_width
    predicted_end = reference_xx[..., 1] + offsets[..., 1] * reference_width
    start = torch.minimum(predicted_start, predicted_end).clamp(0, 1)
    end = torch.maximum(predicted_start, predicted_end).clamp(0, 1)
    return span_xx_to_cxw(torch.stack([start, end], dim=-1))


def _decode_fdr_cumulative_outputs(span_delta_logits, reference_spans, fdr_num_bins,
                                   fdr_reg_scale, fdr_min_ref_width):
    """Decode cumulative FDR logits against the initial decoder reference."""
    span_logits = torch.cumsum(span_delta_logits, dim=0)
    initial_reference = reference_spans[:1].expand_as(reference_spans)
    spans = fdr_logits_to_spans(
        span_logits,
        initial_reference,
        fdr_num_bins,
        fdr_reg_scale,
        fdr_min_ref_width,
    )
    return span_logits, spans, initial_reference


def fdr_logits_to_entropy(span_logits, fdr_num_bins):
    """Return normalized start/end posterior entropy for FDR logits."""
    num_logits = fdr_num_logits(fdr_num_bins)
    logits = span_logits.reshape(*span_logits.shape[:-1], 2, num_logits)
    probabilities = F.softmax(logits, dim=-1).clamp(min=1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    entropy = entropy / float(math.log(num_logits))
    return entropy
