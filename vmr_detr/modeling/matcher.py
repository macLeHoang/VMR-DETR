# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from torch import nn
import torch.nn.functional as F
from vmr_detr.ops.span_utils import generalized_temporal_iou, span_cxw_to_xx, temporal_iou

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    from itertools import permutations

    def linear_sum_assignment(cost_matrix):
        """Small fallback for environments without SciPy.

        VMR-DETR defaults to 10 queries and <=5 GT windows, so exhaustive search is acceptable as a fallback.
        """
        cost = torch.as_tensor(cost_matrix)
        num_rows, num_cols = cost.shape
        if num_rows == 0 or num_cols == 0:
            return [], []

        if num_rows >= num_cols:
            col_idx = tuple(range(num_cols))
            best_rows, best_cost = None, None
            for row_idx in permutations(range(num_rows), num_cols):
                total = cost[list(row_idx), list(col_idx)].sum().item()
                if best_cost is None or total < best_cost:
                    best_rows, best_cost = row_idx, total
            return list(best_rows), list(col_idx)

        row_idx = tuple(range(num_rows))
        best_cols, best_cost = None, None
        for col_idx in permutations(range(num_cols), num_rows):
            total = cost[list(row_idx), list(col_idx)].sum().item()
            if best_cost is None or total < best_cost:
                best_cols, best_cost = col_idx, total
        return list(row_idx), list(best_cols)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    def _compute_cost_matrix(self, outputs, targets):
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]
        sizes = [len(v["spans"]) for v in targets]
        if sum(sizes) == 0:
            return outputs["pred_spans"].new_empty((bs, num_queries, 0)).cpu(), sizes

        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.full(
            [len(tgt_spans)], self.foreground_label,
            dtype=torch.int64, device=out_prob.device
        )   # [total #spans in the batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type in ("l1", "dfl"):
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - \
                pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # pred_spans = pred_spans.repeat(1, n_spans, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, max_v_l, 2)
            # tgt_spans = tgt_spans.view(1, n_spans, 2).repeat(bs * num_queries, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, 2)
            # cost_span = pred_spans[tgt_spans]
            # cost_span = cost_span.view(bs * num_queries, n_spans)

            # giou
            cost_giou = 0

        # Final cost matrix
        # import ipdb; ipdb.set_trace()
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        return C, sizes

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        C, sizes = self._compute_cost_matrix(outputs, targets)
        empty = torch.empty(0, dtype=torch.int64)
        indices = []
        start = 0
        for batch_idx, size in enumerate(sizes):
            if size == 0:
                indices.append((empty, empty))
            else:
                indices.append(linear_sum_assignment(C[batch_idx, :, start:start + size]))
            start += size
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class OneToManyMatcher(HungarianMatcher):
    """Select the top-k lowest-cost queries for each target span.

    This matcher is intended for auxiliary decoder supervision. It keeps a query assigned to at most one target by
    retaining the lowest-cost target when candidate assignments collide.
    """

    def __init__(self, cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75, topk: int = 2):
        super().__init__(cost_class, cost_span, cost_giou, span_loss_type, max_v_l)
        if topk < 1:
            raise ValueError("topk must be >= 1 for one-to-many matching")
        self.topk = topk

    @torch.no_grad()
    def forward(self, outputs, targets):
        C, sizes = self._compute_cost_matrix(outputs, targets)
        indices = []
        empty = torch.empty(0, dtype=torch.int64)

        start = 0
        for batch_idx, size in enumerate(sizes):
            cost = C[batch_idx, :, start:start + size]  # [num_queries, num_target_spans]
            start += size
            num_queries, num_targets = cost.shape
            if num_queries == 0 or num_targets == 0:
                indices.append((empty, empty))
                continue

            k = min(self.topk, num_queries)
            candidate_costs, candidate_src = torch.topk(cost, k=k, dim=0, largest=False)
            candidate_tgt = torch.arange(num_targets, dtype=torch.int64).repeat(k)

            best_by_src = {}
            for src, tgt, match_cost in zip(
                    candidate_src.reshape(-1).tolist(),
                    candidate_tgt.tolist(),
                    candidate_costs.reshape(-1).tolist()):
                previous = best_by_src.get(src)
                if previous is None or match_cost < previous[0] or (
                        match_cost == previous[0] and tgt < previous[1]):
                    best_by_src[src] = (match_cost, tgt)

            selected = sorted((src, tgt) for src, (_, tgt) in best_by_src.items())
            src_idx = torch.as_tensor([src for src, _ in selected], dtype=torch.int64)
            tgt_idx = torch.as_tensor([tgt for _, tgt in selected], dtype=torch.int64)
            indices.append((src_idx, tgt_idx))

        return indices


class TaskAlignedMatcher(nn.Module):
    """Select top-k queries for each target using TOOD-style task alignment."""

    def __init__(self, span_loss_type: str = "l1", max_v_l: int = 75,
                 topk: int = 2, alpha: float = 1.0, beta: float = 6.0):
        super().__init__()
        if span_loss_type not in ("l1", "dfl"):
            raise ValueError("TaskAlignedMatcher requires span_loss_type to be 'l1' or 'dfl'.")
        if topk < 1:
            raise ValueError("topk must be >= 1 for task-aligned matching")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative for task-aligned matching")
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.foreground_label = 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_spans"].shape[:2]
        targets = targets["span_labels"]
        empty = torch.empty(0, dtype=torch.int64)
        indices = []

        pred_scores = outputs["pred_logits"].softmax(-1)[..., self.foreground_label]
        pred_spans_xx = span_cxw_to_xx(outputs["pred_spans"])

        for batch_idx in range(bs):
            tgt_spans = targets[batch_idx]["spans"]
            if num_queries == 0 or len(tgt_spans) == 0:
                indices.append((empty, empty))
                continue

            tgt_spans_xx = span_cxw_to_xx(tgt_spans)
            ious = temporal_iou(pred_spans_xx[batch_idx], tgt_spans_xx)[0].clamp(min=0)
            alignment = pred_scores[batch_idx].unsqueeze(1).pow(self.alpha) * ious.pow(self.beta)

            k = min(self.topk, num_queries)
            candidate_alignments, candidate_src = torch.topk(alignment, k=k, dim=0, largest=True)
            candidate_tgt = torch.arange(
                tgt_spans.shape[0], dtype=torch.int64, device=candidate_src.device
            ).repeat(k)

            best_by_src = {}
            for src, tgt, match_score in zip(
                    candidate_src.reshape(-1).tolist(),
                    candidate_tgt.tolist(),
                    candidate_alignments.reshape(-1).tolist()):
                previous = best_by_src.get(src)
                if previous is None or match_score > previous[0] or (
                        match_score == previous[0] and tgt < previous[1]):
                    best_by_src[src] = (match_score, tgt)

            selected = sorted((src, tgt) for src, (_, tgt) in best_by_src.items())
            src_idx = torch.as_tensor([src for src, _ in selected], dtype=torch.int64)
            tgt_idx = torch.as_tensor([tgt for _, tgt in selected], dtype=torch.int64)
            indices.append((src_idx, tgt_idx))

        return indices


def build_hungarian_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )


def build_task_aligned_matcher(args):
    return TaskAlignedMatcher(
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        topk=getattr(args, "tal_topk", 2),
        alpha=getattr(args, "tal_alpha", 1.0),
        beta=getattr(args, "tal_beta", 6.0),
    )


def build_matcher(args):
    matching_type = getattr(args, "matching_type", "hungarian")
    if matching_type == "hungarian":
        return build_hungarian_matcher(args)
    if matching_type == "tal":
        return build_task_aligned_matcher(args)
    raise ValueError(f"Unsupported matching_type: {matching_type}")


def build_one_to_many_matcher(args):
    return OneToManyMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        topk=getattr(args, "aux_one_to_many_k", 2)
    )
