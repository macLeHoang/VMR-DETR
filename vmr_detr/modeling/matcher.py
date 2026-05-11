# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from vmr_detr.ops.span_utils import generalized_temporal_iou, span_cxw_to_xx


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


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )


def build_one_to_many_matcher(args):
    return OneToManyMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        topk=getattr(args, "aux_one_to_many_k", 2)
    )
