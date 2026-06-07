# Innovation Notes for VMR-DETR

This document summarizes the current implementation ideas in this repository that
extend a QD-DETR-style video moment retrieval baseline. It is intended to support
paper and thesis writing while keeping claims grounded in the code and available
experiments.

## Current Method Positioning

The strongest coherent paper story in the current codebase is:

> Start DETR decoder queries from explicit temporal anchors, then refine their
> boundaries through reference-relative offset distributions.

This motivates the working paper name:

**DBR-DETR: Anchor-Guided Distributional Boundary Refinement for Video Moment Retrieval**

The central method consists of:

1. Temporal anchor query initialization.
2. Fine-grained distributional refinement (FDR) of start and end boundaries.
3. IoU-weighted distribution supervision (implemented as `loss_fgl`).
4. Localization-aware confidence learning with Varifocal Loss (VFL).
5. Log-width regularization for moment duration.

The repository also implements hard-negative saliency training, alternative
matching strategies, decoder-layer self-distillation, and a learned relational
reranker. These should be presented as secondary or optional components unless
controlled ablations show that they consistently improve the main results.

## Implementation Status

The following distinctions are important when describing the method:

| Component | Implemented | Current `train.sh` | Existing recorded checkpoint |
| --- | --- | --- | --- |
| Temporal anchor queries | Yes | Enabled | Enabled |
| FDR boundary refinement | Yes | Enabled | Enabled |
| FGL distribution loss | Yes | Enabled | Enabled |
| Log-width loss | Yes | Enabled | Enabled |
| VFL confidence loss | Yes | Enabled | Enabled |
| Contrastive query-text alignment | Yes | Enabled | Enabled |
| Intra-video hard-negative saliency | Yes | Enabled | Enabled |
| EMA scheduling | Yes | Enabled | Enabled |
| Relational reranker | Yes, experimental | Enabled in working tree | Not present |
| GO-LSD self-distillation | Yes | Disabled | Disabled |
| Direct start/end L1 loss | Yes | Disabled | Disabled |
| Task-aligned matching | Yes | Disabled | Disabled |
| One-to-many auxiliary matching | Yes | Disabled | Disabled |

The temporal pyramid and earlier local/multi-scale blocks were removed from the
current model. They must not be described as contributions of the present
version.

## 1. Temporal Anchor Query Initialization

The option `--query_init temporal_anchors` initializes each decoder query with a
normalized temporal reference `(center, width)` instead of relying only on
randomly initialized reference points.

When explicit widths are supplied, queries are distributed over those duration
scales and their centers are placed so that the initial moments remain within
the normalized video interval. The current Charades-STA recipe uses:

```bash
--query_init temporal_anchors \
--query_anchor_widths 0.08,0.22,0.48
```

The repository also provides a script that estimates duration priors from
training annotations.

Why it matters:

- Each query begins as an interpretable temporal hypothesis.
- Multiple duration scales are represented before decoder refinement.
- Dataset duration statistics can be introduced as a transparent prior.
- FDR can refine boundaries relative to a meaningful initial span.

Main code:

- `vmr_detr/modeling/model.py`: `init_temporal_queries`, `VMRDETR`.
- `vmr_detr/config/options.py`: `--query_init`, `--query_anchor_widths`.
- `vmr_detr/scripts/estimate_query_anchor_widths.py`.

## 2. Fine-Grained Distributional Refinement

With `--span_loss_type fdr`, each query predicts distributions over residual
start and end offsets rather than directly regressing only a center and width.

For a reference span with boundaries `(s_ref, e_ref)` and width `w_ref`, the
decoded boundaries follow the form:

```text
s_pred = s_ref + delta_s * max(w_ref, w_min)
e_pred = e_ref + delta_e * max(w_ref, w_min)
```

The offset support is symmetric around zero and made non-uniform by
`--fdr_reg_scale`. With the default scale greater than one, bins are denser near
zero, giving the model finer resolution for small boundary corrections.

Decoder-layer FDR logits are accumulated across layers and decoded against the
initial reference. This gives the decoder an iterative residual-refinement
interpretation while retaining a stable anchor.

Why it matters:

- Boundary errors are modeled explicitly at the start and end.
- Relative offsets adapt to both short and long reference moments.
- Dense support near zero emphasizes precise local corrections.
- Distributional outputs expose uncertainty information, such as boundary
  entropy, that can support later ranking.

Main code:

- `fdr_offset_support`.
- `fdr_logits_to_spans`.
- `_decode_fdr_cumulative_outputs`.
- `_fdr_offset_targets`.

## 3. IoU-Weighted FDR Supervision (`loss_fgl`)

The FDR training path interpolates each target offset between its two
neighboring support bins. It computes weighted cross-entropy for those bins and
then scales the boundary loss by the temporal IoU of the decoded prediction.

This combines:

- Soft supervision for continuous offsets between discrete bins.
- Stronger distribution supervision for predictions that already have useful
  localization quality.
- Normalization by the logarithm of the number of bins.

The implementation calls this term `loss_fgl`. The paper should explain the
actual equation rather than relying on the acronym alone.

Current recipe:

```bash
--span_loss_type fdr \
--fdr_num_bins 32 \
--fdr_reg_scale 1.5 \
--fgl_loss_coef 1.5
```

## 4. Auxiliary Localization Objectives

### GIoU loss

Temporal generalized IoU remains a primary localization objective and
complements the boundary distributions with direct overlap supervision.

### Log-width regularization

The current recipe applies L1 loss in log-width space:

```text
L_width = |log(w_pred) - log(w_gt)|
```

This penalizes relative duration errors. For example, doubling a short moment
and doubling a long moment receive comparable treatment in log space.

Current recipe:

```bash
--width_loss_type log \
--width_loss_coef 0.5
```

### Direct start/end loss

The code optionally applies L1 loss directly to normalized start/end
coordinates through `--span_xx_loss_coef`. It is currently set to zero and
must not be described as active in the recorded experiment.

## 5. Localization-Aware Confidence with VFL

The current recipe uses `--label_loss_type vfl`. Matched queries receive their
temporal IoU as a soft positive target, while unmatched queries are trained as
negatives with focal weighting.

Why it matters:

- Query confidence is encouraged to reflect localization quality.
- Ranking becomes better aligned with temporal IoU than hard binary labels
  alone.
- It complements FDR: FDR improves boundaries, while VFL encourages the score
  to represent the quality of those boundaries.

The repository also contains a separate `quality` label mode with an epoch
ramp, but the current recipe uses VFL.

Constraints:

- `quality` and `vfl` are supported with Hungarian matching.
- They require continuous span modes such as `l1`, `dfl`, or `fdr`.

## 6. Intra-Video Hard-Negative Saliency Training

The dataset loader builds a pool of clips covered by other annotated moments in
the same video. Candidate moments whose temporal IoU with the current target is
above a threshold are excluded to reduce false-negative supervision.

Two mechanisms are implemented:

1. **Level A sampling:** a configurable fraction of ordinary negative clips is
   replaced by clips from the same-video hard pool.
2. **Level B loss:** dedicated hard-negative clip labels add a margin-ranking
   term against positive saliency clips.

The Level B coefficient is warmed up and linearly ramped to reduce instability
early in training.

Current recipe:

```bash
--intra_video_hard_neg_ratio 0.3 \
--intra_video_hardneg_iou_thd 0.1 \
--saliency_hardneg_margin 0.4 \
--hardneg_loss_coef 0.5 \
--hardneg_warmup_epoch 5 \
--hardneg_ramp_epoch 20
```

This is most naturally framed as stronger discrimination of semantically
plausible but query-incorrect clips. Its contribution to moment retrieval must
be established through an ablation because the loss acts directly on the
saliency branch.

Main code:

- `vmr_detr/data/start_end_dataset.py`: same-video moment indexing and sampling.
- `vmr_detr/modeling/model.py`: hard-negative saliency margin loss.
- `tests/test_intra_video_hard_neg.py`.

## 7. Contrastive Query-Text Alignment

The model optionally applies an NCE-style objective between matched decoder
queries and encoded text tokens. The current recipe enables it from epoch 10.

Why it matters:

- It keeps decoder query representations semantically connected to the input
  language.
- It adds cross-modal supervision beyond span and confidence losses.
- Its coefficient can change after a configured epoch.

In the current script, the base and post-decay coefficients are both `0.3`, so
the configured decay epoch does not actually reduce the weight.

## 8. Learned Relational Reranker (Experimental)

The current working tree adds a `RelationalReranker` over final decoder queries.
Each query token combines:

- Decoder hidden state.
- Predicted center, width, start, and end.
- In-span saliency mean, maximum, and in/out contrast.
- Start/end distribution entropy when available.
- The query's original classification logit.

Self-attention models relations among candidate moments, with pairwise temporal
IoU projected into an attention bias. A scalar head predicts `rerank_logits`.
The reranker is trained with a listwise KL objective whose soft target is
derived from each candidate's maximum IoU with the ground-truth moments.

Important status:

- The implementation and CPU unit tests are present.
- The current working-tree `train.sh` enables the reranker.
- The existing checkpoint and metrics in `results/log_vmr.txt` were produced
  before reranker options were added.
- Therefore, the existing metrics do not validate the reranker.

Do not put relational reranking in the paper title or claim that it improves
performance until a controlled baseline-versus-reranker experiment is complete.

Main code:

- `vmr_detr/modeling/model.py`: `RelationalReranker`, `loss_reranker`.
- `vmr_detr/cli/inference.py`: optional inference-time reranking.
- `tests/test_relational_reranker.py`.

## 9. Optional Components Not Active in the Current Result

### Task-aligned and one-to-many matching

The repository supports:

- Task-aligned matching based on classification confidence and temporal IoU.
- One-to-many matching for auxiliary decoder supervision.

The current recipe uses Hungarian matching for both final and auxiliary
outputs.

### GO-LSD decoder self-distillation

GO-LSD transfers final-layer FDR distributions to earlier decoder layers using
temperature-scaled KL divergence and localization-aware query weights. It
requires FDR and auxiliary decoder losses.

The current recipe sets `--go_lsd_loss_coef 0.0`, so it is disabled.

## 10. EMA Scheduling

The training loop maintains exponential moving average model weights. EMA can
start after a selected epoch and warm its decay from an initial value toward a
target value with a configured schedule.

Current recipe:

```bash
--ema_decay 0.999 \
--ema_scheduler \
--ema_start_epoch 1 \
--ema_start_decay 0.99 \
--ema_warmup_updates 2000 \
--ema_schedule cosine
```

EMA should be described as a training and evaluation stabilization technique,
not as the main architectural contribution.

## Current Charades-STA Result

`results/_eval_metrics_v5.json` records:

| Metric | Value |
| --- | ---: |
| R1@0.3 | 72.23 |
| R1@0.5 | 61.83 |
| R1@0.7 | 41.08 |
| mAP average | 42.45 |
| mAP@0.5 | 72.18 |
| mAP@0.75 | 41.49 |
| mIoU | 52.59 |

The corresponding checkpoint was saved at epoch 60. These values demonstrate
one trained configuration, but they are not evidence of improvement by
themselves. Paper claims require comparisons under identical features,
evaluation code, data splits, and random-seed protocols.

The result contains no long-duration examples under the evaluator's
`[30, 150]` category, so the reported long-moment metrics are zero and should
not be interpreted as model failure on an evaluated long-moment subset.

## Recommended Ablation Order

To support a focused DBR-DETR paper, use one baseline configuration and add:

1. Baseline QD-DETR-style direct span regression.
2. Temporal anchor initialization.
3. FDR with FGL supervision.
4. Log-width regularization.
5. VFL confidence supervision.
6. Intra-video hard negatives.
7. Relational reranker.

Report at least three seeds when computationally feasible. Include parameter
count and inference cost for the reranker. Do not combine all optional modules
into the first comparison because that would make the source of any gain
unclear.

## Paper-Writing Guidance

Defensible wording:

- "We initialize decoder queries as multi-duration temporal hypotheses."
- "We formulate boundary prediction as reference-relative start/end offset
  distributions."
- "A non-uniform support allocates greater resolution to small boundary
  corrections."
- "IoU-aware confidence supervision aligns ranking scores with localization
  quality."
- "Same-video moments provide semantically plausible hard negatives for
  saliency learning."
- "The relational reranker is evaluated as an optional candidate-ranking
  module."

Avoid unsupported wording:

- "State of the art" without a complete current comparison.
- "The reranker improves retrieval" before reranker ablations are available.
- "Multi-scale temporal pyramid" because that module was removed.
- "Joint moment retrieval and highlight detection improvement" based only on
  Charades-STA moment-retrieval evaluation.
- Any numerical gain without a matched baseline and multiple-run analysis.

## Code Map

| Component | Main locations |
| --- | --- |
| Temporal anchors | `model.py`, `options.py`, `estimate_query_anchor_widths.py` |
| FDR/FGL | `model.py`, `options.py`, `test_task_aligned_matching.py` |
| Width and boundary losses | `model.py`, `options.py` |
| VFL and quality labels | `model.py`, `options.py` |
| Hard-negative saliency | `start_end_dataset.py`, `model.py`, `test_intra_video_hard_neg.py` |
| Contrastive alignment | `model.py`, `transformer.py`, `options.py` |
| Relational reranker | `model.py`, `inference.py`, `test_relational_reranker.py` |
| Matching variants | `matcher.py`, `options.py` |
| GO-LSD | `model.py`, `options.py` |
| EMA | `train.py`, `train_utils.py`, `test_ema_scheduler.py` |
