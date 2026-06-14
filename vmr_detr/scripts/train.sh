#!/usr/bin/env bash

# Run identity
dset_name=charades_sta
ctx_mode=video_tef
results_root=results
exp_id=exp_fdr_quality_s05

# Data paths
train_path=/content/drive/MyDrive/Master/Thesis/QD-DETR-Old/data/charades-sta/train.jsonl
eval_path=/content/drive/MyDrive/Master/Thesis/QD-DETR-Old/data/charades-sta/test.jsonl
eval_split_name=val

# Feature selection
feat_root=/content/charades
v_feat_types=slowfast_clip
t_feat_types=clip
v_feat_len_mode=min

# Video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=("${feat_root}/slowfast_features")
  (( v_feat_dim += 2304 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=("${feat_root}/clip_features")
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"blip"* ]]; then
  v_feat_dirs+=("${feat_root}/blip_video_features")
  (( v_feat_dim += 768 ))
fi

# Text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=("${feat_root}/clip_text_features")
  (( t_feat_dim += 512 ))
fi
if [[ ${t_feat_types} == *"blip"* ]]; then
  t_feat_dirs+=("${feat_root}/blip_text_features")
  (( t_feat_dim += 768 ))
fi

# Model
input_dropout=0.5
video_input_proj=linear
enc_layers=3
dec_layers=3
clip_length=1
span_loss_type=fdr
fdr_num_bins=32
fdr_reg_scale=1.5
fdr_min_ref_width=0.0
query_anchor_widths=0.08,0.22,0.48
matching_type=hungarian
aux_matching_type=one_to_many
aux_one_to_many_k=3
set_cost_span=10
set_cost_giou=1
set_cost_class=4

# Losses: localization and labels
span_loss_coef=1.0
span_xx_loss_coef=0.0
fgl_loss_coef=1.5
giou_loss_coef=6.0
width_loss_type=log
width_loss_coef=0.5
label_loss_coef=12.0

# Losses: label supervision
label_loss_type=vfl

# Losses: GO-LSD self-distillation, disabled by default here
go_lsd_loss_coef=0.0
go_lsd_temperature=3.0
go_lsd_start_epoch=10

# Losses: contrastive query/text alignment
contrastive_align_loss_flag=--contrastive_align_loss
contrastive_align_loss_coef=0.3
contrastive_start_epoch=0
contrastive_decay_epoch=30
contrastive_decay_coef=0.3

# Losses: saliency
lw_saliency=1.0
saliency_margin=0.2

# Losses: intra-video hard negatives (Level A + B)
intra_video_hard_neg_ratio=0.3
intra_video_hardneg_iou_thd=0.1
saliency_hardneg_margin=0.4
hardneg_loss_coef=0.5
hardneg_warmup_epoch=5
hardneg_ramp_epoch=20

# Unified localization and confidence refinement stage
stage2_flag=--use_stage2
stage2_dim=128
stage2_inner_bins=4
stage2_boundary_samples=4
stage2_max_shift_clips=3
stage2_shift_frac=0.25
stage2_positive_iou=0.4
stage2_start_epoch=10
stage2_joint_epoch=20
stage2_boundary_loss_coef=0.5
stage2_giou_loss_coef=0.5
stage2_quality_loss_coef=1.0
stage2_at_inference_flag=--stage2_at_inference

# Optimization and evaluation
bsz=32
eval_bsz=32
n_epoch=100
lr=1.5e-4
lr_drop=100
lr_scheduler=step
lrf=0.01
num_workers=2
eval_every_epoch_after=40
ema_decay=0.999
ema_start_epoch=1
ema_start_decay=0.99
ema_warmup_updates=2000
ema_update_every=1
ema_schedule=cosine
max_es_cnt=10
best_metric=MR-full-R1@0.5+0.7

query_anchor_widths_args=()
if [[ -n "${query_anchor_widths}" ]]; then
  echo "Using query_anchor_widths=${query_anchor_widths}"
  query_anchor_widths_args=(--query_anchor_widths "${query_anchor_widths}")
else
  echo "Using default temporal anchor widths."
fi

PYTHONPATH="${PYTHONPATH}:." python vmr_detr/cli/train.py \
  --dset_name "${dset_name}" \
  --ctx_mode "${ctx_mode}" \
  --train_path "${train_path}" \
  --eval_path "${eval_path}" \
  --eval_split_name "${eval_split_name}" \
  --v_feat_dirs "${v_feat_dirs[@]}" \
  --v_feat_dim "${v_feat_dim}" \
  --t_feat_dir "${t_feat_dirs[@]}" \
  --t_feat_dim "${t_feat_dim}" \
  --v_feat_len_mode "${v_feat_len_mode}" \
  --results_root "${results_root}" \
  --exp_id "${exp_id}" \
  --input_dropout "${input_dropout}" \
  --video_input_proj "${video_input_proj}" \
  --query_init temporal_anchors \
  "${query_anchor_widths_args[@]}" \
  --bsz "${bsz}" \
  --eval_bsz "${eval_bsz}" \
  --n_epoch "${n_epoch}" \
  --enc_layers "${enc_layers}" \
  --dec_layers "${dec_layers}" \
  --clip_length "${clip_length}" \
  --span_loss_type "${span_loss_type}" \
  --fdr_num_bins "${fdr_num_bins}" \
  --fdr_reg_scale "${fdr_reg_scale}" \
  --fdr_min_ref_width "${fdr_min_ref_width}" \
  --matching_type "${matching_type}" \
  --aux_matching_type "${aux_matching_type}" \
  --aux_one_to_many_k "${aux_one_to_many_k}" \
  --set_cost_span "${set_cost_span}" \
  --set_cost_giou "${set_cost_giou}" \
  --set_cost_class "${set_cost_class}" \
  --span_loss_coef "${span_loss_coef}" \
  --span_xx_loss_coef "${span_xx_loss_coef}" \
  --fgl_loss_coef "${fgl_loss_coef}" \
  --giou_loss_coef "${giou_loss_coef}" \
  --width_loss_type "${width_loss_type}" \
  --width_loss_coef "${width_loss_coef}" \
  --label_loss_coef "${label_loss_coef}" \
  --label_loss_type "${label_loss_type}" \
  --go_lsd_loss_coef "${go_lsd_loss_coef}" \
  --go_lsd_temperature "${go_lsd_temperature}" \
  --go_lsd_start_epoch "${go_lsd_start_epoch}" \
  --lw_saliency "${lw_saliency}" \
  --saliency_margin "${saliency_margin}" \
  --intra_video_hard_neg_ratio "${intra_video_hard_neg_ratio}" \
  --intra_video_hardneg_iou_thd "${intra_video_hardneg_iou_thd}" \
  --saliency_hardneg_margin "${saliency_hardneg_margin}" \
  --hardneg_loss_coef "${hardneg_loss_coef}" \
  --hardneg_warmup_epoch "${hardneg_warmup_epoch}" \
  --hardneg_ramp_epoch "${hardneg_ramp_epoch}" \
  ${stage2_flag} \
  --stage2_dim "${stage2_dim}" \
  --stage2_inner_bins "${stage2_inner_bins}" \
  --stage2_boundary_samples "${stage2_boundary_samples}" \
  --stage2_max_shift_clips "${stage2_max_shift_clips}" \
  --stage2_shift_frac "${stage2_shift_frac}" \
  --stage2_positive_iou "${stage2_positive_iou}" \
  --stage2_start_epoch "${stage2_start_epoch}" \
  --stage2_joint_epoch "${stage2_joint_epoch}" \
  --stage2_boundary_loss_coef "${stage2_boundary_loss_coef}" \
  --stage2_giou_loss_coef "${stage2_giou_loss_coef}" \
  --stage2_quality_loss_coef "${stage2_quality_loss_coef}" \
  ${stage2_at_inference_flag} \
  ${contrastive_align_loss_flag} \
  --contrastive_align_loss_coef "${contrastive_align_loss_coef}" \
  --contrastive_start_epoch "${contrastive_start_epoch}" \
  --contrastive_decay_epoch "${contrastive_decay_epoch}" \
  --contrastive_decay_coef "${contrastive_decay_coef}" \
  --lr "${lr}" \
  --lr_drop "${lr_drop}" \
  --lr_scheduler "${lr_scheduler}" \
  --lrf "${lrf}" \
  --num_workers "${num_workers}" \
  --eval_every_epoch_after "${eval_every_epoch_after}" \
  --ema_decay "${ema_decay}" \
  --ema_scheduler \
  --ema_start_epoch "${ema_start_epoch}" \
  --ema_start_decay "${ema_start_decay}" \
  --ema_warmup_updates "${ema_warmup_updates}" \
  --ema_update_every "${ema_update_every}" \
  --ema_schedule "${ema_schedule}" \
  --max_es_cnt "${max_es_cnt}" \
  --best_metric "${best_metric}" \
  "$@"
