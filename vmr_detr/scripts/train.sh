#!/usr/bin/env bash

# Run identity
dset_name=charades_sta
ctx_mode=video_tef
results_root=results
exp_id=exp_fdr_no_golsd_quality_s05_metricrank_r1guard_ramp10

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
enc_layers=3
dec_layers=3
clip_length=1
span_loss_type=fdr
dfl_num_bins=32
dfl_ref_prior_sigma=4.0
fdr_num_bins=32
fdr_reg_scale=1.5
fdr_min_ref_width=0.05
matching_type=hungarian
aux_matching_type=hungarian
aux_one_to_many_k=2
use_late_gated_video_fusion_flag=--use_late_gated_video_fusion
use_multiscale_stream_adapter_flag=--use_multiscale_stream_adapter
multiscale_adapter_dropout=0.1
multiscale_adapter_dilations=1,3,5,8,13
multiscale_adapter_kernel_size=5

# Losses: localization and labels
span_loss_coef=1.0
fgl_loss_coef=2.0
giou_loss_coef=1.0
width_loss_type=log
width_loss_coef=0.5
label_loss_coef=4.0

# Losses: label supervision
label_loss_type=vfl
vfl_alpha=0.75
vfl_gamma=2.0
quality_label_strength=0.5
quality_label_iou_gamma=1.0
quality_label_warmup_epoch=10
quality_label_ramp_epoch=30

# Losses: GO-LSD self-distillation, disabled by default here
go_lsd_loss_coef=0.3
go_lsd_temperature=2.0
go_lsd_start_epoch=0

# Losses: contrastive query/text alignment
contrastive_align_loss_flag=--contrastive_align_loss
contrastive_align_loss_coef=0.3
contrastive_start_epoch=10
contrastive_decay_epoch=30
contrastive_decay_coef=0.1

# Losses: saliency
lw_saliency=1.0
saliency_margin=0.2

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
max_es_cnt=10
best_metric=MR-full-R1@0.5+0.7

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
  --bsz "${bsz}" \
  --eval_bsz "${eval_bsz}" \
  --n_epoch "${n_epoch}" \
  --enc_layers "${enc_layers}" \
  --dec_layers "${dec_layers}" \
  --clip_length "${clip_length}" \
  --span_loss_type "${span_loss_type}" \
  --dfl_num_bins "${dfl_num_bins}" \
  --dfl_ref_prior_sigma "${dfl_ref_prior_sigma}" \
  --fdr_num_bins "${fdr_num_bins}" \
  --fdr_reg_scale "${fdr_reg_scale}" \
  --fdr_min_ref_width "${fdr_min_ref_width}" \
  --matching_type "${matching_type}" \
  --aux_matching_type "${aux_matching_type}" \
  --aux_one_to_many_k "${aux_one_to_many_k}" \
  --span_loss_coef "${span_loss_coef}" \
  --fgl_loss_coef "${fgl_loss_coef}" \
  --giou_loss_coef "${giou_loss_coef}" \
  --width_loss_type "${width_loss_type}" \
  --width_loss_coef "${width_loss_coef}" \
  --label_loss_coef "${label_loss_coef}" \
  --label_loss_type "${label_loss_type}" \
  --vfl_alpha "${vfl_alpha}" \
  --vfl_gamma "${vfl_gamma}" \
  --quality_label_strength "${quality_label_strength}" \
  --quality_label_iou_gamma "${quality_label_iou_gamma}" \
  --quality_label_warmup_epoch "${quality_label_warmup_epoch}" \
  --quality_label_ramp_epoch "${quality_label_ramp_epoch}" \
  --go_lsd_loss_coef "${go_lsd_loss_coef}" \
  --go_lsd_temperature "${go_lsd_temperature}" \
  --go_lsd_start_epoch "${go_lsd_start_epoch}" \
  --lw_saliency "${lw_saliency}" \
  --saliency_margin "${saliency_margin}" \
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
  --max_es_cnt "${max_es_cnt}" \
  --best_metric "${best_metric}" \
  ${use_late_gated_video_fusion_flag} \
  ${use_multiscale_stream_adapter_flag} \
  --multiscale_adapter_dropout "${multiscale_adapter_dropout}" \
  --multiscale_adapter_dilations "${multiscale_adapter_dilations}" \
  --multiscale_adapter_kernel_size "${multiscale_adapter_kernel_size}" \
  "$@"
