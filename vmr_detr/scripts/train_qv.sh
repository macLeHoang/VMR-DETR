#!/usr/bin/env bash

# Run identity
dset_name=hl
ctx_mode=video_tef
results_root=results
exp_id=exp_qv_highlight

# Data paths
train_path=/content/drive/MyDrive/Master/Thesis/QD-DETR-Old/data/highlight_trainfull_release.jsonl
eval_path=/content/drive/MyDrive/Master/Thesis/QD-DETR-Old/data/highlight_test_with_gt.jsonl
eval_split_name=val

# Feature selection
feat_root=/content/qvhighlight
v_feat_types=slowfast_clip
t_feat_types=clip
v_feat_len_mode=min

# Data augmentation
txt_drop_ratio=0.0
temporal_aug_prob=0.0
temporal_aug_min_keep=0.3
context_extend_prob=0.0
context_extend_max_frac=1.0
temporal_mask_prob=0.0
temporal_mask_n=1
temporal_mask_max_len=3
feat_noise_prob=0.01
feat_noise_std=0.02
multi_moment_prob=0.0
position_jitter_prob=0.0
position_jitter_context_sec=2.0
position_jitter_max_shift_frac=0.0
aug_stop_epoch=0

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
num_queries=10
clip_length=2
span_loss_type=fdr
fdr_num_bins=32
fdr_reg_scale=1.5
fdr_min_ref_width=0.0
fdr_decoder_refine_flag=
# --fdr_decoder_refine
fdr_guide_start_epoch=0
fdr_guide_ramp_epochs=0
query_anchor_widths=0.0400,0.1067,0.2400
# Pooled-text decoder query content initialization: none | mean | last
query_text_init=mean
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
contrastive_align_loss_flag=
# --contrastive_align_loss
contrastive_align_loss_coef=0.3
contrastive_start_epoch=0
contrastive_decay_epoch=-1
contrastive_decay_coef=0.0

# Losses: saliency
lw_saliency=1.0
saliency_margin=0.2

# Losses: intra-video hard negatives (Level A + B)
intra_video_hard_neg_ratio=0.0
intra_video_hardneg_iou_thd=0.1
saliency_hardneg_margin=0.4
hardneg_loss_coef=0.0
hardneg_warmup_epoch=5
hardneg_ramp_epoch=20

# Losses: region/ranking, disabled by default here
rank_within_loss_coef=0.0
region_contrast_loss_coef=0.0

# Optimization and evaluation
bsz=32
eval_bsz=32
n_epoch=100
lr=1.5e-4
lr_drop=100
lr_scheduler=cosine
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

query_text_init_args=()
case "${query_text_init}" in
  none|mean|last)
    echo "Using query_text_init=${query_text_init}"
    query_text_init_args=(--query_text_init "${query_text_init}")
    ;;
  *)
    echo "Invalid query_text_init=${query_text_init}; expected one of: none, mean, last." >&2
    exit 1
    ;;
esac

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
  --temporal_aug_prob "${temporal_aug_prob}" \
  --temporal_aug_min_keep "${temporal_aug_min_keep}" \
  --context_extend_prob "${context_extend_prob}" \
  --context_extend_max_frac "${context_extend_max_frac}" \
  --txt_drop_ratio "${txt_drop_ratio}" \
  --temporal_mask_prob "${temporal_mask_prob}" \
  --temporal_mask_n "${temporal_mask_n}" \
  --temporal_mask_max_len "${temporal_mask_max_len}" \
  --feat_noise_prob "${feat_noise_prob}" \
  --feat_noise_std "${feat_noise_std}" \
  --multi_moment_prob "${multi_moment_prob}" \
  --position_jitter_prob "${position_jitter_prob}" \
  --position_jitter_context_sec "${position_jitter_context_sec}" \
  --position_jitter_max_shift_frac "${position_jitter_max_shift_frac}" \
  --aug_stop_epoch "${aug_stop_epoch}" \
  --results_root "${results_root}" \
  --exp_id "${exp_id}" \
  --input_dropout "${input_dropout}" \
  --video_input_proj "${video_input_proj}" \
  --num_queries "${num_queries}" \
  --query_init temporal_anchors \
  "${query_anchor_widths_args[@]}" \
  "${query_text_init_args[@]}" \
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
  ${fdr_decoder_refine_flag} \
  --fdr_guide_start_epoch "${fdr_guide_start_epoch}" \
  --fdr_guide_ramp_epochs "${fdr_guide_ramp_epochs}" \
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
  ${contrastive_align_loss_flag} \
  --contrastive_align_loss_coef "${contrastive_align_loss_coef}" \
  --contrastive_start_epoch "${contrastive_start_epoch}" \
  --contrastive_decay_epoch "${contrastive_decay_epoch}" \
  --contrastive_decay_coef "${contrastive_decay_coef}" \
  --decoder_text_xattn \
  --rank_within_loss_coef "${rank_within_loss_coef}" \
  --region_contrast_loss_coef "${region_contrast_loss_coef}" \
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
