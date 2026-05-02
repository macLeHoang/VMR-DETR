dset_name=charades_sta
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_types=clip
results_root=results
exp_id=exp_nepoch_100_slowfast_clip_gated_contrastive

######## data paths
train_path=/content/drive/MyDrive/Master/Thesis/QD-DETR-Old/data/charades-sta/train.jsonl
eval_path=/content/drive/MyDrive/Master/Thesis/QD-DETR-Old/data/charades-sta/test.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/content/charades

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi
if [[ ${v_feat_types} == *"blip"* ]]; then
  v_feat_dirs+=(${feat_root}/blip_video_features)
  (( v_feat_dim += 768 ))
fi

# text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/clip_text_features)
  (( t_feat_dim += 512 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${t_feat_types} == *"blip"* ]]; then
  t_feat_dirs+=(${feat_root}/blip_text_features)
  (( t_feat_dim += 768 ))
fi

#### training
bsz=32
eval_bsz=32
n_epoch=100
clip_length=1
contrastive_align_loss_coef=0.3
contrastive_start_epoch=10
dec_layers=3
enc_layers=3
lr=1.5e-04
lr_drop=100
v_feat_len_mode=time_grid
num_workers=2

eval_every_epoch_after=40
ema_decay=0.999
max_es_cnt=10


PYTHONPATH=$PYTHONPATH:. python vmr_detr/cli/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dirs[@]} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--n_epoch ${n_epoch} \
--eval_bsz ${eval_bsz} \
--contrastive_align_loss \
--contrastive_align_loss_coef ${contrastive_align_loss_coef} \
--contrastive_start_epoch ${contrastive_start_epoch} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
--clip_length ${clip_length} \
--v_feat_len_mode ${v_feat_len_mode} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--exp_id ${exp_id} \
--num_workers ${num_workers} \
--eval_every_epoch_after ${eval_every_epoch_after} \
--ema_decay ${ema_decay} \
--use_gated_video_fusion \
--max_es_cnt ${max_es_cnt} \
${@:1}
