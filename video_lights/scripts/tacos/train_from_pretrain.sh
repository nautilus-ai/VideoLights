dset_name=tacos
ctx_mode=video_tef
v_feat_types=slowfast_clip_blip
t_feat_types=clip_blip
results_root=results/tacos
exp_id=ft_bicmf-3

######## data paths
train_path=data/tacos/train.jsonl
eval_path=data/tacos/val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/processed/tacos

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
lr=2e-4
lr_drop=200
n_epoch=200
eval_bsz=32
#pretrain_path=results/pretrain/hl-video_tef-pt-sf-clip-2024_03_30_12_44_33/model_best.ckpt
pretrain_path=results/pretrain/hl-video_tef-pt_hl_sf-clip-blip_fuse_calign_bicmf-3-2024_08_03_18_45_40/model_best.ckpt

PYTHONPATH=$PYTHONPATH:. python video_lights/train.py \
--dset_name ${dset_name} \
--resume ${pretrain_path} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dirs ${t_feat_dirs[@]} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--max_v_l -1 \
--clip_length 2 \
--lr ${lr} \
--lr_drop ${lr_drop} \
--n_epoch ${n_epoch} \
--contrastive_align_loss_coef 0.002 \
--lw_saliency 4 \
--eval_bsz ${eval_bsz} \
${@:1}
