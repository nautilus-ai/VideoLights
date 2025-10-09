dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip_blip
t_feat_types=clip_blip
results_root=results
exp_id=exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/qvhl/features

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

echo $v_feat_dim
echo ${v_feat_dirs[@]}

# text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/clip_text_features)
  (( t_feat_dim += 512 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${t_feat_types} == *"blip"* ]]; then
  t_feat_dirs+=(${feat_root}/blip_aug_text_features_openai)
  (( t_feat_dim += 768 ))
fi

echo $t_feat_dim
echo ${t_feat_dirs[@]}

#### training
bsz=32


PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dirs ${t_feat_dirs} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1}
