dset_name=activitynet
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_types=clip
results_root=results/activitynet
exp_id=exp

######## data paths
train_path=data/activitynet/train.jsonl
eval_path=data/activitynet/val_1.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/activitynet

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_clip)
  (( v_feat_dim += 512 ))
fi
#if [[ ${v_feat_types} == *"blip"* ]]; then
#  v_feat_dirs+=(${feat_root}/blip_video_features)
#  (( v_feat_dim += 768 ))
#fi

# text features
t_feat_dim=0
t_feat_dirs=()
if [[ ${t_feat_types} == *"clip"* ]]; then
  t_feat_dirs+=(${feat_root}/txt_clip)
  (( t_feat_dim += 512 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
#if [[ ${t_feat_types} == *"blip"* ]]; then
#  t_feat_dirs+=(${feat_root}/blip_text_features)
#  (( t_feat_dim += 768 ))
#fi

#### training
seed=2018
bsz=24
eval_bsz=32
lr=5e-05
lr_drop=100
n_epoch=100
max_v_l=-1
lw_saliency=1.0
VTC_loss_coef=0.3
CTC_loss_coef=0.5
# use_txt_pos=True
label_loss_coef=4
#pretrain_path=results/hl-video_tef-pt-2024_03_24_17_02_26/model_best.ckpt


PYTHONPATH=$PYTHONPATH:. python tr_detr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dirs ${t_feat_dirs[@]} \
--t_feat_dim ${t_feat_dim} \
--max_v_l ${max_v_l} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
${@:1}
