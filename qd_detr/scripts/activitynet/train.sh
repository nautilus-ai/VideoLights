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
bsz=24
eval_bsz=32
lr=5e-05
lr_drop=100
dec_layers=3
enc_layers=3
bicmf_layers=1
max_v_l=-1
contrastive_align_loss_coef=0.1
hard_pos_neg_loss_coef=1
giou_loss_coef=2
main_metric="MR-full-R1@0.3"
#pretrain_path=results/hl-video_tef-pt-2024_03_24_17_02_26/model_best.ckpt


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
