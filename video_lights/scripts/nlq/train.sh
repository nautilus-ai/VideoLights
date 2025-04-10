dset_name=nlq
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_types=clip
results_root=results/nlq
exp_id=exp

######## data paths
train_path=data/nlq/nlq_train.jsonl
eval_path=data/nlq/nlq_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/NLQ

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
bsz=32
eval_bsz=32
lr=2e-04
lr_drop=200
dec_layers=3
enc_layers=3
bicmf_layers=1
max_v_l=-1
clip_length=2
contrastive_align_loss_coef=0.002
hard_pos_neg_loss_coef=1
main_metric="MR-full-R1@0.3"
#pretrain_path=results/hl-video_tef-pt-2024_03_24_17_02_26/model_best.ckpt


PYTHONPATH=$PYTHONPATH:. python video_lights/train.py \
--dset_name ${dset_name} \
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
--exp_id exp-bicmf_${bicmf_layers}-en_${enc_layers}-dec_${dec_layers}-tcl-hl-scsl-cal_${contrastive_align_loss_coef}-${v_feat_types} \
--max_v_l ${max_v_l} \
--clip_length ${clip_length} \
--lr ${lr} \
--lr_drop ${lr_drop} \
--n_epoch 100 \
--lw_saliency 4 \
--eval_bsz ${eval_bsz} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
--bicmf_layers ${bicmf_layers} \
--mr_to_hd_loss \
--hard_pos_neg_loss \
--hard_pos_neg_loss_coef ${hard_pos_neg_loss_coef} \
--contrastive_align_loss \
--contrastive_align_loss_coef ${contrastive_align_loss_coef} \
--main_metric ${main_metric} \
${@:1}
