dset_name=youtube_uni
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results/youtubeuni
exp_id=exp-qddec-2000


######## data paths
# train_path=data/youtube_uni/youtube_train.jsonl
# eval_path=data/youtube_uni/youtube_anno.jsonl
train_path=data/youtube_uni/youtube_train.jsonl
eval_path=data/youtube_uni/youtube_valid.jsonl
eval_split_name=val

######## setup video+text features
feat_root=../Datasets/processed/youtube_uni

# # video features
v_feat_dim=2816
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/vid_clip)
v_feat_dirs+=(${feat_root}/vid_slowfast)

# # text features
t_feat_dirs=()
t_feat_dirs+=(${feat_root}/txt_clip/) # maybe not used
t_feat_dim=512


#### training
bsz=4
lr=2e-4


for dset_domain in dog gymnastics parkour skating skiing surfing
do
    for seed in 0 1 1009 2017 2018
    do
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
        --results_root ${results_root}/${dset_domain} \
        --exp_id ${exp_id}_${seed} \
        --max_v_l 1000 \
        --n_epoch 2000 \
        --max_es_cnt -1 \
        --seed $seed \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        --clip_length 1 \
        --lw_saliency 4 \
        ${@:1}
    done
done
