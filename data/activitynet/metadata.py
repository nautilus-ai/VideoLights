import pdb
import os
import json
import numpy as np
from tqdm import tqdm

clip_dir = '/data/home/qinghonglin/dataset/anet/features_clip_fps_2'
slowfast_dir = '/data/home/qinghonglin/dataset/anet/features_slowfast_fps_2'

clip_list = os.listdir(clip_dir)
slowfast_list = os.listdir(slowfast_dir)

for x in clip_list:
    if x.startswith('v_') and len(x) == 17:
        os.rename( os.path.join(clip_dir, x), os.path.join(clip_dir, x.replace('v_','')))

for x in slowfast_list:
    if x.startswith('v_') and len(x) == 17:
        os.rename( os.path.join(slowfast_dir, x), os.path.join(slowfast_dir, x.replace('v_','')))

clip_list = os.listdir(clip_dir)
slowfast_list = os.listdir(slowfast_dir)

inter_list = set(clip_list).intersection(set(slowfast_list))
inter_list = list(inter_list)

vid_dur = {}
clip_len = 2
# for vid in tqdm(inter_list):
    # feat = np.load(os.path.join('/data/home/qinghonglin/dataset/anet/features_clip_fps_2', vid))['features']
    # vid_dur[vid] = feat.shape[0] * clip_len

split = 'val_2'

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def load_jsonl(filename):
    with open(filename, "r") as f:
      return [json.loads(l.strip("\n")) for l in f.readlines()]

meta_path = os.path.join('/data/home/qinghonglin/dataset/anet/metadata', f'{split}.jsonl')
f = open(meta_path, 'w')

vid_infos = load_json(f'/data/home/qinghonglin/dataset/anet/metadata/offical/{split}.json')

for k, infos in tqdm(vid_infos.items()):
    if k[2:] + '.npz' in inter_list:
        annos = infos['sentences']
        for i, anno in enumerate(annos):
            start_end = infos['timestamps'][i]
            sample={
            'qid': f'{k}_{i}',
            'query': anno,
            'duration': infos['duration'],
            'vid': k[2:],
            'relevant_windows': [start_end]
            }
            if sample['relevant_windows'][0][1] > sample['relevant_windows'][0][0] and sample['duration'] > 0:
                f.write(json.dumps(sample))
                f.write('\n')
f.close()