import numpy as np
import pandas as pd
import os
data_name = 'family'

### read data and the learned embeddings
entities = pd.read_csv(os.path.join('..', 'datasets', data_name, 'entities.txt'),
                       delimiter='\t', names=['name'])
entity2id_dict = {v['name']: idx for idx, v in entities.iterrows()}
id2entity_dict = {idx: v['name'] for idx, v in entities.iterrows()}
relations = pd.read_csv(os.path.join('..', 'datasets', data_name, 'relations.txt'),
                        delimiter='\t', names=['name'])
rel2id_dict = {v['name']: idx for idx, v in relations.iterrows()}
id2rel_dict = {idx: v['name'] for idx, v in relations.iterrows()}
_facts = pd.read_csv(os.path.join('..', 'datasets', data_name, 'facts.txt'),
                     delimiter='\t', names=['h', 'r', 't'])
_trains = pd.read_csv(os.path.join('..', 'datasets', data_name, 'train.txt'),
                      delimiter='\t', names=['h', 'r', 't'])
_valids = pd.read_csv(os.path.join('..', 'datasets', data_name, 'valid.txt'),
                      delimiter='\t', names=['h', 'r', 't'])
trains = _facts.append(_trains, ignore_index=True).append(_valids, ignore_index=True)


## read the trained embeddings
head_embeds = np.loadtxt('mode1.mat')
rel_embeds = np.loadtxt('mode3.mat')
tail_embeds = np.loadtxt('mode2.mat')
with open('mode1.map', 'r') as f_head:
    head_id2splattid_dict = {}
    for idx, line in enumerate(f_head.readlines()):
        line = line.strip()
        head_id2splattid_dict[int(line)] = idx
with open('mode2.map', 'r') as f_tail:
    tail_id2splattid_dict = {}
    for idx, line in enumerate(f_tail.readlines()):
        line = line.strip()
        tail_id2splattid_dict[int(line)] = idx

## to check whether all the heads and tails appear in the splatt map

## generate observed heads given (r, tail)
observed_rt2h_dict = {}
for h,r,t in trains.values:

    if entity2id_dict[h] not in head_id2splattid_dict:
        continue
    head_idx_in_splatt = head_id2splattid_dict[entity2id_dict[h]]
    if (r,t) in observed_rt2h_dict:
        observed_rt2h_dict[(r,t)].append(head_idx_in_splatt)
    else:
        observed_rt2h_dict[(r,t)] = [head_idx_in_splatt]
tests = pd.read_csv(os.path.join('..', 'datasets', data_name, 'test.txt'),
                    delimiter='\t', names=['h', 'r', 't'])


def score(h,r,t):
    """

    :param h: (batch_size, dim)
    :param r: (dim, )
    :param t: (dim, )
    :return:
    """
    return np.dot(h, np.transpose(r*t))

# hit@1, hit@3, hit@5, hit@10
pred_l_all = [[],[],[],[]]
hit_thres = [1,3,5,10]
for gt_h, r, t in tests.values:
    r_idx = rel2id_dict[r]
    t_idx = tail_id2splattid_dict[entity2id_dict[t]]
    r_embed = rel_embeds[r_idx]
    t_embed = tail_embeds[t_idx]
    scores = score(head_embeds, r_embed, tail_embeds)
    scores[observed_rt2h_dict[(r,t)]] = float('-inf')
    preds = np.argsort(scores)
    gt_h_idx = head_id2splattid_dict[entity2id_dict[gt_h]]
    print('gt_h_idx', gt_h_idx)
    for idx, thres in enumerate(hit_thres):
        print(thres, preds[:thres])
        if gt_h_idx in preds[:thres]:
            pred_l_all[idx].append(1)
        else:
            pred_l_all[idx].append(0)

for idx, preds in pred_l_all:
    print("hit@{}: {}".format(hit_thres[idx], sum(preds)/len(preds)))



