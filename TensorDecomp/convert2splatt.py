import os
import pandas as pd

data_name = 'family'
entities = pd.read_csv(os.path.join('..', 'datasets', data_name, 'entities.txt'), delimiter='\t', names=['name'])
entity2id_dict = {v['name']: idx+1 for idx, v in entities.iterrows()}
relations = pd.read_csv(os.path.join('..', 'datasets', data_name, 'relations.txt'), delimiter='\t', names=['name'])
rel2id_dict = {v['name']: idx+1 for idx, v in relations.iterrows()}

_facts = pd.read_csv(os.path.join('..', 'datasets', data_name, 'facts.txt'), delimiter='\t', names=['h', 'r', 't'])
_trains = pd.read_csv(os.path.join('..', 'datasets', data_name, 'train.txt'), delimiter='\t', names=['h', 'r', 't'])
_valids = pd.read_csv(os.path.join('..', 'datasets', data_name, 'valid.txt'), delimiter='\t', names=['h', 'r', 't'])
tests = pd.read_csv(os.path.join('..', 'datasets', data_name, 'test.txt'), delimiter='\t', names=['h', 'r', 't'])
trains = _facts.append(_trains, ignore_index=True).append(_valids, ignore_index=True)
n = len(trains)
trains = trains.drop_duplicates()
print("{} duplicates found.".format(n - len(trains)))
assert len(_facts) + len(_trains) + len(_valids) == len(trains)

trains['h'] = list(map(entity2id_dict.get, trains['h'].values))
trains['r'] = list(map(rel2id_dict.get, trains['r'].values))
trains['t'] = list(map(entity2id_dict.get, trains['t'].values))
#print(trains)
print("#unique head:{}, #unique relation:{}, #unqiue tail:{}".format(
    trains['h'].nunique(), trains['r'].nunique(), trains['t'].nunique()))


### save files
save_dir = os.path.join('data', data_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
#print(trains)
with open(os.path.join(save_dir, 'train.tns'), 'w') as save_file:
    for h,r,t in trains.values:
        save_file.write('{} {} {} 1.0\n'.format(h,t,r))

# entities.to_csv(os.path.join(save_dir, 'entities.dict'), sep='\t', header=False)
# relations.to_csv(os.path.join(save_dir, 'relations.dict'), sep='\t', header=False)
# trains.to_csv(os.path.join(save_dir, 'train.txt'), sep='\t', header=False, index=False)
# tests.to_csv(os.path.join(save_dir, 'valid.txt'), sep='\t', header=False, index=False)
# tests.to_csv(os.path.join(save_dir, 'test.txt'), sep='\t', header=False, index=False)

print('{} train, {} valid, {} test'.format(len(trains), len(tests), len(tests)))
