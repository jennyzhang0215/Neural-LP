import os
import pandas as pd

data_name = 'family'
entities = pd.read_csv(os.path.join('..', 'datasets', data_name, 'entities.txt'), delimiter='\t')
relations = pd.read_csv(os.path.join('..', 'datasets', data_name, 'relations.txt'), delimiter='\t')
facts = pd.read_csv(os.path.join('..', 'datasets', data_name, 'facts.txt'), delimiter='\t')
trains = pd.read_csv(os.path.join('..', 'datasets', data_name, 'train.txt'), delimiter='\t')
valids = pd.read_csv(os.path.join('..', 'datasets', data_name, 'valid.txt'), delimiter='\t')
tests = pd.read_csv(os.path.join('..', 'datasets', data_name, 'test.txt'), delimiter='\t')

save_dir = os.path.join('data', data_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
#entities.to_csv(os.path.join(save_dir, 'entities.dict'), sep='\t', header=False)
#relations.to_csv(os.path.join(save_dir, 'relations.dict'), sep='\t', header=False)
trains = facts.append(trains, ignore_index=True)
trains.to_csv(os.path.join(save_dir, 'train.txt'), sep='\t', header=False, index=False)
valids.to_csv(os.path.join(save_dir, 'valid.txt'), sep='\t', header=False, index=False)
tests.to_csv(os.path.join(save_dir, 'test.txt'), sep='\t', header=False, index=False)
