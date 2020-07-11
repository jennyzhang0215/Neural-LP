import numpy as np
import os

def gen_dict(file_name):
    d_dict = {}
    f = open(file_name, "r")
    for line in f:
        vec = line.strip('\n').split("\t")
        d_dict[vec[1]] = int(vec[0])
    return d_dict

data_name = 'fb15k-237'
test_file = os.path.join(data_name, 'test.txt')
rel_file = os.path.join(data_name, 'relations.tsv')
entity_file = os.path.join(data_name, 'entities.tsv')
rel_dict = gen_dict(rel_file)
entity_dict = gen_dict(entity_file)

print("rel_dict", rel_dict)
print("entity_dict", entity_dict)

tests = np.loadtxt(test_file, delimiter='\t', dtype=str)
#heads = list(map(entity_dict.get, tests[:1, 0]))
rels = list(map(rel_dict.get, tests[:1, 1]))
tails = list(map(entity_dict.get, tests[:1, 2]))

#np.savetxt(os.path.join(data_name, 'head.list'), heads, fmt='%s')
np.savetxt(os.path.join(data_name, 'rel.list'), rels, fmt='%s')
np.savetxt(os.path.join(data_name, 'tail.list'), tails, fmt='%s')
