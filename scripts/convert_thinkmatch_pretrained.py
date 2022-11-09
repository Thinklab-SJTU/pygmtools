# Convert pretrained model weights from ThinkMatch https://github.com/Thinklab-SJTU/ThinkMatch
# The ThinkMatch pretrained models contain CNN layers, this script drops the CNN weights
# To call this script: python3 scripts/convert_thinkmatch_pretrained.py

import sys
sys.path.insert(0, '.')

import os
import pygmtools as pygm
import torch
pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)

######################################
# Modify here!
source_dir = 'your/thinkmatch/pretrain/dir'
target_dir = 'your/target/dir'

model_name = 'cie'
dataset_names = ['voc', 'willow']
######################################

source_files, target_files = [], []

for dname in dataset_names:
    source_files.append(f'pretrained_params_vgg16_{model_name}_{dname}.pt')
    target_files.append(f'{model_name}_{dname}_pytorch.pt')

# Generate a batch of isomorphic graphs
batch_size = 10
X_gt = torch.zeros(batch_size, 4, 4)
X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
A1 = 1. * (torch.rand(batch_size, 4, 4) > 0.5)
torch.diagonal(A1, dim1=1, dim2=2)[:] = 0
e_feat1 = (torch.rand(batch_size, 4, 4) * A1).unsqueeze(-1)
A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
e_feat2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), e_feat1.squeeze(-1)), X_gt).unsqueeze(-1)
feat1 = torch.rand(batch_size, 4, 1024) - 0.5
feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
n1 = n2 = torch.tensor([4] * batch_size)

conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)  # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

######################################
# Modify here!
_, network = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True, pretrain=False)
######################################

for src_f, tgt_f in zip(source_files, target_files):
    src_path = os.path.join(source_dir, src_f)
    tgt_path = os.path.join(target_dir, tgt_f)
    pygm.pytorch_backend._load_model(network, src_path, torch.device('cpu'), False)
    pygm.pytorch_backend._save_model(network, tgt_path)
