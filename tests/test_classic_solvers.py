import sys
sys.path.insert(0, '.')

import numpy as np
import pygmtools as pygm
import torch
import functools
import itertools


graph_num_nodes = list(range(10, 30, 2))
node_feat_dim = 10
batch_size = len(graph_num_nodes)

backends = ['pytorch', 'numpy']

# Test matching of isomorphic graphs
pygm.BACKEND = 'pytorch'
X_gt = torch.zeros(batch_size, max(graph_num_nodes), max(graph_num_nodes))
A1 = torch.zeros(batch_size, max(graph_num_nodes), max(graph_num_nodes))
A2 = torch.zeros(batch_size, max(graph_num_nodes), max(graph_num_nodes))
F1 = torch.rand(batch_size, max(graph_num_nodes), node_feat_dim)
F2 = torch.rand(batch_size, max(graph_num_nodes), node_feat_dim)
for b, num_node in enumerate(graph_num_nodes):
    X_gt[b, torch.arange(0, num_node, dtype=torch.int64), torch.randperm(num_node)] = 1
    A1[b, :num_node, :num_node] = torch.rand(num_node, num_node)
    A1_diagonal = torch.diagonal(A1[b])
    A1_diagonal[:] = 0
    A2[b] = torch.mm(torch.mm(X_gt[b].t(), A1[b]), X_gt[b])
    F2[b] = torch.mm(X_gt[b], F1[b])
n1 = torch.tensor(graph_num_nodes, dtype=torch.int)
n2 = torch.tensor(graph_num_nodes, dtype=torch.int)

def data_from_numpy(*data):
    return_list = []
    for d in data:
        return_list.append(pygm.utils.from_numpy(d))
    return return_list

def data_to_numpy(*data):
    return_list = []
    for d in data:
        return_list.append(pygm.utils.to_numpy(d))
    return return_list

A1, A2, F1, F2, n1, n2 = data_to_numpy(A1, A2, F1, F2, n1, n2)

# RRWM
for alpha, beta, edge_aff_func, node_aff_func in \
    itertools.product(
        [0.1, 0.5, 0.9], # alpha
        [0.1, 1, 10], # beta
        [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn], # edge aff func
        [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn], # node aff func
    ):
    last_K = None
    last_X = None
    for working_backend in backends:
        pygm.BACKEND = working_backend
        _A1, _A2, _F1, _F2, _n1, _n2 = data_from_numpy(A1, A2, F1, F2, n1, n2)
        _conn1, _edge1, _ne1 = pygm.utils.dense_to_sparse(_A1)
        _conn2, _edge2, _ne2 = pygm.utils.dense_to_sparse(_A2)

        _K = pygm.utils.build_aff_mat(_F1, _edge1, _conn1, _F2, _edge2, _conn2, _n1, _ne1, _n2, _ne2, node_aff_func, edge_aff_func)
        if last_K is not None:
            print(np.abs(pygm.utils.to_numpy(_K) - last_K).sum())
            assert np.abs(pygm.utils.to_numpy(_K) - last_K).sum() < 0.1
        last_K = pygm.utils.to_numpy(_K)
        _X = pygm.rrwm(_K, _n1, _n2, alpha=alpha, beta=beta)
        if last_X is not None:
            print(np.abs(pygm.utils.to_numpy(_X) - last_X).sum())
            assert np.abs(pygm.utils.to_numpy(_X) - last_X).sum() < 1e-4
        last_X = pygm.utils.to_numpy(_X)

        pygm.hungarian(_X, _n1, _n2)
