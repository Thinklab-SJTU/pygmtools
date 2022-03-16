import sys
sys.path.insert(0, '.')

import numpy as np
import pygmtools as pygm
import torch
import functools
import itertools
from tqdm import tqdm

# Some test utils functions
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


# The testing function
def _test_classic_solver_on_isomorphic_graphs(graph_num_nodes, node_feat_dim, solver_func, matrix_params, backends):
    assert 'edge_aff_fn' in matrix_params
    assert 'node_aff_fn' in matrix_params
    if backends[0] != 'pytorch':
        backends.insert(0, 'pytorch') # force pytorch as the reference backend

    batch_size = len(graph_num_nodes)

    # Generate isomorphic graphs
    pygm.BACKEND = 'pytorch'
    X_gt = torch.zeros(batch_size, max(graph_num_nodes), max(graph_num_nodes))
    A1 = torch.zeros(batch_size, max(graph_num_nodes), max(graph_num_nodes))
    A2 = torch.zeros(batch_size, max(graph_num_nodes), max(graph_num_nodes))
    F1 = torch.rand(batch_size, max(graph_num_nodes), node_feat_dim)
    F2 = torch.rand(batch_size, max(graph_num_nodes), node_feat_dim)
    for b, num_node in enumerate(graph_num_nodes):
        X_gt[b, torch.arange(0, num_node, dtype=torch.int64), torch.randperm(num_node)] = 1
        A1[b, :num_node, :num_node] = torch.rand(num_node, num_node)
        torch.diagonal(A1[b])[:] = 0
        A2[b] = torch.mm(torch.mm(X_gt[b].t(), A1[b]), X_gt[b])
        F2[b] = torch.mm(X_gt[b].t(), F1[b])
    n1 = torch.tensor(graph_num_nodes, dtype=torch.int)
    n2 = torch.tensor(graph_num_nodes, dtype=torch.int)

    A1, A2, F1, F2, n1, n2, X_gt = data_to_numpy(A1, A2, F1, F2, n1, n2, X_gt)

    # call the solver
    total = 1
    for val in matrix_params.values():
        total *= len(val)
    for values in tqdm(itertools.product(*matrix_params.values()), total=total):
        aff_param_dict = {}
        solver_param_dict = {}
        for k, v in zip(matrix_params.keys(), values):
            if k in ['node_aff_fn', 'edge_aff_fn']:
                aff_param_dict[k] = v
            else:
                solver_param_dict[k] = v

        last_K = None
        last_X = None
        for working_backend in backends:
            pygm.BACKEND = working_backend
            _A1, _A2, _F1, _F2, _n1, _n2 = data_from_numpy(A1, A2, F1, F2, n1, n2)
            _conn1, _edge1, _ne1 = pygm.utils.dense_to_sparse(_A1)
            _conn2, _edge2, _ne2 = pygm.utils.dense_to_sparse(_A2)

            _K = pygm.utils.build_aff_mat(_F1, _edge1, _conn1, _F2, _edge2, _conn2, _n1, _ne1, _n2, _ne2,
                                          **aff_param_dict)
            if last_K is not None:
                assert np.abs(pygm.utils.to_numpy(_K) - last_K).sum() < 0.1, \
                    f"Incorrect affinity matrix for {working_backend}, " \
                    f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                    f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
            last_K = pygm.utils.to_numpy(_K)
            _X = solver_func(_K, _n1, _n2, **solver_param_dict)
            if last_X is not None:
                assert np.abs(pygm.utils.to_numpy(_X) - last_X).sum() < 1e-4, \
                    f"Incorrect GM solution for {working_backend}, " \
                    f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                    f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
            last_X = pygm.utils.to_numpy(_X)

            accuracy = (pygm.utils.to_numpy(pygm.hungarian(_X, _n1, _n2)) * X_gt).sum() / X_gt.sum()
            assert accuracy == 1, f"GM is inaccurate for {working_backend}, " \
                                  f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                                  f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"


def test_rrwm():
    _test_classic_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 10, pygm.rrwm, {
        'alpha': [0.1, 0.5, 0.9],
        'beta': [0.1, 1, 10],
        'sk_iter': [10, 20],
        'max_iter': [20, 50],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, ['pytorch', 'numpy'])


def test_sm():
    _test_classic_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 10, pygm.sm, {
        'max_iter': [10, 50, 100],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, ['pytorch', 'numpy'])


def test_ipfp():
    _test_classic_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 10, pygm.ipfp, {
        'max_iter': [10, 50, 100],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, ['pytorch', 'numpy'])


if __name__ == '__main__':
    test_rrwm()
    test_sm()
    test_ipfp()
