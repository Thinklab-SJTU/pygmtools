import sys

sys.path.insert(0, '.')

import numpy as np
import torch
import functools
import itertools
from tqdm import tqdm

from test_utils import *


# The testing function for quadratic assignment
def _test_classic_solver_on_isomorphic_graphs(graph_num_nodes, node_feat_dim, solver_func, matrix_params, backends):
    assert 'edge_aff_fn' in matrix_params
    assert 'node_aff_fn' in matrix_params
    if backends[0] != 'pytorch':
        backends.insert(0, 'pytorch')  # force pytorch as the reference backend

    batch_size = len(graph_num_nodes)

    # Generate isomorphic graphs
    pygm.BACKEND = 'pytorch'
    X_gt, A1, A2, F1, F2 = [], [], [], [], []
    for b, num_node in enumerate(graph_num_nodes):
        As_b, X_gt_b, Fs_b = pygm.utils.generate_isomorphic_graphs(num_node, node_feat_dim=node_feat_dim)
        X_gt.append(X_gt_b)
        A1.append(As_b[0])
        A2.append(As_b[1])
        F1.append(Fs_b[0])
        F2.append(Fs_b[1])
    n1 = torch.tensor(graph_num_nodes, dtype=torch.int)
    n2 = torch.tensor(graph_num_nodes, dtype=torch.int)
    A1, A2, F1, F2, X_gt = (pygm.utils.build_batch(_) for _ in (A1, A2, F1, F2, X_gt))
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
            assert accuracy == 1, f"GM is inaccurate for {working_backend}, accuracy={accuracy:.4f}, " \
                                  f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                                  f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"


# The testing function for linear assignment
def _test_classic_solver_on_linear_assignment(num_nodes1, num_nodes2, node_feat_dim, solver_func, matrix_params,
                                              backends):
    batch_size = len(num_nodes1)

    # iterate over matrix parameters
    total = 1
    for val in matrix_params.values():
        total *= len(val)
    for values in tqdm(itertools.product(*matrix_params.values()), total=total):
        prob_param_dict = {}
        solver_param_dict = {}
        for k, v in zip(matrix_params.keys(), values):
            if k in ['outlier_num', 'unmatch']:
                prob_param_dict[k] = v
            else:
                solver_param_dict[k] = v
        unmatch = prob_param_dict['outlier_num'] > 0 if 'outlier_num' in prob_param_dict else False

        # Generate random node features
        pygm.BACKEND = 'pytorch'
        torch.manual_seed(3)
        X_gt, F1, F2, unmatch1, unmatch2 = [], [], [], [], []
        for b, (num_node1, num_node2) in enumerate(zip(num_nodes1, num_nodes2)):
            outlier_num = prob_param_dict['outlier_num'] if 'outlier_num' in prob_param_dict else 0
            max_inlier_index = max(num_node1, num_node2)
            As_b, X_gt_b, Fs_b = pygm.utils.generate_isomorphic_graphs(max_inlier_index + outlier_num * 2,
                                                                       node_feat_dim=node_feat_dim)
            Fs_b = Fs_b / torch.norm(Fs_b, dim=-1, p='fro', keepdim=True)  # normalize features
            outlier_indices_1 = list(range(max_inlier_index, max_inlier_index + outlier_num))
            outlier_indices_2 = list(range(max_inlier_index + outlier_num, max_inlier_index + outlier_num * 2))
            idx1 = list(set(list(range(num_node1)) + outlier_indices_1))
            idx2 = list(set(list(range(num_node2)) + outlier_indices_2))
            idx2 = X_gt_b.nonzero(as_tuple=False)[:, 1][idx2].numpy().tolist()  # permute idx2 according to X_gt_b
            idx2.sort()
            F1.append(Fs_b[0][idx1])
            F2.append(Fs_b[1][idx2])
            X_gt.append(X_gt_b[idx1, :][:, idx2])
            if unmatch:
                unmatch1.append(torch.ones(num_node1 + outlier_num) * 0.49)
                unmatch2.append(torch.ones(num_node2 + outlier_num) * 0.49)
        n1 = torch.tensor(num_nodes1, dtype=torch.int) + outlier_num
        n2 = torch.tensor(num_nodes2, dtype=torch.int) + outlier_num
        F1, F2, X_gt = (pygm.utils.build_batch(_) for _ in (F1, F2, X_gt))
        F1, F2, n1, n2, X_gt = data_to_numpy(F1, F2, n1, n2, X_gt)
        if unmatch:
            unmatch1, unmatch2 = (pygm.utils.build_batch(_) for _ in (unmatch1, unmatch2))
            unmatch1, unmatch2 = data_to_numpy(unmatch1, unmatch2)

        last_X = None
        for working_backend in backends:
            pygm.BACKEND = working_backend
            _F1, _F2, _n1, _n2 = data_from_numpy(F1, F2, n1, n2)

            linear_sim = []
            for b in range(batch_size):
                linear_sim.append(pygm.utils._mm(_F1[b], pygm.utils._transpose(_F2[b], 0, 1)))
            linear_sim = pygm.utils.build_batch(linear_sim)

            # call the solver
            if unmatch:
                _unmatch1, _unmatch2 = data_from_numpy(unmatch1, unmatch2)
                _X = solver_func(linear_sim, _n1, _n2, _unmatch1, _unmatch2, **solver_param_dict)

                # get the corresponding hungarian solution
                _X_np = pygm.utils.to_numpy(_X)
                X_hung = pygm.utils.to_numpy(pygm.hungarian(_X, _n1, _n2,
                                                            pygm.utils.from_numpy(1 - _X_np.sum(-1)) * 0.5,
                                                            pygm.utils.from_numpy(1 - _X_np.sum(-2)) * 0.5))
                accuracy = (X_hung * X_gt).sum() / max(X_hung.sum(), X_gt.sum())
            else:
                _X = solver_func(linear_sim, _n1, _n2, **solver_param_dict)
                accuracy = (pygm.utils.to_numpy(pygm.hungarian(_X, _n1, _n2)) * X_gt).sum() / X_gt.sum()

            assert accuracy == 1, f"GM is inaccurate for {working_backend}, accuracy={accuracy:.4f}, " \
                                  f"params: {';'.join([k + '=' + str(v) for k, v in prob_param_dict.items()])};" \
                                  f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"

            if last_X is not None:
                assert np.abs(pygm.utils.to_numpy(_X) - last_X).sum() < 5e-4, \
                    f"Incorrect GM solution for {working_backend}\n" \
                    f"params: {';'.join([k + '=' + str(v) for k, v in prob_param_dict.items()])}\n" \
                    f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
            last_X = pygm.utils.to_numpy(_X)


def test_hungarian():
    _test_classic_solver_on_linear_assignment(list(range(10, 30, 2)), list(range(30, 10, -2)), 10, pygm.hungarian, {
        'nproc': [1, 2, 4],
        'outlier_num': [0, 5, 10]
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])


def test_sinkhorn():
    # test non-symmetric matching
    args1 = (list(range(10, 30, 2)), list(range(30, 10, -2)), 10, pygm.sinkhorn, {
        'tau': [0.1, 0.01],
        'max_iter': [10, 20, 50],
        'batched_operation': [True, False],
        'dummy_row': [True, ],
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])

    # test symmetric matching
    args2 = (list(range(10, 30, 2)), list(range(10, 30, 2)), 10, pygm.sinkhorn, {
        'tau': [0.1, 0.01],
        'max_iter': [10, 20, 50],
        'batched_operation': [True, False],
        'dummy_row': [True, False],
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])

    # test outlier matching (non-symmetric)
    args3 = (list(range(10, 30, 2)), list(range(30, 10, -2)), 10, pygm.sinkhorn, {
        'tau': [0.01, 0.001],
        'max_iter': [500, 1000],
        'batched_operation': [True, False],
        'dummy_row': [True, False],
        'outlier_num': [5, 10]
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])

    # test outlier matching (symmetric)
    args4 = (list(range(10, 30, 2)), list(range(10, 30, 2)), 10, pygm.sinkhorn, {
        'tau': [0.01, 0.001],
        'max_iter': [500, 1000],
        'batched_operation': [True, False],
        'dummy_row': [True, False],
        'outlier_num': [5, 10]
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])

    _test_classic_solver_on_linear_assignment(*args1)
    _test_classic_solver_on_linear_assignment(*args2)
    _test_classic_solver_on_linear_assignment(*args3)
    _test_classic_solver_on_linear_assignment(*args4)


def test_rrwm():
    _test_classic_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 10, pygm.rrwm, {
        'alpha': [0.1, 0.5, 0.9],
        'beta': [0.1, 1, 10],
        'sk_iter': [10, 20],
        'max_iter': [20, 50],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])


def test_sm():
    _test_classic_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 10, pygm.sm, {
        'max_iter': [10, 50, 100],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])


def test_ipfp():
    _test_classic_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 10, pygm.ipfp, {
        'max_iter': [10, 50, 100],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow', 'mindspore'])


if __name__ == '__main__':
    test_hungarian()
    test_sinkhorn()
    test_rrwm()
    test_sm()
    test_ipfp()
