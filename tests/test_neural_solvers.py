# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import sys

sys.path.insert(0, '.')

import numpy as np
import torch
import functools
import itertools
from tqdm import tqdm

from test_utils import *

import platform
os_name = platform.system()
backends = ['pytorch', 'numpy', 'jittor', 'paddle'] if os_name == 'Linux' else ['pytorch', 'numpy', 'paddle']


# The testing function for quadratic assignment
def _test_neural_solver_on_isomorphic_graphs(graph_num_nodes, node_feat_dim, solver_func, mode, matrix_params, backends):
    if mode == 'lawler-qap':
        assert 'edge_aff_fn' in matrix_params
        assert 'node_aff_fn' in matrix_params
    if backends[0] != 'pytorch':
        backends.insert(0, 'pytorch') # force pytorch as the reference backend

    batch_size = len(graph_num_nodes)

    # Generate isomorphic graphs
    pygm.BACKEND = 'pytorch'
    torch.manual_seed(0)
    X_gt, A1, A2, F1, F2, EF1, EF2 = [], [], [], [], [], [], []
    for b, num_node in enumerate(graph_num_nodes):
        As_b, X_gt_b, Fs_b = pygm.utils.generate_isomorphic_graphs(num_node, node_feat_dim=node_feat_dim)
        Fs_b = Fs_b - 0.5
        X_gt.append(X_gt_b)
        A1.append(As_b[0])
        A2.append(As_b[1])
        F1.append(Fs_b[0])
        F2.append(Fs_b[1])
        EF1.append((torch.rand(num_node, num_node) * As_b[0]).unsqueeze(-1) / 10)
        EF2.append(torch.mm(torch.mm(X_gt_b.t(), EF1[-1].squeeze(-1)), X_gt_b).unsqueeze(-1))
    n1 = torch.tensor(graph_num_nodes, dtype=torch.int)
    n2 = torch.tensor(graph_num_nodes, dtype=torch.int)
    A1, A2, F1, F2, EF1, EF2, X_gt = (pygm.utils.build_batch(_) for _ in (A1, A2, F1, F2, EF1, EF2, X_gt))
    if batch_size > 1:
        A1, A2, F1, F2, EF1, EF2, n1, n2, X_gt = data_to_numpy(A1, A2, F1, F2, EF1, EF2, n1, n2, X_gt)
    else:
        A1, A2, F1, F2, EF1, EF2, n1, n2, X_gt = data_to_numpy(
            A1.squeeze(0), A2.squeeze(0), F1.squeeze(0), F2.squeeze(0), EF1.squeeze(0), EF2.squeeze(0), n1, n2,
            X_gt.squeeze(0)
        )

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
            _A1, _A2, _F1, _F2, _EF1, _EF2, _n1, _n2 = data_from_numpy(A1, A2, F1, F2, EF1, EF2, n1, n2)
            if batch_size == 1:
                if mode == 'lawler-qap':
                    _n1, _n2 = _n1.item(), _n2.item()
                else:
                    _n1, _n2 = None, None

            if mode == 'lawler-qap':
                if batch_size > 1:
                    _conn1, _edge1, _ne1 = pygm.utils.dense_to_sparse(_A1)
                    _conn2, _edge2, _ne2 = pygm.utils.dense_to_sparse(_A2)
                    _K = pygm.utils.build_aff_mat(_F1, _edge1, _conn1, _F2, _edge2, _conn2, _n1, _ne1, _n2, _ne2,
                                                  **aff_param_dict)
                else:
                    _conn1, _edge1 = pygm.utils.dense_to_sparse(_A1)
                    _conn2, _edge2 = pygm.utils.dense_to_sparse(_A2)
                    _K = pygm.utils.build_aff_mat(_F1, _edge1, _conn1, _F2, _edge2, _conn2, _n1, None, _n2, None,
                                                  **aff_param_dict)
                if last_K is not None:
                    assert np.abs(pygm.utils.to_numpy(_K) - last_K).sum() < 0.1, \
                        f"Incorrect affinity matrix for {working_backend}, " \
                        f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                        f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                last_K = pygm.utils.to_numpy(_K)
                _X1, net = solver_func(_K, _n1, _n2, return_network=True, **solver_param_dict)
                _X2 = solver_func(_K, _n1, _n2, network=net, **solver_param_dict)
            elif mode == 'individual-graphs':
                _X1, net = solver_func(_F1, _F2, _A1, _A2, _n1, _n2, return_network=True, **solver_param_dict)
                _X2 = solver_func(_F1, _F2, _A1, _A2, _n1, _n2, network=net, **solver_param_dict)
            elif mode == 'individual-graphs-edge':
                _X1, net = solver_func(_F1, _F2, _A1, _A2, _EF1, _EF2, _n1, _n2, return_network=True, **solver_param_dict)
                _X2 = solver_func(_F1, _F2, _A1, _A2, _EF1, _EF2, _n1, _n2, network=net, **solver_param_dict)
            else:
                raise ValueError(f'Unknown mode: {mode}!')

            net2 = pygm.utils.get_network(solver_func, **solver_param_dict)
            assert type(net) == type(net2)

            assert np.abs(pygm.utils.to_numpy(_X1) - pygm.utils.to_numpy(_X2)).sum() < 1e-4, \
                f"GM result inconsistent for predefined network object. backend={working_backend}, " \
                f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"

            if 'pretrain' in solver_param_dict and solver_param_dict['pretrain'] is None:
                _X1 = pygm.hungarian(_X1, _n1, _n2)

            if last_X is not None:
                assert np.abs(pygm.utils.to_numpy(_X1) - last_X).sum() < 5e-3, \
                    f"Incorrect GM solution for {working_backend}, " \
                    f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                    f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
            last_X = pygm.utils.to_numpy(_X1)

            accuracy = (pygm.utils.to_numpy(pygm.hungarian(_X1, _n1, _n2)) * X_gt).sum() / X_gt.sum()
            assert accuracy == 1, f"GM is inaccurate for {working_backend}, accuracy={accuracy:.4f}, " \
                                  f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                                  f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"


# The testing function for genn_astar
def _test_genn_astar(graph_num_nodes, node_feat_dim, solver_func, matrix_params, backends):
    if backends[0] != 'pytorch':
        backends.insert(0, 'pytorch') # force pytorch as the reference backend
    backends = ['pytorch'] # Due to currently only supporting pytorch, testing is only conducted under pytorch
    batch_size = len(graph_num_nodes)
    
    # Generate isomorphic graphs
    pygm.BACKEND = 'pytorch'
    torch.manual_seed(0)
    X_gt, A1, A2, F1, F2, = [], [], [], [], [],
    for b, num_node in enumerate(graph_num_nodes):
        As_b, X_gt_b, Fs_b = pygm.utils.generate_isomorphic_graphs(num_node, node_feat_dim=node_feat_dim)
        Fs_b = Fs_b - 0.5
        X_gt.append(X_gt_b)
        A1.append(As_b[0])
        A2.append(As_b[1])
        F1.append(Fs_b[0])
        F2.append(Fs_b[1])
    n1 = torch.tensor(graph_num_nodes, dtype=torch.int)
    n2 = torch.tensor(graph_num_nodes, dtype=torch.int)
    A1, A2, F1, F2,  X_gt = (pygm.utils.build_batch(_) for _ in (A1, A2, F1, F2, X_gt))
    if batch_size > 1:
        A1, A2, F1, F2, n1, n2, X_gt = data_to_numpy(A1, A2, F1, F2, n1, n2, X_gt)
    else:
        A1, A2, F1, F2, n1, n2, X_gt = data_to_numpy(
            A1.squeeze(0), A2.squeeze(0), F1.squeeze(0), F2.squeeze(0),  n1, n2, X_gt.squeeze(0)
        )
    
    # call the solver
    total = 1
    for val in matrix_params.values():
        total *= len(val)
    for values in tqdm(itertools.product(*matrix_params.values()), total=total):
        solver_param_dict = {}
        for k, v in zip(matrix_params.keys(), values):
            solver_param_dict[k] = v
            
        last_X = None
        for working_backend in backends:
            pygm.BACKEND = working_backend
            _A1, _A2, _F1, _F2, _n1, _n2 = data_from_numpy(A1, A2, F1, F2, n1, n2)
            _X1, net = solver_func(_F1, _F2, _A1, _A2, _n1, _n2, return_network=True, **solver_param_dict)
            _X2 = solver_func(_F1, _F2, _A1, _A2, _n1, _n2, network=net, **solver_param_dict)
            net2 = pygm.utils.get_network(solver_func, **solver_param_dict)
            assert type(net) == type(net2)

            assert np.abs(pygm.utils.to_numpy(_X1) - pygm.utils.to_numpy(_X2)).sum() < 1e-4, \
                f"GM result inconsistent for predefined network object. backend={working_backend}; " \
                f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"

            if 'pretrain' in solver_param_dict and solver_param_dict['pretrain'] is None:
                _X1 = pygm.hungarian(_X1, _n1, _n2)

            if last_X is not None:
                assert np.abs(pygm.utils.to_numpy(_X1) - last_X).sum() < 5e-3, \
                    f"Incorrect GM solution for {working_backend}; " \
                    f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                    
            last_X = pygm.utils.to_numpy(_X1)
            accuracy = (pygm.utils.to_numpy(pygm.hungarian(_X1, _n1, _n2)) * X_gt).sum() / X_gt.sum()
            assert accuracy == 1, f"GM is inaccurate for {working_backend}, accuracy={accuracy:.4f}; " \
                                  f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                                  

def test_pca_gm():
    _test_neural_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 1024, pygm.pca_gm, 'individual-graphs', {
        'pretrain': ['voc', 'willow', 'voc-all'],
    }, backends)

    # non-batched input
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.pca_gm, 'individual-graphs', {
        'pretrain': ['voc'],
    }, backends)

    # test more layers
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.pca_gm, 'individual-graphs', {
        'num_layers': [3],
        'pretrain': [None],
    }, backends)


def test_ipca_gm():
    _test_neural_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 1024, pygm.ipca_gm, 'individual-graphs', {
        'pretrain': ['voc', 'willow'],
    }, backends)

    # non-batched input
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.ipca_gm, 'individual-graphs', {
        'pretrain': ['voc'],
    }, backends)

    # test more layers
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.ipca_gm, 'individual-graphs', {
        'num_layers': [3],
        'pretrain': [None],
    }, backends)


def test_cie():
    _test_neural_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 1024, pygm.cie, 'individual-graphs-edge', {
            'pretrain': ['voc', 'willow'],
        }, backends)

    # non-batched input
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.cie, 'individual-graphs-edge', {
        'pretrain': ['voc'],
    }, backends)

    # test more layers
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.cie, 'individual-graphs-edge', {
        'num_layers': [3],
        'pretrain': [None],
    }, backends)


def test_ngm():
    _test_neural_solver_on_isomorphic_graphs(list(range(10, 30, 2)), 1024, pygm.ngm, 'lawler-qap', {
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn],
        'pretrain': ['voc', 'willow'],
    }, backends)

    # non-batched input
    _test_neural_solver_on_isomorphic_graphs([10], 1024, pygm.ngm, 'lawler-qap', {
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)],
        'pretrain': ['voc'],
    }, backends)


def test_genn_astar():
    # test pretrained by AIDS700nef
    args1 = (list(range(10, 30, 2)), 36, pygm.genn_astar,{
        "pretrain": ["AIDS700nef"],
        "beam_width": [0, 1, 2],
        "trust_fact": [0.9, 0.95, 1.0],
        "no_pred_size": [0, 1],
    
    }, backends)
    
    # non-batched input
    args2 = ([10], 36, pygm.genn_astar,{
        'pretrain':  ["AIDS700nef"],
        "beam_width": [0, 1, 2],
        "trust_fact": [0.9, 0.95, 1.0],
        "no_pred_size": [0, 1],
    }, backends)
    
    # test pretrained by LINUX
    args3 = (list(range(10, 30, 2)), 8, pygm.genn_astar,{
        'pretrain':  ['LINUX'],
        "beam_width": [0, 1, 2],
        "trust_fact": [0.9, 0.95, 1.0],
        "no_pred_size": [0, 1],
    }, backends)
    
    # non-batched input
    args4 = ([10], 8, pygm.genn_astar,{
        'pretrain':  ['LINUX'],
        "beam_width": [0, 1, 2],
        "trust_fact": [0.9, 0.95, 1.0],
        "no_pred_size": [0, 1],
    }, backends)
    
    _test_genn_astar(*args1)
    _test_genn_astar(*args2)
    _test_genn_astar(*args3)
    _test_genn_astar(*args4)


if __name__ == '__main__':
    test_pca_gm()
    test_ipca_gm()
    test_cie()
    test_ngm()
    test_genn_astar()
