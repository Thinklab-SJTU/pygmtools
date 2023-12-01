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

import random

sys.path.insert(0, '.')

import numpy as np
import torch
import functools
import itertools
from tqdm import tqdm

from test_utils import *
import platform

os_name = platform.system()
backends = ['pytorch', 'numpy', 'paddle', 'jittor'] if os_name == 'Linux' else ['pytorch', 'numpy', 'paddle']
if os_name == 'Linux':
    import jittor as jt


# The testing function
def _test_mgm_solver_on_isomorphic_graphs(num_graph, num_node, node_feat_dim, solver_func, mode, matrix_params, backends):
    if mode == 'lawler-qap':
        assert 'edge_aff_fn' in matrix_params
    assert 'node_aff_fn' in matrix_params
    if backends[0] != 'pytorch':
        backends.insert(0, 'pytorch')  # force pytorch as the reference backend

    # Generate isomorphic graphs
    pygm.BACKEND = 'pytorch'
    # As, Fs are for kb-qap algorithms
    torch.manual_seed(1)
    As, X_gt, Fs = pygm.utils.generate_isomorphic_graphs(num_node, num_graph, node_feat_dim)
    # As_1, As_2, Fs_1, Fs_2 are for lawler-qap algorithms
    As_1, As_2, Fs_1, Fs_2 = [], [], [], []
    for i in range(num_graph):
        for j in range(num_graph):
            As_1.append(As[i])
            As_2.append(As[j])
            Fs_1.append(Fs[i])
            Fs_2.append(Fs[j])
    As_1 = torch.stack(As_1, dim=0)
    As_2 = torch.stack(As_2, dim=0)
    Fs_1 = torch.stack(Fs_1, dim=0)
    Fs_2 = torch.stack(Fs_2, dim=0)

    As, Fs, As_1, As_2, Fs_1, Fs_2, X_gt = data_to_numpy(As, Fs, As_1, As_2, Fs_1, Fs_2, X_gt)

    # call the solver
    total = 1
    for val in matrix_params.values():
        total *= len(val)
    for values in tqdm(itertools.product(*matrix_params.values()), total=total):
        aff_param_dict = {}  # for affinity functions (supported keys: 'node_aff_fn', 'edge_aff_fn')
        solver_param_dict = {}  # for solvers
        for k, v in zip(matrix_params.keys(), values):
            if k in ['node_aff_fn', 'edge_aff_fn']:
                aff_param_dict[k] = v
            else:
                if k == 'x0' and v is not None:
                    # (1-matrix_params['x0']) matchings are correct, the others are randomly permuted
                    x0_prob = v
                    x0 = []
                    for i, j in itertools.product(range(num_graph), repeat=2):
                        if i == j or random.random() > v:
                            x0.append(X_gt[i, j])
                        else:
                            _rand_perm = np.zeros((num_node, num_node), dtype=np.float32)
                            _rand_perm[np.arange(num_node), np.random.permutation(num_node)] = 1
                            x0.append(_rand_perm)
                    x0 = np.stack(x0)
                    v = x0.reshape((num_graph, num_graph, num_node, num_node))
                solver_param_dict[k] = v

        last_K = None
        last_X = None
        for working_backend in backends:
            pygm.BACKEND = working_backend
            if 'x0' in solver_param_dict and solver_param_dict['x0'] is not None:
                solver_param_dict['x0'] = pygm.utils.from_numpy(solver_param_dict['x0'])
            if 'ns' in solver_param_dict and solver_param_dict['ns'] is not None:
                solver_param_dict['ns'] = pygm.utils.from_numpy(solver_param_dict['ns'])
            if mode == 'lawler-qap':
                _As_1, _As_2, _Fs_1, _Fs_2, _X_gt = data_from_numpy(As_1, As_2, Fs_1, Fs_2, X_gt)
                _conn1, _edge1, _ne1 = pygm.utils.dense_to_sparse(_As_1)
                _conn2, _edge2, _ne2 = pygm.utils.dense_to_sparse(_As_2)

                _K = pygm.utils.build_aff_mat(_Fs_1, _edge1, _conn1, _Fs_2, _edge2, _conn2, None, _ne1, None, _ne2,
                                              **aff_param_dict)
                _K = _K.reshape((num_graph, num_graph, num_node ** 2, num_node ** 2))
                if last_K is not None:
                    assert np.abs(pygm.utils.to_numpy(_K) - last_K).sum() < 0.1, \
                        f"Incorrect affinity matrix for {working_backend}, " \
                        f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                        f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                last_K = pygm.utils.to_numpy(_K)
                _X = solver_func(_K, **solver_param_dict)
                if last_X is not None:
                    assert np.abs(pygm.utils.to_numpy(_X) - last_X).sum() < 1e-4, \
                        f"Incorrect GM solution for {working_backend}, " \
                        f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                        f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                last_X = pygm.utils.to_numpy(_X)

                accuracy = (pygm.utils.to_numpy(_X) * X_gt).sum() / X_gt.sum()
                if 'x0' not in solver_param_dict or solver_param_dict['x0'] is None:
                    assert accuracy == 1, f"GM is inaccurate for {working_backend}, accuracy={accuracy}, " \
                                          f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                                          f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                else:
                    assert accuracy >= 1 - x0_prob, f"GM is inaccurate for {working_backend}, accuracy={accuracy}, " \
                                                    f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                                                    f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
            elif mode == 'kb-qap':
                Fs1 = np.expand_dims(Fs, 1).repeat(num_graph, axis=1).reshape(num_graph ** 2, num_node, node_feat_dim)
                Fs2 = np.expand_dims(Fs, 0).repeat(num_graph, axis=0).reshape(num_graph ** 2, num_node, node_feat_dim)
                _As, _Fs1, _Fs2, _X_gt = data_from_numpy(As, Fs1, Fs2, X_gt)
                node_aff_mat = aff_param_dict['node_aff_fn'](_Fs1, _Fs2)
                node_aff_mat = node_aff_mat.reshape((num_graph, num_graph, num_node, num_node))
                _X = solver_func(_As, node_aff_mat, **solver_param_dict)

                if last_X is not None:
                    diff = 0
                    for i, j in itertools.product(range(num_graph), repeat=2):
                        diff += np.abs(pygm.utils.to_numpy(_X[i, j]) - last_X[i, j]).sum()
                    # solve again
                    if diff > 1e-4:
                        _X = solver_func(_As, node_aff_mat, **solver_param_dict)
                        diff = 0
                        for i, j in itertools.product(range(num_graph), repeat=2):
                            diff += np.abs(pygm.utils.to_numpy(_X[i, j]) - last_X[i, j]).sum()
                    assert diff < 1e-4, \
                        f"Incorrect GM solution for {working_backend}, " \
                        f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                        f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
                last_X = pygm.utils.to_numpy(_X)

                matched = 0
                total = 0
                for i, j in itertools.product(range(num_graph), repeat=2):
                    if 'ns' in solver_param_dict and solver_param_dict['ns'] is not None:
                        nsi = pygm.utils.to_numpy(solver_param_dict['ns'][i]).item()
                        nsj = pygm.utils.to_numpy(solver_param_dict['ns'][j]).item()
                        matched += (pygm.utils.to_numpy(_X[i, j]) * X_gt[i, j, :nsi, :nsj]).sum()
                        total += X_gt[i, j, :nsi, :nsj].sum()
                    else:
                        matched += (pygm.utils.to_numpy(_X[i, j]) * X_gt[i, j]).sum()
                        total += X_gt[i, j].sum()
                accuracy = matched / total
                assert accuracy == 1, f"GM is inaccurate for {working_backend}, accuracy={accuracy}, " \
                                      f"params: {';'.join([k + '=' + str(v) for k, v in aff_param_dict.items()])};" \
                                      f"{';'.join([k + '=' + str(v) for k, v in solver_param_dict.items()])}"
            else:
                raise ValueError(f'Unknown mode: {mode}')
            if 'x0' in solver_param_dict and solver_param_dict['x0'] is not None:
                solver_param_dict['x0'] = pygm.utils.to_numpy(solver_param_dict['x0'])
            if 'ns' in solver_param_dict and solver_param_dict['ns'] is not None:
                solver_param_dict['ns'] = pygm.utils.to_numpy(solver_param_dict['ns'])


def test_cao():
    num_nodes = 5
    num_graphs = 10
    _test_mgm_solver_on_isomorphic_graphs(num_graphs, num_nodes, 10, pygm.cao, 'lawler-qap', {
        'mode': ['time', 'memory'],
        'x0': [None, 0.2, 0.5],
        'lambda_init': [0.1, 0.3],
        'qap_solver': [functools.partial(pygm.ipfp, n1max=num_nodes, n2max=num_nodes), None],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, backends)


def test_mgm_floyd():
    num_nodes = 5
    num_graphs = 10
    _test_mgm_solver_on_isomorphic_graphs(num_graphs, num_nodes, 10, pygm.mgm_floyd, 'lawler-qap', {
        'mode': ['time', 'memory'],
        'x0': [None, 0.2, 0.5],
        'param_lambda': [0.1, 0.3],
        'qap_solver': [functools.partial(pygm.ipfp, n1max=num_nodes, n2max=num_nodes), None],
        'edge_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.), pygm.utils.inner_prod_aff_fn],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn]
    }, backends)


def test_gamgm():
    num_nodes = 5
    num_graphs = 10
    # test without outliers
    _test_mgm_solver_on_isomorphic_graphs(num_graphs, num_nodes, 10, pygm.gamgm, 'kb-qap', {
        'sk_init_tau': [0.5, 0.1],
        'sk_min_tau': [0.1, 0.05],
        'param_lambda': [0.1, 0.5],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1), pygm.utils.inner_prod_aff_fn],
        'verbose': [True]
    }, backends)

    # test with outliers
    _test_mgm_solver_on_isomorphic_graphs(num_graphs, num_nodes, 10, pygm.gamgm, 'kb-qap', {
        'sk_init_tau': [0.5],
        'sk_gamma': [0.8],
        'sk_min_tau': [0.1],
        'param_lambda': [0.],
        'node_aff_fn': [functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)],
        'verbose': [True],
        'n_univ': [10],
        'outlier_thresh': [0., 0.1],
        'ns': [np.array([num_nodes] * (num_graphs // 2) + [num_nodes - 1] * (num_graphs - num_graphs // 2))],
    }, backends)


def test_gamgm_backward():
    # Pytorch
    pygm.BACKEND = 'pytorch'
    torch.manual_seed(1)

    # Generate 10 isomorphic graphs
    graph_num = 10
    As, X_gt, Fs = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=10, node_feat_dim=20)

    # Compute node-wise similarity by inner-product and Sinkhorn
    W = torch.matmul(Fs.unsqueeze(1), Fs.transpose(1, 2).unsqueeze(0))
    W = pygm.sinkhorn(W.reshape(graph_num ** 2, 4, 4)).reshape(graph_num, graph_num, 4, 4)

    # This function is differentiable by the black-box trick
    W.requires_grad_(True)  # tell PyTorch to track the gradients
    X = pygm.gamgm(As, W)
    loss = 0 # a random loss function
    for i, j in itertools.product(range(graph_num), repeat=2):
        loss += (X[i, j] * torch.rand_like(X[i, j])).sum()

    # Backward pass via black-box trick
    loss.backward()
    assert torch.sum(W.grad != 0) > 0

    # Jittor
    if os_name == 'Linux':
        pygm.BACKEND = 'jittor'
        jt.set_global_seed(2)

        # Generate 10 isomorphic graphs
        graph_num = 10
        As, X_gt, Fs = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=10, node_feat_dim=20)

        # Compute node-wise similarity by inner-product and Sinkhorn
        W = jt.matmul(Fs.unsqueeze(1), Fs.transpose(1, 2).unsqueeze(0))
        W = pygm.sinkhorn(W.reshape(graph_num ** 2, 4, 4)).reshape(graph_num, graph_num, 4, 4)

        # This function is differentiable by the black-box trick
        class Model(jt.nn.Module):
            def __init__(self, W):
                self.W = W

            def execute(self, As):
                X = pygm.gamgm(As, self.W)
                return X

        W.start_grad()
        model = Model(W)
        X = model(As)
        loss = 0
        for i, j in itertools.product(range(graph_num), repeat=2):
            loss += (X[i, j] * jt.rand_like(X[i, j])).sum()

        # Backward pass via black-box trick
        optim = jt.nn.SGD(model.parameters(), lr=0.1)
        optim.step(loss)
        grad = W.opt_grad(optim)
        print(jt.sum(grad != 0))
        assert jt.sum(grad != 0) > 0


if __name__ == '__main__':
    test_gamgm_backward()
    test_gamgm()
    test_mgm_floyd()
    test_cao()
