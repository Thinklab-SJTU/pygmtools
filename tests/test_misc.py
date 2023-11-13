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
import pygmtools as pygm
import itertools
import numpy as np

import platform
os_name = platform.system()
backends = ['pytorch', 'numpy', 'paddle', 'jittor', 'tensorflow'] if os_name == 'Linux' else ['pytorch', 'numpy', 'paddle', 'tensorflow']

def test_env_report():
    pygm.env_report()


def test_set_backend():
    for bk in backends:
        pygm.set_backend(bk)
    for bk in ['torch', 'tf', 'paddlepaddle', '0000']:
        try:
            pygm.set_backend(bk)
        except ValueError:
            pass


def test_generate_isomorphic_graphs():
    for backend in backends:
        pygm.BACKEND = backend
        A, X = pygm.utils.generate_isomorphic_graphs(10)
        A_shape, X_shape = pygm.utils._get_shape(A), pygm.utils._get_shape(X)
        assert A_shape[0] == 2 and A_shape[1] == 10 and A_shape[2] == 10
        assert X_shape[0] == 10 and X_shape[1] == 10

        A_np, X_np = pygm.utils.to_numpy(A), pygm.utils.to_numpy(X)
        assert np.all(np.matmul(np.matmul(X_np.transpose(), A_np[0]), X_np) == A_np[1])

        A, X, F = pygm.utils.generate_isomorphic_graphs(node_num=10, graph_num=5, node_feat_dim=20)
        A_shape, X_shape, F_shape = pygm.utils._get_shape(A), pygm.utils._get_shape(X), pygm.utils._get_shape(F)
        assert A_shape[0] == 5 and A_shape[1] == 10 and A_shape[2] == 10
        assert X_shape[0] == 5 and X_shape[1] == 5 and X_shape[2] == 10 and X_shape[3] == 10
        assert F_shape[0] == 5 and F_shape[1] == 10 and F_shape[2] == 20

        for i, j in itertools.product(range(5), repeat=2):
            Ai_np, Aj_np = pygm.utils.to_numpy(A[i]), pygm.utils.to_numpy(A[j])
            X_np = pygm.utils.to_numpy(X[i, j])
            assert np.all(np.matmul(np.matmul(X_np.transpose(), Ai_np), X_np) == Aj_np)
    try:
        pygm.utils.generate_isomorphic_graphs(10, backend='null')
    except NotImplementedError:
        pass


def test_permutation_loss():
    num_nodes = 10
    for backend in backends:
        if backend == 'numpy':
            continue
        pygm.BACKEND = backend
        A, X_gt = pygm.utils.generate_isomorphic_graphs(num_nodes)
        n1 = n2 = num_nodes
        conn1, edge1 = pygm.utils.dense_to_sparse(A[0])
        conn2, edge2 = pygm.utils.dense_to_sparse(A[1])
        K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2)
        S = pygm.rrwm(K, n1, n2)  # solving QAP
        X = pygm.hungarian(S)  # to discrete solution
        assert pygm.utils.permutation_loss(X, X_gt) == 0
        loss2 = pygm.utils.permutation_loss(
            pygm.utils.from_numpy(np.full_like(pygm.utils.to_numpy(X), 1 / num_nodes)), X_gt)
        assert (loss2 - 3.25083) < 1e-4

    try:
        pygm.utils.permutation_loss(X, X_gt, backend='null')
    except NotImplementedError:
        pass


def test_multi_matching_result():
    num_nodes = 10
    num_graphs = 5
    pygm.BACKEND = 'numpy'
    As, X = pygm.utils.generate_isomorphic_graphs(num_nodes, num_graphs)
    mmX = pygm.utils.MultiMatchingResult()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mmX[i, j] = X[i, j]

    for backend in backends:
        newX = pygm.utils.from_numpy(mmX, backend=backend)
        newX.__repr__()
        newX.__str__()

    try:
        pygm.utils.from_numpy(X, backend='null')
    except NotImplementedError:
        pass


if __name__ == '__main__':
    test_env_report()
    test_generate_isomorphic_graphs()
    test_permutation_loss()
    test_multi_matching_result()
