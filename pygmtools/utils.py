"""
Utility functions: problem formulating, data processing, and beyond.
"""

# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import time
import functools
import importlib
import copy
import requests
import os
from appdirs import user_cache_dir
import hashlib
import shutil
from tqdm.auto import tqdm
import inspect
import wget
import numpy as np
import pygmtools
import networkx as nx
import urllib.request

NOT_IMPLEMENTED_MSG = \
    'The backend function for {} is not implemented. ' \
    'If you are a user, please check the spelling, and use other backends as workarounds. ' \
    'If you are a developer, it will be truly appreciated if you could develop and share your' \
    ' implementation with the community! See our Github: https://github.com/Thinklab-SJTU/pygmtools'


def build_aff_mat(node_feat1, edge_feat1, connectivity1, node_feat2, edge_feat2, connectivity2,
                  n1=None, ne1=None, n2=None, ne2=None,
                  node_aff_fn=None, edge_aff_fn=None,
                  backend=None):
    r"""
    Build affinity matrix for graph matching from input node/edge features. The affinity matrix encodes both node-wise
    and edge-wise affinities and formulates the Quadratic Assignment Problem (QAP), which is the mathematical form of
    graph matching.

    :param node_feat1: :math:`(b\times n_1 \times f_{node})` the node feature of graph1
    :param edge_feat1: :math:`(b\times ne_1 \times f_{edge})` the edge feature of graph1
    :param connectivity1: :math:`(b\times ne_1 \times 2)` sparse connectivity information of graph 1.
                          ``connectivity1[i, j, 0]`` is the starting node index of edge ``j`` at batch ``i``, and
                          ``connectivity1[i, j, 1]`` is the ending node index of edge ``j`` at batch ``i``
    :param node_feat2: :math:`(b\times n_2 \times f_{node})` the node feature of graph2
    :param edge_feat2: :math:`(b\times ne_2 \times f_{edge})` the edge feature of graph2
    :param connectivity2: :math:`(b\times ne_2 \times 2)` sparse connectivity information of graph 2.
                          ``connectivity2[i, j, 0]`` is the starting node index of edge ``j`` at batch ``i``, and
                          ``connectivity2[i, j, 1]`` is the ending node index of edge ``j`` at batch ``i``
    :param n1: :math:`(b)` number of nodes in graph1. If not given, it will be inferred based on the shape of
               ``node_feat1`` or the values in ``connectivity1``
    :param ne1: :math:`(b)` number of edges in graph1. If not given, it will be inferred based on the shape of
               ``edge_feat1``
    :param n2: :math:`(b)` number of nodes in graph2. If not given, it will be inferred based on the shape of
               ``node_feat2`` or the values in ``connectivity2``
    :param ne2: :math:`(b)` number of edges in graph2. If not given, it will be inferred based on the shape of
               ``edge_feat2``
    :param node_aff_fn: (default: inner_prod_aff_fn) the node affinity function with the characteristic
                        ``node_aff_fn(2D Tensor, 2D Tensor) -> 2D Tensor``, which accepts two node feature tensors and
                        outputs the node-wise affinity tensor. See :func:`~pygmtools.utils.inner_prod_aff_fn` as an
                        example.
    :param edge_aff_fn: (default: inner_prod_aff_fn) the edge affinity function with the characteristic
                        ``edge_aff_fn(2D Tensor, 2D Tensor) -> 2D Tensor``, which accepts two edge feature tensors and
                        outputs the edge-wise affinity tensor. See :func:`~pygmtools.utils.inner_prod_aff_fn` as an
                        example.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1n_2 \times n_1n_2)` the affinity matrix

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # Generate a batch of graphs
            >>> batch_size = 10
            >>> A1 = np.random.rand(batch_size, 4, 4)
            >>> A2 = np.random.rand(batch_size, 4, 4)
            >>> n1 = n2 = np.repeat([4], batch_size)

            # Build affinity matrix by the default inner-product function
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2)

            # Build affinity matrix by gaussian kernel
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
            >>> K2 = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # Build affinity matrix based on node features
            >>> F1 = np.random.rand(batch_size, 4, 10)
            >>> F2 = np.random.rand(batch_size, 4, 10)
            >>> K3 = pygm.utils.build_aff_mat(F1, edge1, conn1, F2, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # The affinity matrices K, K2, K3 can be further processed by GM solvers

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'

            # Generate a batch of graphs
            >>> batch_size = 10
            >>> A1 = torch.rand(batch_size, 4, 4)
            >>> A2 = torch.rand(batch_size, 4, 4)
            >>> n1 = n2 = torch.tensor([4] * batch_size)

            # Build affinity matrix by the default inner-product function
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2)

            # Build affinity matrix by gaussian kernel
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
            >>> K2 = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # Build affinity matrix based on node features
            >>> F1 = torch.rand(batch_size, 4, 10)
            >>> F2 = torch.rand(batch_size, 4, 10)
            >>> K3 = pygm.utils.build_aff_mat(F1, edge1, conn1, F2, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # The affinity matrices K, K2, K3 can be further processed by GM solvers
    
    .. dropdown:: Paddle Example

        ::
            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'

            # Generate a batch of graphs
            >>> batch_size = 10
            >>> A1 = paddle.rand((batch_size, 4, 4))
            >>> A2 = paddle.rand((batch_size, 4, 4))
            >>> n1 = n2 = paddle.t0_tensor([4] * batch_size)

            # Build affinity matrix by the default inner-product function
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2)

            # Build affinity matrix by gaussian kernel
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
            >>> K2 = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # Build affinity matrix based on node features
            >>> F1 = paddle.rand((batch_size, 4, 10))
            >>> F2 = paddle.rand((batch_size, 4, 10))
            >>> K3 = pygm.utils.build_aff_mat(F1, edge1, conn1, F2, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # The affinity matrices K, K2, K3 can be further processed by GM solvers

    .. dropdown:: mindspore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'mindspore'

            # Generate a batch of graphs
            >>> batch_size = 10
            >>> A1 = mindspore.numpy.rand((batch_size, 4, 4))
            >>> A2 = mindspore.numpy.rand((batch_size, 4, 4))
            >>> n1 = n2 = mindspore.Tensor([4] * batch_size)

            # Build affinity matrix by the default inner-product function
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2)

            # Build affinity matrix by gaussian kernel
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
            >>> K2 = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # Build affinity matrix based on node features
            >>> F1 = mindspore.numpy.rand((batch_size, 4, 10))
            >>> F2 = mindspore.numpy.rand((batch_size, 4, 10))
            >>> K3 = pygm.utils.build_aff_mat(F1, edge1, conn1, F2, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

            # The affinity matrices K, K2, K3 can be further processed by GM solvers

    """
    if backend is None:
        backend = pygmtools.BACKEND
    __get_shape = functools.partial(_get_shape, backend=backend)

    # check the correctness of input
    batch_size = None
    non_batched_input = False
    if node_feat1 is not None or node_feat2 is not None:
        assert all([_ is not None for _ in (node_feat1, node_feat2)]), \
            'The following arguments must all be given if you want to compute node-wise affinity: ' \
            'node_feat1, node_feat2'
        _check_data_type(node_feat1, backend)
        _check_data_type(node_feat2, backend)
        if all([_check_shape(_, 2, backend) for _ in (node_feat1, node_feat2)]):
            non_batched_input = True
            node_feat1, node_feat2 = [_unsqueeze(_, 0, backend) for _ in (node_feat1, node_feat2)]
            if type(n1) is int: n1 = from_numpy(np.array([n1]), backend=backend)
            if type(n2) is int: n2 = from_numpy(np.array([n2]), backend=backend)
        elif all([_check_shape(_, 3, backend) for _ in (node_feat1, node_feat2)]):
            pass
        else:
            raise ValueError(
                f'The shape of the following tensors are illegal, expected 3-dimensional, '
                f'got node_feat1={len(__get_shape(node_feat1))}d; node_feat2={len(__get_shape(node_feat2))}d!'
            )
        if batch_size is None:
            batch_size = __get_shape(node_feat1)[0]
        assert __get_shape(node_feat1)[0] == __get_shape(node_feat2)[0] == batch_size, 'batch size mismatch'
    if edge_feat1 is not None or edge_feat2 is not None:
        assert all([_ is not None for _ in (edge_feat1, edge_feat2, connectivity1, connectivity2)]), \
            'The following arguments must all be given if you want to compute edge-wise affinity: ' \
            'edge_feat1, edge_feat2, connectivity1, connectivity2'
        if all([_check_shape(_, 2, backend) for _ in (edge_feat1, edge_feat2, connectivity1, connectivity2)]):
            non_batched_input = True
            edge_feat1, edge_feat2, connectivity1, connectivity2 = \
                [_unsqueeze(_, 0, backend) for _ in (edge_feat1, edge_feat2, connectivity1, connectivity2)]
            if type(ne1) is int: ne1 = from_numpy(np.array([ne1]), backend=backend)
            if type(ne2) is int: ne2 = from_numpy(np.array([ne2]), backend=backend)
        elif all([_check_shape(_, 3, backend) for _ in (edge_feat1, edge_feat2, connectivity1, connectivity2)]):
            pass
        else:
            raise ValueError(
                f'The shape of the following tensors are illegal, expected 3-dimensional, '
                f'got edge_feat1:{len(__get_shape(edge_feat1))}d; edge_feat2:{len(__get_shape(edge_feat2))}d; '
                f'connectivity1:{len(__get_shape(connectivity1))}d; connectivity2:{len(__get_shape(connectivity2))}d!'
            )
        assert __get_shape(connectivity1)[2] == __get_shape(connectivity1)[2] == 2, \
            'the last dimension of connectivity1, connectivity2 must be 2-dimensional'
        if batch_size is None:
            batch_size = __get_shape(edge_feat1)[0]
        assert __get_shape(edge_feat1)[0] == __get_shape(edge_feat2)[0] == __get_shape(connectivity1)[0] == \
               __get_shape(connectivity2)[0] == batch_size, 'batch size mismatch'

    # assign the default affinity functions if not given
    if node_aff_fn is None:
        node_aff_fn = functools.partial(inner_prod_aff_fn, backend=backend)
    if edge_aff_fn is None:
        edge_aff_fn = functools.partial(inner_prod_aff_fn, backend=backend)

    node_aff = node_aff_fn(node_feat1, node_feat2) if node_feat1 is not None else None
    edge_aff = edge_aff_fn(edge_feat1, edge_feat2) if edge_feat1 is not None else None

    result = _aff_mat_from_node_edge_aff(node_aff, edge_aff, connectivity1, connectivity2, n1, n2, ne1, ne2,
                                         backend=backend)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def inner_prod_aff_fn(feat1, feat2, backend=None):
    r"""
    Inner product affinity function. The affinity is defined as

    .. math::
        \mathbf{f}_1^\top \cdot \mathbf{f}_2

    :param feat1: :math:`(b\times n_1 \times f)` the feature vectors :math:`\mathbf{f}_1`
    :param feat2: :math:`(b\times n_2 \times f)` the feature vectors :math:`\mathbf{f}_2`
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1\times n_2)` element-wise inner product affinity matrix
    """
    if backend is None:
        backend = pygmtools.BACKEND

    _check_data_type(feat1, backend)
    _check_data_type(feat2, backend)
    args = (feat1, feat2)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.inner_prod_aff_fn
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def gaussian_aff_fn(feat1, feat2, sigma=1., backend=None):
    r"""
    Gaussian kernel affinity function. The affinity is defined as

    .. math::
        \exp(-\frac{(\mathbf{f}_1 - \mathbf{f}_2)^2}{\sigma})

    :param feat1: :math:`(b\times n_1 \times f)` the feature vectors :math:`\mathbf{f}_1`
    :param feat2: :math:`(b\times n_2 \times f)` the feature vectors :math:`\mathbf{f}_2`
    :param sigma: (default: 1) the parameter :math:`\sigma` in Gaussian kernel
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return:  :math:`(b\times n_1\times n_2)` element-wise Gaussian affinity matrix
    """
    if backend is None:
        backend = pygmtools.BACKEND

    _check_data_type(feat1, backend)
    _check_data_type(feat2, backend)
    args = (feat1, feat2, sigma)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.gaussian_aff_fn
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def build_batch(input, return_ori_dim=False, backend=None):
    r"""
    Build a batched tensor from a list of tensors. If the list of tensors are with different sizes of dimensions, it
    will be padded to the largest dimension.

    The batched tensor and the number of original dimensions will be returned.

    :param input: list of input tensors
    :param return_ori_dim: (default: False) return the original dimension
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: batched tensor, (if ``return_ori_dim=True``) a list of the original dimensions

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # batched adjacency matrices
            >>> A1 = np.random.rand(4, 4)
            >>> A2 = np.random.rand(5, 5)
            >>> A3 = np.random.rand(3, 3)
            >>> batched_A, ori_shape = pygm.utils.build_batch([A1, A2, A3], return_ori_dim=True)
            >>> batched_A.shape
            (3, 5, 5)
            >>> ori_shape
            ([4, 5, 3], [4, 5, 3])

            # batched node features (feature dimension=10)
            >>> F1 = np.random.rand(4, 10)
            >>> F2 = np.random.rand(5, 10)
            >>> F3 = np.random.rand(3, 10)
            >>> batched_F = pygm.utils.build_batch([F1, F2, F3])
            >>> batched_F.shape
            (3, 5, 10)

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'

            # batched adjacency matrices
            >>> A1 = torch.rand(4, 4)
            >>> A2 = torch.rand(5, 5)
            >>> A3 = torch.rand(3, 3)
            >>> batched_A, ori_shape = pygm.utils.build_batch([A1, A2, A3], return_ori_dim=True)
            >>> batched_A.shape
            torch.Size([3, 5, 5])
            >>> ori_shape
            (tensor([4, 5, 3]), tensor([4, 5, 3]))

            # batched node features (feature dimension=10)
            >>> F1 = torch.rand(4, 10)
            >>> F2 = torch.rand(5, 10)
            >>> F3 = torch.rand(3, 10)
            >>> batched_F = pygm.utils.build_batch([F1, F2, F3])
            >>> batched_F.shape
            torch.Size([3, 5, 10])

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'

            # batched adjacency matrices
            >>> A1 = paddle.rand((4, 4))
            >>> A2 = paddle.rand((5, 5))
            >>> A3 = paddle.rand((3, 3))
            >>> batched_A, ori_shape = pygm.utils.build_batch([A1, A2, A3], return_ori_dim=True)
            >>> batched_A.shape
            [3, 5, 5]
            >>> ori_shape
            (Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [4, 5, 3]),
             Tensor(shape=[3], dtype=int64, place=Place(cpu), stop_gradient=True, [4, 5, 3]))

            # batched node features (feature dimension=10)
            >>> F1 = paddle.rand((4, 10))
            >>> F2 = paddle.rand((5, 10))
            >>> F3 = paddle.rand((3, 10))
            >>> batched_F = pygm.utils.build_batch([F1, F2, F3])
            >>> batched_F.shape
            [3, 5, 10]

    .. dropdown:: mindspore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'mindspore'

            # batched adjacency matrices
            >>> A1 = mindspore.numpy.rand((4, 4))
            >>> A2 = mindspore.numpy.rand((5, 5))
            >>> A3 = mindspore.numpy.rand((3, 3))
            >>> batched_A, ori_shape = pygm.utils.build_batch([A1, A2, A3], return_ori_dim=True)
            >>> batched_A.shape
            (3, 5, 5)
            >>> ori_shape
            (Tensor(shape=[3], dtype=Int64, value= [4, 5, 3]),
             Tensor(shape=[3], dtype=Int64, value= [4, 5, 3]))

            # batched node features (feature dimension=10)
            >>> F1 = mindspore.numpy.rand((4, 10))
            >>> F2 = mindspore.numpy.rand((5, 10))
            >>> F3 = mindspore.numpy.rand((3, 10))
            >>> batched_F = pygm.utils.build_batch([F1, F2, F3])
            >>> batched_F.shape
            (3, 5, 10)

    """
    if backend is None:
        backend = pygmtools.BACKEND
    for item in input:
        _check_data_type(item, backend)
    args = (input, return_ori_dim)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.build_batch
    except ImportError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def dense_to_sparse(dense_adj, backend=None):
    r"""
    Convert a dense connectivity/adjacency matrix to a sparse connectivity/adjacency matrix and an edge weight tensor.

    :param dense_adj: :math:`(b\times n\times n)` the dense adjacency matrix. This function also supports non-batched
                      input where the batch dimension ``b`` is ignored
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if batched input:
             :math:`(b\times ne\times 2)` sparse connectivity matrix, :math:`(b\times ne\times 1)` edge weight tensor,
             :math:`(b)` number of edges

             if non-batched input:
             :math:`(ne\times 2)` sparse connectivity matrix, :math:`(ne\times 1)` edge weight tensor,

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'
            >>> np.random.seed(0)

            >>> batch_size = 10
            >>> A = np.random.rand(batch_size, 4, 4)
            >>> A[:, np.arange(4), np.arange(4)] = 0 # remove the diagonal elements
            >>> A.shape
            (10, 4, 4)

            >>> conn, edge, ne = pygm.utils.dense_to_sparse(A)
            >>> conn.shape # connectivity: (batch x num_edge x 2)
            (10, 12, 2)

            >>> edge.shape # edge feature (batch x num_edge x feature_dim)
            (10, 12, 1)

            >>> ne
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(0)

            >>> batch_size = 10
            >>> A = torch.rand(batch_size, 4, 4)
            >>> torch.diagonal(A, dim1=1, dim2=2)[:] = 0 # remove the diagonal elements
            >>> A.shape
            torch.Size([10, 4, 4])

            >>> conn, edge, ne = pygm.utils.dense_to_sparse(A)
            >>> conn.shape # connectivity: (batch x num_edge x 2)
            torch.Size([10, 12, 2])

            >>> edge.shape # edge feature (batch x num_edge x feature_dim)
            torch.Size([10, 12, 1])

            >>> ne
            tensor([12, 12, 12, 12, 12, 12, 12, 12, 12, 12])

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'
            >>> paddle.seed(0)

            >>> batch_size = 10
            >>> A = paddle.rand((batch_size, 4, 4))
            >>> paddle.diagonal(A, axis1=1, axis2=2)[:] = 0 # remove the diagonal elements
            >>> A.shape
            [10, 4, 4]

            >>> conn, edge, ne = pygm.utils.dense_to_sparse(A)
            >>> conn.shape # connectivity: (batch x num_edge x 2)
            torch.Size([10, 16, 2])

            >>> edge.shape # edge feature (batch x num_edge x feature_dim)
            torch.Size([10, 16, 1])

            >>> ne
            Tensor(shape=[10], dtype=int64, place=Place(cpu), stop_gradient=True,
                    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16])

    .. dropdown:: mindspore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'mindspore'
            >>> _ = mindspore.set_seed(0)

            >>> batch_size = 10
            >>> A = mindspore.numpy.rand((batch_size, 4, 4))
            >>> mindspore.numpy.diagonal(A, axis1=1, axis2=2)[:] = 0 # remove the diagonal elements
            >>> A.shape
            (10, 4, 4)

            >>> conn, edge, ne = pygm.utils.dense_to_sparse(A)
            >>> conn.shape # connectivity: (batch x num_edge x 2)
            (10, 16, 2)

            >>> edge.shape # edge feature (batch x num_edge x feature_dim)
            (10, 16, 1)

            >>> ne
            [16 16 16 16 16 16 16 16 16 16]

    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(dense_adj, backend)
    if _check_shape(dense_adj, 2, backend):
        dense_adj = _unsqueeze(dense_adj, 0, backend)
        non_batched_input = True
    elif _check_shape(dense_adj, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument dense_adj is expected to be 2-dimensional or 3-dimensional, got '
                         f'dense_adj:{len(_get_shape(dense_adj))}!')

    args = (dense_adj,)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.dense_to_sparse
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    if non_batched_input:
        return _squeeze(result[0], 0, backend), _squeeze(result[1], 0, backend)
    else:
        return result


def compute_affinity_score(X, K, backend=None):
    r"""
    Compute the affinity score of graph matching. It is the objective score of the corresponding Quadratic Assignment
    Problem.

    .. math::

        \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})

    here :math:`\texttt{vec}` means column-wise vectorization.

    :param X: :math:`(b\times n_1 \times n_2)` the permutation matrix that represents the matching result
    :param K: :math:`(b\times n_1n_2 \times n_1n_2)` the affinity matrix
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b)` the objective score

    .. note::

       This function also supports non-batched input if the batch dimension of ``X, K`` is ignored.

    .. dropdown:: Pytorch Example

        ::

            >>> import pygmtools as pygm
            >>> import torch
            >>> pygm.BACKEND = 'pytorch'

            # Generate a graph matching problem
            >>> X_gt = torch.zeros(4, 4)
            >>> X_gt[torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] =1
            >>> A1 = torch.rand(4, 4)
            >>> A2 = torch.mm(torch.mm(X_gt.transpose(0,1), A1), X_gt)
            >>> conn1, edge1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, None, None, None, None, edge_aff_fn=gaussian_aff)

            # Compute the objective score of ground truth matching
            >>> pygm.utils.compute_affinity_score(X_gt, K)
            tensor(16.)

    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(X, backend)
    _check_data_type(K, backend)
    if _check_shape(X, 2, backend) and _check_shape(K, 2, backend):
        X = _unsqueeze(X, 0, backend)
        K = _unsqueeze(K, 0, backend)
        non_batched_input = True
    elif _check_shape(X, 3, backend) and _check_shape(X, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument K, X are expected to have the same number of dimensions (=2 or 3), got'
                         f'X:{len(_get_shape(X))} and K:{len(_get_shape(K))}!')
    args = (X, K)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.compute_affinity_score
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def to_numpy(input, backend=None):
    r"""
    Convert a tensor to a numpy ndarray.
    This is the helper function to convert tensors across different backends via numpy.

    :param input: input tensor/:mod:`~pygmtools.utils.MultiMatchingResult`
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: numpy ndarray
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input,)
    # pygmtools built-in types
    if type(input) is MultiMatchingResult:
        fn = MultiMatchingResult.to_numpy
    # tf/torch/.. tensor types
    else:
        try:
            mod = importlib.import_module(f'pygmtools.{backend}_backend')
            fn = mod.to_numpy
        except (ModuleNotFoundError, AttributeError):
            raise NotImplementedError(
                NOT_IMPLEMENTED_MSG.format(backend)
            )
    return fn(*args)


def from_numpy(input, device=None, backend=None):
    r"""
    Convert a numpy ndarray to a tensor.
    This is the helper function to convert tensors across different backends via numpy.

    :param input: input ndarray/:mod:`~pygmtools.utils.MultiMatchingResult`
    :param device: (default: None) the target device
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: tensor for the backend
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, device)
    # pygmtools built-in types
    if type(input) is MultiMatchingResult:
        fn = functools.partial(MultiMatchingResult.from_numpy, new_backend=backend)
    # tf/torch/.. tensor types
    else:
        try:
            mod = importlib.import_module(f'pygmtools.{backend}_backend')
            fn = mod.from_numpy
        except (ModuleNotFoundError, AttributeError):
            raise NotImplementedError(
                NOT_IMPLEMENTED_MSG.format(backend)
            )
    return fn(*args)


def generate_isomorphic_graphs(node_num, graph_num=2, node_feat_dim=0, backend=None):
    r"""
    Generate a set of isomorphic graphs, for testing purposes and examples.

    :param node_num: number of nodes in each graph
    :param graph_num: (default: 2) number of graphs
    :param node_feat_dim: (default: 0) number of node feature dimensions
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if ``graph_num==2``, this function returns :math:`(m\times n \times n)` the adjacency matrix, and
             :math:`(n \times n)` the permutation matrix;

             else, this function returns :math:`(m\times n \times n)` the adjacency matrix, and
             :math:`(m\times m\times n \times n)` the multi-matching permutation matrix
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (node_num, graph_num, node_feat_dim)
    assert node_num > 0 and graph_num >= 2, "input data not understood."
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.generate_isomorphic_graphs
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    if node_feat_dim > 0:
        As, X_gt, Fs = fn(*args)
        if graph_num == 2:
            return As, X_gt[0, 1], Fs
        else:
            return As, X_gt, Fs
    else:
        As, X_gt = fn(*args)
        if graph_num == 2:
            return As, X_gt[0, 1]
        else:
            return As, X_gt


class MultiMatchingResult:
    r"""
    A memory-efficient class for multi-graph matching results. For non-cycle consistent results, the dense storage
    for :math:`m` graphs with :math:`n` nodes requires a size of :math:`(m\times m \times n \times n)`, and this
    implementation requires :math:`((m-1)\times m \times n \times n / 2)`. For cycle consistent result, this
    implementation requires only :math:`(m\times n\times n)`.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> np.random.seed(0)

            >>> X = pygm.utils.MultiMatchingResult(backend='numpy')
            >>> X[0, 1] = np.zeros((4, 4))
            >>> X[0, 1][np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
            >>> X
            MultiMatchingResult:
            {'0,1': array([[0., 0., 1., 0.],
                [0., 0., 0., 1.],
                [0., 1., 0., 0.],
                [1., 0., 0., 0.]])}
            >>> X[1, 0]
            array([[0., 0., 0., 1.],
                [0., 0., 1., 0.],
                [1., 0., 0., 0.],
                [0., 1., 0., 0.]])
    """
    def __init__(self, cycle_consistent=False, backend=None):
        self.match_dict = {}
        self._cycle_consistent = cycle_consistent
        if backend is None:
            self.backend = pygmtools.BACKEND
        else:
            self.backend = backend

    def __getitem__(self, item):
        assert len(item) == 2, "key should be the indices of two graphs, e.g. (0, 1)"
        idx1, idx2 = item
        if self._cycle_consistent:
            return _mm(self.match_dict[idx1], _transpose(self.match_dict[idx2], 0, 1, self.backend), self.backend)
        else:
            if idx1 < idx2:
                return self.match_dict[f'{idx1},{idx2}']
            else:
                return _transpose(self.match_dict[f'{idx2},{idx1}'], 0, 1, self.backend)

    def __setitem__(self, key, value):
        if self._cycle_consistent:
            assert type(key) is int, "key should be the index of one graph, and value should be the matching to universe"
            self.match_dict[key] = value
        else:
            assert len(key) == 2, "key should be the indices of two graphs, e.g. (0, 1)"
            idx1, idx2 = key
            if idx1 < idx2:
                self.match_dict[f'{idx1},{idx2}'] = value
            else:
                self.match_dict[f'{idx2},{idx1}'] = _transpose(value, 0, 1, self.backend)

    def __str__(self):
        return 'MultiMatchingResult:\n' + self.match_dict.__str__()

    def __repr__(self):
        return 'MultiMatchingResult:\n' + self.match_dict.__repr__()

    @staticmethod
    def from_numpy(data, device=None, new_backend=None):
        r"""
        Convert a numpy-backend MultiMatchingResult data to another backend.

        :param data: the numpy-backend data
        :param device: (default: None) the target device
        :param new_backend: (default: ``pygmtools.BACKEND`` variable) the target backend
        :return: a new MultiMatchingResult instance for ``new_backend`` on ``device``
        """
        new_data = copy.deepcopy(data)
        new_data.from_numpy_(device, new_backend)
        return new_data

    @staticmethod
    def to_numpy(data):
        r"""
        Convert an any-type MultiMatchingResult to numpy backend.

        :param data: the any-type data
        :return: a new MultiMatchingResult instance for numpy
        """
        new_data = copy.deepcopy(data)
        new_data.to_numpy_()
        return new_data

    def from_numpy_(self, device=None, new_backend=None):
        """
        In-place operation for :func:`~pygmtools.utils.MultiMatchingResult.from_numpy`.
        """
        if self.backend != 'numpy':
            raise ValueError('Attempting to convert from non-numpy data.')
        if new_backend is None:
            new_backend = pygmtools.BACKEND
        self.backend = new_backend
        for k, v in self.match_dict.items():
            self.match_dict[k] = from_numpy(v, device, self.backend)

    def to_numpy_(self):
        """
        In-place operation for :func:`~pygmtools.utils.MultiMatchingResult.to_numpy`.
        """
        for k, v in self.match_dict.items():
            self.match_dict[k] = to_numpy(v, self.backend)
        self.backend = 'numpy'


def get_network(nn_solver_func, **params):
    r"""
    Get the network object of a neural network solver.

    :param nn_solver_func: the neural network solver function, for example ``pygm.pca_gm``
    :param params: keyword parameters to define the neural network
    :return: the network object

    .. dropdown:: Pytorch Example

        ::

            >>> import pygmtools as pygm
            >>> import torch
            >>> pygm.BACKEND = 'pytorch'
            >>> pygm.utils.get_network(pygm.pca_gm, pretrain='willow')
            PCA_GM_Net(
              (gnn_layer_0): Siamese_Gconv(
                (gconv): Gconv(
                  (a_fc): Linear(in_features=1024, out_features=2048, bias=True)
                  (u_fc): Linear(in_features=1024, out_features=2048, bias=True)
                )
              )
              (cross_graph_0): Linear(in_features=4096, out_features=2048, bias=True)
              (affinity_0): WeightedInnerProdAffinity()
              (affinity_1): WeightedInnerProdAffinity()
              (gnn_layer_1): Siamese_Gconv(
                (gconv): Gconv(
                  (a_fc): Linear(in_features=2048, out_features=2048, bias=True)
                  (u_fc): Linear(in_features=2048, out_features=2048, bias=True)
                )
              )
            )

            # the neural network can be integrated into a deep learning pipeline
            >>> net = pygm.utils.get_network(pygm.pca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    """
    if 'return_network' in params:
        params.pop('return_network')
    # count parameters w/o default value
    sig = inspect.signature(nn_solver_func)
    required_params = 0
    for p in sig.parameters.items():
        if p[1].default is inspect._empty:
            required_params += 1
    _, net = nn_solver_func(*[None] * required_params, # fill the required parameters by None
                            return_network=True, **params)
    return net


def permutation_loss(pred_dsmat, gt_perm, n1=None, n2=None, backend=None):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
    :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
    :param n1: (optional) :math:`(b)` number of exact pairs in the first graph.
    :param n2: (optional) :math:`(b)` number of exact pairs in the second graph.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(1)` averaged permutation loss

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required if you want to specify the exact number of nodes of each instance in the batch.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch. This function
        also supports non-batched input if the batch dimension (:math:`b`) is ignored.
    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(pred_dsmat, backend)
    _check_data_type(gt_perm, backend)
    dsmat_shape = _get_shape(pred_dsmat, backend)
    perm_shape = _get_shape(gt_perm, backend)
    if len(dsmat_shape) == len(perm_shape) == 2:
        pred_dsmat = _unsqueeze(pred_dsmat, 0, backend)
        gt_perm = _unsqueeze(gt_perm, 0, backend)
    elif len(dsmat_shape) == len(perm_shape) == 3:
        pass
    else:
        raise ValueError(f'the input arguments pred_dsmat and gt_perm are expected to be 2-dimensional or 3-dimensional,'
                         f' got pred_dsmat:{len(dsmat_shape)}, gt_perm:{len(perm_shape)}!')

    for d1, d2 in zip(dsmat_shape, perm_shape):
        if d1 != d2:
            raise ValueError(f'dimension mismatch for pred_dsmat and gt_perm, got pred_dsmat:{dsmat_shape}, gt_perm:{gt_perm}!')

    args = (pred_dsmat, gt_perm, n1, n2)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.permutation_loss
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    return fn(*args)


###################################################
#   Private Functions that Unseeable from Users   #
###################################################


def _aff_mat_from_node_edge_aff(node_aff, edge_aff, connectivity1, connectivity2,
                                n1, n2, ne1, ne2,
                                backend=None):
    r"""
    Build affinity matrix K from node and edge affinity matrices.

    :param node_aff: :math:`(b\times n_1 \times n_2)` the node affinity matrix
    :param edge_aff: :math:`(b\times ne_1 \times ne_2)` the edge affinity matrix
    :param connectivity1: :math:`(b\times ne_1 \times 2)` sparse connectivity information of graph 1
    :param connectivity2: :math:`(b\times ne_2 \times 2)` sparse connectivity information of graph 2
    :param n1: :math:`(b)` number of nodes in graph1. If not given, it will be inferred based on the shape of
               ``node_feat1`` or the values in ``connectivity1``
    :param ne1: :math:`(b)` number of edges in graph1. If not given, it will be inferred based on the shape of
               ``edge_feat1``
    :param n2: :math:`(b)` number of nodes in graph2. If not given, it will be inferred based on the shape of
               ``node_feat2`` or the values in ``connectivity2``
    :param ne2: :math:`(b)` number of edges in graph2. If not given, it will be inferred based on the shape of
               ``edge_feat2``
    :return: :math:`(b\times n_1n_2 \times n_1n_2)` the affinity matrix
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (node_aff, edge_aff, connectivity1, connectivity2, n1, n2, ne1, ne2)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._aff_mat_from_node_edge_aff
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _check_data_type(input, *args):
    r"""
    Check whether the input data meets the backend. If not met, it will raise an ValueError
    Three overloads of this function:
    _check_data_type(input, backend)
    _check_data_type(input, var_name, backend)
    _check_data_type(input, var_name, raise_err, backend)

    :param input: input data (must be Tensor/ndarray)
    :param var_name: name of the variable
    :param raise_err: raise an error if input data not true
    :return: True or False
    """
    if len(args) == 3:
        var_name, raise_err, backend = args
    elif len(args) == 2:
        var_name, backend = args
        raise_err = True
    elif len(args) == 1:
        backend = args[0]
        var_name = None
        raise_err = True
    elif len(args) == 0:
        backend = None
        var_name = None
        raise_err = True
    else:
        raise RuntimeError(f'Unknown arguments: {args}')

    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, var_name, raise_err)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._check_data_type
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _check_shape(input, num_dim, backend=None):
    r"""
    Check the shape of the input tensor

    :param input: the input tensor
    :param num_dim: number of dimensions
    :return: True or False
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, num_dim)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._check_shape
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _get_shape(input, backend=None):
    r"""
    Get the shape of the input tensor

    :param input: the input tensor
    :return: a list of ints indicating the shape
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input,)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._get_shape
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _squeeze(input, dim, backend=None):
    r"""
    Squeeze the input tensor at the given dimension. This function is expected to behave the same as torch.squeeze

    :param input: input tensor
    :param dim: squeezed dimension
    :return: squeezed tensor
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, dim)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._squeeze
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _unsqueeze(input, dim, backend=None):
    r"""
    Unsqueeze the input tensor at the given dimension. This function is expected to behave the same as torch.unsqueeze

    :param input: input tensor
    :param dim: unsqueezed dimension
    :return: unsqueezed tensor
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, dim)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._unsqueeze
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _transpose(input, dim1, dim2, backend=None):
    r"""
    Swap the dim1 and dim2 dimensions of the input tensor.

    :param input: input tensor
    :param dim1: swapped dimension 1
    :param dim2: swapped dimension 2
    :return: transposed tensor
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, dim1, dim2)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._transpose
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def _mm(input1, input2, backend=None):
    r"""
    Matrix multiplication.

    :param input1: input tensor 1
    :param input2: input tensor 2
    :return: multiplication result
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input1, input2)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod._mm
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def download(filename, url, md5=None, retries=10, to_cache=True):
    r"""
    Check if content exits. If not, download the content to ``<user cache path>/pygmtools/<filename>``. ``<user cache path>``
    depends on your system. For example, on Debian, it should be ``$HOME/.cache``.
    :param filename: the destination file name
    :param url: the url
    :param md5: (optional) the md5sum to verify the content. It should match the result of ``md5sum file`` on Linux.
    :param retries: (default: 10) max number of retries
    :return: the full path to the file: ``<user cache path>/pygmtools/<filename>``
    """
    if retries <= 0:
        raise RuntimeError('Max Retries exceeded!')

    if to_cache:
        dirs = user_cache_dir("pygmtools")
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        filename = os.path.join(dirs, filename)
    if not os.path.exists(filename):
        print(f'\nDownloading to {filename}...')
        if retries % 3 == 1:
            try:
                down_res = requests.get(url, stream=True)
                file_size = int(down_res.headers.get('Content-Length', 0))
                with tqdm.wrapattr(down_res.raw, "read", total=file_size) as content:
                    with open(filename, 'wb') as file:
                        shutil.copyfileobj(content, file)
            except requests.exceptions.ConnectionError as err:
                print('Warning: Network error. Retrying...\n', err)
                return download(filename, url, md5, retries - 1)
        elif retries % 3 == 2:
            try:
                wget.download(url,out=filename)
            except:
                return download(filename, url, md5, retries - 1)
        else:
            try:
                urllib.request.urlretrieve(url, filename)
            except:
                return download(filename, url, md5, retries - 1)  
            
    if md5 is not None:
        md5_returned = _get_md5(filename)
        if md5 != md5_returned:
            print('Warning: MD5 check failed for the downloaded content. Retrying...')
            os.remove(filename)
            time.sleep(1)
            return download(filename, url, md5, retries - 1)
    return filename


def _get_md5(filename):
    hash_md5 = hashlib.md5()
    chunk = 8192
    with open(filename, 'rb') as file_to_check:
        while True:
            buffer = file_to_check.read(chunk)
            if not buffer:
                break
            hash_md5.update(buffer)
        md5_returned = hash_md5.hexdigest()
        return md5_returned


###################################################
#       Support NetworkX and GraphML formats      #
###################################################


def build_aff_mat_from_networkx(G1:nx.Graph, G2:nx.Graph, node_aff_fn=None, edge_aff_fn=None, backend=None):
    r"""
    Convert networkx object to Adjacency matrix
    
    :param G1: networkx object, whose type must be networkx.Graph
    :param G2: networkx object, whose type must be networkx.Graph
    :param node_aff_fn: (default: inner_prod_aff_fn) the node affinity function with the characteristic
                        ``node_aff_fn(2D Tensor, 2D Tensor) -> 2D Tensor``, which accepts two node feature tensors and
                        outputs the node-wise affinity tensor. See :func:`~pygmtools.utils.inner_prod_aff_fn` as an
                        example.
    :param edge_aff_fn: (default: inner_prod_aff_fn) the edge affinity function with the characteristic
                        ``edge_aff_fn(2D Tensor, 2D Tensor) -> 2D Tensor``, which accepts two edge feature tensors and
                        outputs the edge-wise affinity tensor. See :func:`~pygmtools.utils.inner_prod_aff_fn` as an
                        example.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: the affinity matrix corresponding to the networkx object G1 and G2

    .. dropdown:: Example

        ::

            >>> import networkx as nx
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # Generate networkx images
            >>> G1 = nx.DiGraph()
            >>> G1.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 0.8), (3, 4, 0.7)])
            >>> G2 = nx.DiGraph()
            >>> G2.add_weighted_edges_from([(1, 2, 0.3), (2, 3, 0.6), (3, 4, 0.9), (4, 5, 0.4)])

            # Obtain Affinity Matrix
            >>> K = pygm.utils.build_aff_mat_from_networkx(G1, G2)
            >>> K.shape
            (20,20)
            
            # The affinity matrices K can be further processed by GM solvers
    """
    if backend is None:
        backend = pygmtools.BACKEND
    A1 = from_numpy(np.asarray(from_networkx(G1)))
    A2 = from_numpy(np.asarray(from_networkx(G2)))
    conn1, edge1 = dense_to_sparse(A1, backend=backend)
    conn2, edge2 = dense_to_sparse(A2, backend=backend)
    K = build_aff_mat(None, edge1, conn1, None, edge2, conn2, node_aff_fn=node_aff_fn, edge_aff_fn=edge_aff_fn, backend=backend)
    return K


def build_aff_mat_from_graphml(G1_path, G2_path, node_aff_fn=None, edge_aff_fn=None, backend=None):
    r"""
    Convert networkx object to Adjacency matrix
    
    :param G1_path: The file path of the graphml object
    :param G2_path: The file path of the graphml object
    :param node_aff_fn: (default: inner_prod_aff_fn) the node affinity function with the characteristic
                        ``node_aff_fn(2D Tensor, 2D Tensor) -> 2D Tensor``, which accepts two node feature tensors and
                        outputs the node-wise affinity tensor. See :func:`~pygmtools.utils.inner_prod_aff_fn` as an
                        example.
    :param edge_aff_fn: (default: inner_prod_aff_fn) the edge affinity function with the characteristic
                        ``edge_aff_fn(2D Tensor, 2D Tensor) -> 2D Tensor``, which accepts two edge feature tensors and
                        outputs the edge-wise affinity tensor. See :func:`~pygmtools.utils.inner_prod_aff_fn` as an
                        example.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: the affinity matrix corresponding to the graphml object G1 and G2
    

    .. dropdown:: Example

        ::

            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # example file (.graphml) path
            >>> G1_path = 'examples/data/graph1.graphml'
            >>> G2_path = 'examples/data/graph2.graphml'

            # Obtain Affinity Matrix
            >>> K = pygm.utils.build_aff_mat_from_graphml(G1_path, G2_path)
            >>> K.shape
            (121,121)
            
            # The affinity matrices K can be further processed by GM solvers
    """
    if backend is None:
        backend = pygmtools.BACKEND
    A1 = from_numpy(np.asarray(from_graphml(G1_path)))
    A2 = from_numpy(np.asarray(from_graphml(G2_path)))
    conn1, edge1 = dense_to_sparse(A1, backend=backend)
    conn2, edge2 = dense_to_sparse(A2, backend=backend)
    K = build_aff_mat(None, edge1, conn1, None, edge2, conn2, node_aff_fn=node_aff_fn, edge_aff_fn=edge_aff_fn, backend=backend)
    return K
 
    
def from_networkx(G:nx.Graph):
    r"""
    Convert networkx object to Adjacency matrix
    
    :param G: networkx object, whose type must be networkx.Graph
    :return: the adjacency matrix corresponding to the networkx object

    .. dropdown:: Example

        ::

            >>> import networkx as nx
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # Generate networkx graphs
            >>> G1 = nx.DiGraph()
            >>> G1.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 0.8), (3, 4, 0.7)])
            >>> G2 = nx.DiGraph()
            >>> G2.add_weighted_edges_from([(1, 2, 0.3), (2, 3, 0.6), (3, 4, 0.9), (4, 5, 0.4)])

            # Obtain Adjacency matrix
            >>> pygm.utils.from_networkx(G1)
            matrix([[0. , 0.5, 0. , 0. ],
                    [0. , 0. , 0.8, 0. ],
                    [0. , 0. , 0. , 0.7],
                    [0. , 0. , 0. , 0. ]])
                    
            >>> pygm.utils.from_networkx(G2)
            matrix([[0. , 0.3, 0. , 0. , 0. ],
                    [0. , 0. , 0.6, 0. , 0. ],
                    [0. , 0. , 0. , 0.9, 0. ],
                    [0. , 0. , 0. , 0. , 0.4],
                    [0. , 0. , 0. , 0. , 0. ]])
            
    """
    is_directed = isinstance(G, nx.DiGraph)
    adj_matrix = nx.to_numpy_matrix(G,nodelist=G.nodes()) if is_directed else nx.to_numpy_matrix(G)
    return adj_matrix


def to_networkx(adj_matrix, backend=None):
    """
    Convert adjacency matrix to NetworkX object
    
    :param adj_matrix: the adjacency matrix to convert
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: the NetworkX object corresponding to the adjacency matrix
    
    .. dropdown:: Example

        ::

            >>> import networkx as nx
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # Generate adjacency matrix
            >>> adj_matrix = np.random.random(size=(4,4))
            
            # Obtain NetworkX object
            >>> pygm.utils.to_networkx(adj_matrix)
            <networkx.classes.digraph.DiGraph at ...>
    """
    if backend is None:
        backend = pygmtools.BACKEND
    adj_matrix = to_numpy(adj_matrix, backend=backend)
    
    if adj_matrix.ndim == 3 and adj_matrix.shape[0] == 1:
        adj_matrix.squeeze(0)
    assert adj_matrix.ndim == 2, 'Request the dimension of adj_matrix is 2'
    
    G = nx.DiGraph() if np.any(adj_matrix != adj_matrix.T) else nx.Graph()
    G.add_nodes_from(range(adj_matrix.shape[0]))
    for i, j in zip(*np.where(adj_matrix)):
        G.add_edge(i, j, weight=adj_matrix[i, j])
    return G


def from_graphml(filename):
    r"""
    Convert graphml object to Adjacency matrix
    
    :param filename: graphml file path
    :return: the adjacency matrix corresponding to the graphml object

    .. dropdown:: Example

        ::

            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'

            # example file (.graphml) path
            >>> G1_path = 'examples/data/graph1.graphml'
            >>> G2_path = 'examples/data/graph2.graphml'

            # Obtain Adjacency matrix
            >>> G1 = pygm.utils.from_graphml(G1_path)
            >>> G1.shape
            (11,11)
                    
            >>> G1 = pygm.utils.from_graphml(G2_path)
            >>> G2.shape
            (11,11)
    """
    if not filename.endswith('.graphml'):
        raise ValueError("File name should end with '.graphml'")
    if not os.path.isfile(filename):
        raise ValueError("File not found: {}".format(filename))
    return from_networkx(nx.read_graphml(filename))


def to_graphml(adj_matrix, filename, backend=None):
    r"""
    Write an adjacency matrix to a GraphML file
    
    :param adj_matrix: numpy.ndarray, the adjacency matrix to write
    :param filename: str, the name of the output file
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.

    .. dropdown:: Example

        ::

            >>> import pygmtools as pygm
            >>> import numpy as np
            >>> pygm.BACKEND = 'numpy'
            
            # Generate adjacency matrix
            >>> adj_matrix = np.random.random(size=(4,4))
            >>> filename = 'examples/data/test.graphml'
            >>> adj_matrix
            array([[0.29440151, 0.66468829, 0.05403941, 0.85887567],
                   [0.48120964, 0.01429095, 0.73536659, 0.02962113],
                   [0.3815578 , 0.93356234, 0.01332568, 0.61149257],
                   [0.15422904, 0.64656912, 0.93219422, 0.784769  ]])

            # Write GraphML file
            >>> pygm.utils.to_graphml(adj_matrix, filename)
            
            # Check the generated GraphML file
            >>> pygm.utils.from_graphml(filename)
            array([[0.29440151, 0.66468829, 0.05403941, 0.85887567],
                   [0.48120964, 0.01429095, 0.73536659, 0.02962113],
                   [0.3815578 , 0.93356234, 0.01332568, 0.61149257],
                   [0.15422904, 0.64656912, 0.93219422, 0.784769  ]])            
    """
    nx.write_graphml(to_networkx(adj_matrix, backend), filename)
    