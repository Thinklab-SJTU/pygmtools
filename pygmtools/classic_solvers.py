r"""
Classic (learning-free) **two-graph matching** solvers. These two-graph matching solvers are recommended to solve
matching problems with two explicit graphs, or problems formulated as Quadratic Assignment Problem (QAP).

The two-graph matching problem considers both nodes and edges, formulated as a QAP (Lawler's):

.. math::

    &\max_{\mathbf{X}} \ \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})\\
    s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}
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

import importlib
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, _check_shape, _get_shape,\
    _unsqueeze, _squeeze, _check_data_type,from_numpy
import numpy as np

def sm(K, n1=None, n2=None, n1max=None, n2max=None, x0=None,
       max_iter: int=50,
       backend=None):
    r"""
    Spectral Graph Matching solver for graph matching (Lawler's QAP).
    This algorithm is also known as Power Iteration method, because it works by computing the leading
    eigenvector of the input affinity matrix by power iteration.

    For each iteration,

    .. math::

        \mathbf{v}_{k+1} = \mathbf{K} \mathbf{v}_k / ||\mathbf{K} \mathbf{v}_k||_2

    :param K: :math:`(b\times n_1n_2 \times n_1n_2)` the input affinity matrix, :math:`b`: batch size.
    :param n1: :math:`(b)` number of nodes in graph1 (optional if n1max is given, and all n1=n1max).
    :param n2: :math:`(b)` number of nodes in graph2 (optional if n2max is given, and all n2=n2max).
    :param n1max: :math:`(b)` max number of nodes in graph1 (optional if n1 is given, and n1max=max(n1)).
    :param n2max: :math:`(b)` max number of nodes in graph2 (optional if n2 is given, and n2max=max(n2)).
    :param x0: :math:`(b\times n_1 \times n_2)` an initial matching solution for warm-start.
               If not given, x0 will be randomly generated.
    :param max_iter: (default: 50) max number of iterations. More iterations will help the solver to converge better,
                     at the cost of increased inference time.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1 \times n_2)` the solved doubly-stochastic matrix

    .. note::
        Either ``n1`` or ``n1max`` should be specified because it cannot be inferred from the input tensor size.
        Same for ``n2`` or ``n2max``.

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded and all elements in ``n1`` are equal, all in ``n2`` are equal.

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. note::
        This solver is differentiable and supports gradient back-propagation.

    .. warning::
        The solver's output is normalized with a squared sum of 1, which is in line with the original implementation. If
        a doubly-stochastic matrix is required, please call :func:`~pygmtools.classic_solvers.sinkhorn` after this. If a
        discrete permutation matrix is required, please call :func:`~pygmtools.classic_solvers.hungarian`. Note that the
        Hungarian algorithm will truncate the gradient and the Sinkhorn algorithm will not.

    .. dropdown:: Numpy Example

        ::
        
            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'
            >>> np.random.seed(1)
    
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
            >>> A1 = np.random.rand(batch_size, 4, 4)
            >>> A2 = np.matmul(np.matmul(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> n1 = n2 = np.repeat([4], batch_size)
    
            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
            # Solve by SM. Note that X is normalized with a squared sum of 1
            >>> X = pygm.sm(K, n1, n2)
            >>> (X ** 2).sum(axis=(1, 2))
            array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    
            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)
    
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = torch.zeros(batch_size, 4, 4)
            >>> X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
            >>> A1 = torch.rand(batch_size, 4, 4)
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = n2 = torch.tensor([4] * batch_size)
    
            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
            # Solve by SM. Note that X is normalized with a squared sum of 1
            >>> X = pygm.sm(K, n1, n2)
            >>> (X ** 2).sum(dim=(1, 2))
            tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                    1.0000])
    
            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            tensor(1.)
    
            # This solver supports gradient back-propogation
            >>> K = K.requires_grad_(True)
            >>> pygm.sm(K, n1, n2).sum().backward()
            >>> len(torch.nonzero(K.grad))
            2560

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'
            >>> _ = paddle.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = paddle.rand((batch_size, 4, 4))
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> n1 = n2 = paddle.to_tensor([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by SM. Note that X is normalized with a squared sum of 1
            >>> X = pygm.sm(K, n1, n2)
            >>> (X ** 2).sum(axis=(1, 2))
            Tensor(shape=[10], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [1.        , 1.        , 0.99999994, 0.99999994, 1.00000012, 
                     1.        , 1.00000012, 1.        , 1.        , 0.99999994])

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [1.])

            # This solver supports gradient back-propogation
            >>> K.stop_gradient = False
            >>> pygm.sm(K, n1, n2).sum().backward()
            >>> len(paddle.nonzero(K.grad))
            2560

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'jittor'
            >>> _ = jt.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = jt.zeros((batch_size, 4, 4))
            >>> X_gt[:, jt.arange(0, 4, dtype=jt.int64), jt.randperm(4)] = 1
            >>> A1 = jt.rand(batch_size, 4, 4)
            >>> A2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = n2 = jt.Var([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by SM. Note that X is normalized with a squared sum of 1
            >>> X = pygm.sm(K, n1, n2)
            >>> (X ** 2).sum(dim=1).sum(dim=1)
            jt.Var([0.9999998  1.         0.9999999  1.0000001  1.         1.
                    0.9999999  0.99999994 1.0000001  1.        ], dtype=float32)

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            jt.Var([1.], dtype=float32)
            
            # This solver supports gradient back-propogation
            >>> from jittor import nn
            >>> class Model(nn.Module):
            ...     def __init__(self, K):
            ...         self.K = K
            ...     def execute(self, K, n1, n2):
            ...         X = pygm.sm(K, n1, n2)
            ...         return X

            >>> model = Model(K)
            >>> optim = nn.SGD(model.parameters(), lr=0.1)
            >>> X = model(K, n1, n2)
            >>> loss = X.sum()
            >>> optim.step(loss)
            >>> len(jt.nonzero(K.opt_grad(optim)))
            2560

    .. dropdown:: MindSpore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'mindspore'
            >>> _ = mindspore.set_seed(1)
            >>> mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = mindspore.numpy.zeros((batch_size, 4, 4))
            >>> X_gt[:, mindspore.numpy.arange(0, 4, dtype=mindspore.int64), mindspore.ops.Randperm(4)(mindspore.Tensor([4], dtype=mindspore.int32))] = 1
            >>> A1 = mindspore.numpy.rand((batch_size, 4, 4))
            >>> A2 = mindspore.ops.BatchMatMul()(mindspore.ops.BatchMatMul()(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> n1 = n2 = mindspore.Tensor([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by SM. Note that X is normalized with a squared sum of 1
            >>> X = pygm.sm(K, n1, n2)
            >>> (X ** 2).sum(axis=(1, 2))
            [1.0000002  0.9999998  1.0000002  0.99999964 1.         1.0000001
            1.         1.         1.         0.99999994]

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0

            # This solver supports gradient back-propogation
            >>> def fn(K, n1, n2):
            >>>     res = pygm.sm(K, n1, n2).sum()
            >>>     return res

            >>> g = mindspore.ops.grad(fn)(K, n1, n2)
            >>> mindspore.ops.count_nonzero(g)
            
            # This solver supports gradient back-propogation
            >>> from jittor import nn
            >>> class Model(nn.Module):
            ...     def __init__(self, K):
            ...         self.K = K
            ...     def execute(self, K, n1, n2):
            ...         X = pygm.sm(K, n1, n2)
            ...         return X

            >>> model = Model(K)
            >>> optim = nn.SGD(model.parameters(), lr=0.1)
            >>> X = model(K, n1, n2)
            >>> loss = X.sum()
            >>> optim.step(loss)
            >>> len(jt.nonzero(K.opt_grad(optim)))
            2560

    .. dropdown:: Tensorflow Example

        ::

            >>> import tensorflow as tf
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'tensorflow'
            >>> _ = tf.random.set_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = tf.Variable(tf.zeros([batch_size, 4, 4]))
            >>> indices = tf.stack([tf.range(4),tf.random.shuffle(tf.range(4))], axis=1)
            >>> updates = tf.ones([4])
            >>> for i in range(batch_size):
            ...     _ = X_gt[i].assign(tf.tensor_scatter_nd_update(X_gt[i], indices, updates))
            >>> A1 = tf.random.uniform([batch_size, 4, 4])
            >>> A2 = tf.matmul(tf.matmul(tf.transpose(X_gt, perm=[0, 2, 1]), A1), X_gt)
            >>> n1 = n2 = tf.constant([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by SM. Note that X is normalized with a squared sum of 1
            >>> X = pygm.sm(K, n1, n2)
            >>> (X ** 2).sum(axis=(1, 2))
            [1.         0.9999998  0.99999976 1.         0.99999976 1.
            1.         1.0000001  1.0000001  1.        ]

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0
            >>> tf.reduce_sum((X ** 2), axis=[1, 2])
            <tf.Tensor: shape=(10,), dtype=float32, numpy=
            array([1.        , 1.0000001 , 1.        , 0.9999999 , 1.        ,
                   1.        , 1.0000001 , 0.99999994, 1.        , 0.9999998 ],
                  dtype=float32)>

            # Accuracy
            >>> tf.reduce_sum((pygm.hungarian(X) * X_gt))/ tf.reduce_sum(X_gt)
            <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

            This solver supports gradient back-propogation
            >>> K = tf.Variable(K)
            >>> with tf.GradientTape() as tape:
            ...     y = tf.reduce_sum(pygm.sm(K, n1, n2))
            ...     len(tf.where(tape.gradient(y, K)))
            2560

    .. note::
        If you find this graph matching solver useful for your research, please cite:

        ::

            @inproceedings{sm,
              title={A spectral technique for correspondence problems using pairwise constraints},
              author={Leordeanu, Marius and Hebert, Martial},
              year={2005},
              pages={1482-1489},
              booktitle={International Conference on Computer Vision},
              publisher={IEEE}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(K, 'K', backend)
    if _check_shape(K, 2, backend):
        K = _unsqueeze(K, 0, backend)
        non_batched_input = True
        if type(n1) is int and n1max is None:
            n1max = n1
            n1 = None
        if type(n2) is int and n2max is None:
            n2max = n2
            n2 = None
    elif _check_shape(K, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument K is expected to be 2-dimensional or 3-dimensional, got '
                         f'K:{len(_get_shape(K, backend))}dims!')
    __check_gm_arguments(n1, n2, n1max, n2max)
    args = (K, n1, n2, n1max, n2max, x0, max_iter)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.sm
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def rrwm(K, n1=None, n2=None, n1max=None, n2max=None, x0=None,
         max_iter: int=50, sk_iter: int=20, alpha: float=0.2, beta: float=30,
         backend=None):
    r"""
    Reweighted Random Walk Matching (RRWM) solver for graph matching (Lawler's QAP). This algorithm is implemented by
    power iteration with Sinkhorn reweighted jumps.

    The official matlab implementation is available at https://cv.snu.ac.kr/research/~RRWM/

    :param K: :math:`(b\times n_1n_2 \times n_1n_2)` the input affinity matrix, :math:`b`: batch size.
    :param n1: :math:`(b)` number of nodes in graph1 (optional if n1max is given, and all n1=n1max).
    :param n2: :math:`(b)` number of nodes in graph2 (optional if n2max is given, and all n2=n2max).
    :param n1max: :math:`(b)` max number of nodes in graph1 (optional if n1 is given, and n1max=max(n1)).
    :param n2max: :math:`(b)` max number of nodes in graph2 (optional if n2 is given, and n2max=max(n2)).
    :param x0: :math:`(b\times n_1 \times n_2)` an initial matching solution for warm-start.
               If not given, x0 will filled with :math:`\frac{1}{n_1 n_2})`.
    :param max_iter: (default: 50) max number of iterations (i.e. number of random walk steps) in RRWM.
                     More iterations will be lead to more accurate result, at the cost of increased inference time.
    :param sk_iter: (default: 20) max number of Sinkhorn iterations. More iterations will be lead to more accurate
                    result, at the cost of increased inference time.
    :param alpha: (default: 0.2) the parameter controlling the importance of the reweighted jump. alpha should lie
                  between 0 and 1. If ``alpha=0``, it means no reweighted jump;
                  if alpha=1, the reweighted jump provides all information.
    :param beta: (default: 30) the temperature parameter of exponential function before the Sinkhorn operator.
                 ``beta`` should be larger than 0. A larger ``beta`` means more confidence in the jump. A larger
                 ``beta`` will usually require a larger ``sk_iter``.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1 \times n_2)` the solved matching matrix

    .. note::
        Either ``n1`` or ``n1max`` should be specified because it cannot be inferred from the input tensor size.
        Same for ``n2`` or ``n2max``.

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded and all elements in ``n1`` are equal, all in ``n2`` are equal.

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. note::
        This solver is differentiable and supports gradient back-propagation.

    .. warning::
        The solver's output is normalized with a sum of 1, which is in line with the original implementation. If a doubly-
        stochastic matrix is required, please call :func:`~pygmtools.classic_solvers.sinkhorn` after this. If a discrete
        permutation matrix is required, please call :func:`~pygmtools.classic_solvers.hungarian`. Note that the
        Hungarian algorithm will truncate the gradient and the Sinkhorn algorithm will not.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'
            >>> np.random.seed(1)
    
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
            >>> A1 = np.random.rand(batch_size, 4, 4)
            >>> A2 = np.matmul(np.matmul(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> n1 = n2 = np.repeat([4], batch_size)
    
            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
            # Solve by RRWM. Note that X is normalized with a sum of 1
            >>> X = pygm.rrwm(K, n1, n2, beta=100)
            >>> X.sum(axis=(1, 2))
            array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    
            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)
    
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = torch.zeros(batch_size, 4, 4)
            >>> X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
            >>> A1 = torch.rand(batch_size, 4, 4)
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = n2 = torch.tensor([4] * batch_size)
    
            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
            # Solve by RRWM. Note that X is normalized with a sum of 1
            >>> X = pygm.rrwm(K, n1, n2, beta=100)
            >>> X.sum(dim=(1, 2))
            tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                    1.0000])
    
            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            tensor(1.)
    
            # This solver supports gradient back-propogation
            >>> K = K.requires_grad_(True)
            >>> pygm.rrwm(K, n1, n2, beta=100).sum().backward()
            >>> len(torch.nonzero(K.grad))
            272

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'
            >>> _ = paddle.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = paddle.rand((batch_size, 4, 4))
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> n1 = n2 = paddle.to_tensor([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by RRWM. Note that X is normalized with a sum of 1
            >>> X = pygm.rrwm(K, n1, n2, beta=100)
            >>> X.sum(axis=(1, 2))
            Tensor(shape=[10], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.99999988, 0.99999988, 0.99999994, 0.99999994, 1.        , 
                     1.        , 1.        , 1.00000012, 1.00000012, 1.        ])

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [1.])

            # This solver supports gradient back-propogation
            >>> K.stop_gradient = False
            >>> pygm.rrwm(K, n1, n2, beta=100).sum().backward()
            >>> len(paddle.nonzero(K.grad))
            544

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'jittor'
            >>> _ = jt.seed(1)


            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = jt.zeros((batch_size, 4, 4))
            >>> X_gt[:, jt.arange(0, 4, dtype=jt.int64), jt.randperm(4)] = 1
            >>> A1 = jt.rand(batch_size, 4, 4)
            >>> A2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = n2 = jt.Var([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by RRWM. Note that X is normalized with a sum of 1
            >>> X = pygm.rrwm(K, n1, n2, beta=100)
            >>> X.sum(dims=(1, 2))
            jt.Var([1.         1.0000001  1.         0.99999976 1.         
                    1.         1.         1.0000001  0.99999994 1.        ], dtype=float32)

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            jt.Var([1.], dtype=float32)
            
            # This solver supports gradient back-propogation
            >>> from jittor import nn
            >>> class Model(nn.Module):
            ...     def __init__(self, K):
            ...         self.K = K
            ...     def execute(self, K, n1, n2, beta):
            ...         X = pygm.rrwm(K, n1, n2, beta=beta)
            ...         return X

            >>> model = Model(K)
            >>> optim = nn.SGD(model.parameters(), lr=0.1)
            >>> X = model(K, n1, n2, beta=100)
            >>> loss = X.sum()
            >>> optim.step(loss)
            >>> len(jt.nonzero(K.opt_grad(optim)))
            1536

    .. dropdown:: MindSpore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'mindspore'
            >>> _ = mindspore.set_seed(1)
            >>> mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = mindspore.numpy.zeros((batch_size, 4, 4))
            >>> X_gt[:, mindspore.numpy.arange(0, 4, dtype=mindspore.int64), mindspore.ops.Randperm(4)(mindspore.Tensor([4], dtype=mindspore.int32))] = 1
            >>> A1 = mindspore.numpy.rand((batch_size, 4, 4))
            >>> A2 = mindspore.ops.BatchMatMul()(mindspore.ops.BatchMatMul()(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> n1 = n2 = mindspore.Tensor([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by RRWM. Note that X is normalized with a sum of 1
            >>> X = pygm.rrwm(K, n1, n2, beta=100)
            >>> X.sum(axis=(1, 2))
            [1.         0.99999994 0.99999994 1.         1.0000002  1.
            1.         1.         1.0000001  1.0000001 ]

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0

            # This solver supports gradient back-propogation
            >>> def fn(K, n1, n2, beta):
            >>>     X = pygm.rrwm(K, n1, n2, beta=beta)
            >>>     X_gt = mindspore.numpy.zeros((batch_size, 4, 4))
            >>>     X_gt[:, mindspore.numpy.arange(0, 4, dtype=mindspore.int64),mindspore.ops.Randperm(4)(mindspore.Tensor([4], dtype=mindspore.int32))] = 1
            >>>     res = pygm.utils.permutation_loss(X, X_gt)
            >>>     return res

            >>> g = mindspore.ops.grad(fn)(K, n1, n2, beta=100)
            >>> mindspore.ops.count_nonzero(g)
            2560

    .. dropdown:: Tensorflow Example

        ::

            >>> import tensorflow as tf
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'tensorflow'
            >>> _ = tf.random.set_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = tf.Variable(tf.zeros([batch_size, 4, 4]))
            >>> indices = tf.stack([tf.range(4),tf.random.shuffle(tf.range(4))], axis=1)
            >>> updates = tf.ones([4])
            >>> for i in range(batch_size):
            ...     _ = X_gt[i].assign(tf.tensor_scatter_nd_update(X_gt[i], indices, updates))
            >>> A1 = tf.random.uniform([batch_size, 4, 4])
            >>> A2 = tf.matmul(tf.matmul(tf.transpose(X_gt, perm=[0, 2, 1]), A1), X_gt)
            >>> n1 = n2 = tf.constant([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by RRWM. Note that X is normalized with a sum of 1
            >>> X = pygm.rrwm(K, n1, n2, beta=100)
            >>> tf.reduce_sum(X, axis=[1, 2])
            <tf.Tensor: shape=(10,), dtype=float32, numpy=
            array([1.        , 1.        , 1.        , 0.99999994, 1.        ,
                   1.        , 1.        , 0.99999994, 0.99999994, 1.0000001 ],
                  dtype=float32)>

            # Accuracy
            >>> tf.reduce_sum((pygm.hungarian(X) * X_gt)) / tf.reduce_sum(X_gt)
            <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

            # This solver supports gradient back-propogation
            >>> K = tf.Variable(K)
            >>> with tf.GradientTape(persistent=True) as tape:
            ...     y = tf.reduce_sum(pygm.rrwm(K, n1, n2, beta=100))
            ...     len(tf.where(tape.gradient(y, K)))
            768

    .. note::
        If you find this graph matching solver useful in your research, please cite:

        ::

            @inproceedings{rrwm,
              title={Reweighted random walks for graph matching},
              author={Cho, Minsu and Lee, Jungmin and Lee, Kyoung Mu},
              booktitle={European conference on Computer vision},
              pages={492--505},
              year={2010},
              organization={Springer}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(K, 'K', backend)
    if _check_shape(K, 2, backend):
        K = _unsqueeze(K, 0, backend)
        non_batched_input = True
        if type(n1) is int and n1max is None:
            n1max = n1
            n1 = None
        if type(n2) is int and n2max is None:
            n2max = n2
            n2 = None
    elif _check_shape(K, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument K is expected to be 2-dimensional or 3-dimensional, got '
                         f'K:{len(_get_shape(K, backend))}dims!')
    __check_gm_arguments(n1, n2, n1max, n2max)
    assert 0 <= alpha <= 1, f'illegal value of alpha, it should lie between 0 and 1, got alpha={alpha}!.'
    assert beta > 0, f'illegal value of beta, it should be larger than 0, got beta={beta}!'

    args = (K, n1, n2, n1max, n2max, x0, max_iter, sk_iter, alpha, beta)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.rrwm
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def ipfp(K, n1=None, n2=None, n1max=None, n2max=None, x0=None,
         max_iter: int=50,
         backend=None):
    r"""
    Integer Projected Fixed Point (IPFP) method for graph matching (Lawler's QAP).

    :param K: :math:`(b\times n_1n_2 \times n_1n_2)` the input affinity matrix, :math:`b`: batch size.
    :param n1: :math:`(b)` number of nodes in graph1 (optional if n1max is given, and all n1=n1max).
    :param n2: :math:`(b)` number of nodes in graph2 (optional if n2max is given, and all n2=n2max).
    :param n1max: :math:`(b)` max number of nodes in graph1 (optional if n1 is given, and n1max=max(n1)).
    :param n2max: :math:`(b)` max number of nodes in graph2 (optional if n2 is given, and n2max=max(n2)).
    :param x0: :math:`(b\times n_1 \times n_2)` an initial matching solution for warm-start.
               If not given, x0 will filled with :math:`\frac{1}{n_1 n_2})`.
    :param max_iter: (default: 50) max number of iterations in IPFP.
                     More iterations will be lead to more accurate result, at the cost of increased inference time.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1 \times n_2)` the solved matching matrix

    .. note::
        Either ``n1`` or ``n1max`` should be specified because it cannot be inferred from the input tensor size.
        Same for ``n2`` or ``n2max``.

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded and all elements in ``n1`` are equal, all in ``n2`` are equal.

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. note::
        This solver is non-differentiable. The output is a discrete matching matrix (i.e. permutation matrix).

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'
            >>> np.random.seed(1)
    
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
            >>> A1 = np.random.rand(batch_size, 4, 4)
            >>> A2 = np.matmul(np.matmul(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> n1 = n2 = np.repeat([4], batch_size)
    
            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
            # Solve by IPFP
            >>> X = pygm.ipfp(K, n1, n2)
            >>> X[0]
            array([[0., 0., 0., 1.],
                   [0., 0., 1., 0.],
                   [1., 0., 0., 0.],
                   [0., 1., 0., 0.]])
    
            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)
    
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = torch.zeros(batch_size, 4, 4)
            >>> X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
            >>> A1 = torch.rand(batch_size, 4, 4)
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = torch.tensor([4] * batch_size)
            >>> n2 = torch.tensor([4] * batch_size)
    
            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    
            # Solve by IPFP
            >>> X = pygm.ipfp(K, n1, n2)
            >>> X[0]
            tensor([[0., 1., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.],
                    [1., 0., 0., 0.]])
    
            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            tensor(1.)

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'
            >>> _ = paddle.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = paddle.rand((batch_size, 4, 4))
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> n1 = paddle.to_tensor([4] * batch_size)
            >>> n2 = paddle.to_tensor([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by IPFP
            >>> X = pygm.ipfp(K, n1, n2)
            >>> X[0]
            Tensor(shape=[4, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [[0., 1., 0., 0.],
                    [0., 0., 0., 1.],
                    [0., 0., 1., 0.],
                    [1., 0., 0., 0.]])

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [1.])

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'jittor'
            >>> _ = jt.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = jt.zeros((batch_size, 4, 4))
            >>> X_gt[:, jt.arange(0, 4, dtype=jt.int64), jt.randperm(4)] = 1
            >>> A1 = jt.rand(batch_size, 4, 4)
            >>> A2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = jt.Var([4] * batch_size)
            >>> n2 = jt.Var([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by IPFP
            >>> X = pygm.ipfp(K, n1, n2)
            >>> X[0]
            jt.Var([[1. 0. 0. 0.]
                    [0. 0. 1. 0.]
                    [0. 0. 0. 1.]
                    [0. 1. 0. 0.]], dtype=float32)

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            jt.Var([1.], dtype=float32)

    .. dropdown:: MindSpore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'mindspore'
            >>> _ = mindspore.set_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = mindspore.numpy.zeros((batch_size, 4, 4))
            >>> X_gt[:, mindspore.numpy.arange(0, 4, dtype=mindspore.int64), mindspore.ops.Randperm(4)(mindspore.Tensor([4], dtype=mindspore.int32))] = 1
            >>> A1 = mindspore.numpy.rand((batch_size, 4, 4))
            >>> A2 = mindspore.ops.BatchMatMul()(mindspore.ops.BatchMatMul()(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> n1 = mindspore.Tensor([4] * batch_size)
            >>> n2 = mindspore.Tensor([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by IPFP
            >>> X = pygm.ipfp(K, n1, n2)
            >>> X[0]
            [[1. 0. 0. 0.]
             [0. 0. 0. 1.]
             [0. 0. 1. 0.]
             [0. 1. 0. 0.]]

            # Accuracy
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum()
            1.0

    .. dropdown:: Tensorflow Example

        ::

            >>> import tensorflow as tf
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'tensorflow'
            >>> _ = tf.random.set_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = tf.Variable(tf.zeros([batch_size, 4, 4]))
            >>> indices = tf.stack([tf.range(4),tf.random.shuffle(tf.range(4))], axis=1)
            >>> updates = tf.ones([4])
            >>> for i in range(batch_size):
            ...     _ = X_gt[i].assign(tf.tensor_scatter_nd_update(X_gt[i], indices, updates))
            >>> A1 = tf.random.uniform([batch_size, 4, 4])
            >>> A2 = tf.matmul(tf.matmul(tf.transpose(X_gt, perm=[0, 2, 1]), A1), X_gt)
            >>> n1 = n2 = tf.constant([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by IPFP
            >>> X = pygm.ipfp(K, n1, n2)
            >>> X[0]
            <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
            array([[0., 0., 1., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 0., 1.],
                   [1., 0., 0., 0.]], dtype=float32)>

            # Accuracy
            >>> tf.reduce_sum((pygm.hungarian(X) * X_gt)) / tf.reduce_sum(X_gt)
            <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

    .. note::
        If you find this graph matching solver useful in your research, please cite:

        ::

            @article{ipfp,
              title={An integer projected fixed point method for graph matching and map inference},
              author={Leordeanu, Marius and Hebert, Martial and Sukthankar, Rahul},
              journal={Advances in neural information processing systems},
              volume={22},
              year={2009}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(K, 'K', backend)
    if _check_shape(K, 2, backend):
        K = _unsqueeze(K, 0, backend)
        non_batched_input = True
        if type(n1) is int and n1max is None:
            n1max = n1
            n1 = None
        if type(n2) is int and n2max is None:
            n2max = n2
            n2 = None
    elif _check_shape(K, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument K is expected to be 2-dimensional or 3-dimensional, got '
                         f'K:{len(_get_shape(K, backend))}dims!')
    __check_gm_arguments(n1, n2, n1max, n2max)

    args = (K, n1, n2, n1max, n2max, x0, max_iter)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.ipfp
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def astar(feat1, feat2, A1, A2, n1=None, n2=None, channel=None, beam_width=0, backend=None):
    r"""
    A\* (A-star) solver for graph matching (Lawler's QAP).
    The **A\*** solver was originally proposed to solve the graph edit distance (GED) problem. It finds the optimal
    matching between two graphs (which is equivalent to an edit path between them) by priority search.

    Here we implement the A\* solver with Hungarian heuristic. See the following paper for more details:
    `An Exact Graph Edit Distance Algorithm for Solving Pattern Recognition Problems <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=5e3a679c3bdf5c5d72c5cc91c8cc9d019a500842>`_

    :param feat1: :math:`(b\times n_1 \times d)` input feature of graph1
    :param feat2: :math:`(b\times n_2 \times d)` input feature of graph2
    :param A1: :math:`(b\times n_1 \times n_1)` input adjacency matrix of graph1
    :param A2: :math:`(b\times n_2 \times n_2)` input adjacency matrix of graph2
    :param n1: :math:`(b)` number of nodes in graph1. Optional if all equal to :math:`n_1`
    :param n2: :math:`(b)` number of nodes in graph2. Optional if all equal to :math:`n_2`
    :param channel: (default: None)  Channel size of the input layer. If given, it must match the feature dimension (d) of feat1, feat2. 
        If not given, it will be defined by the feature dimension (d) of feat1, feat2. 
        Ignored if the network object isgiven (ignored if network!=None)
    :param beam_width: (default: 0) Size of beam-search witdh (0 = no beam).
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix

    .. warning::
        If ``beam_width==0``, the algorithm will find the optimal solution, and it may take a very long time.
        You may set a ``beam_width`` to lower the size of the search tree, at the cost of losing the optimal guarantee.

    .. note::
        Graph matching problem (Lawler's QAP) and graph edit distance problem are two sides of the same coin. If you
        want to transform your graph edit distance problem into Lawler's QAP, first compute the edit cost of each
        node pairs and edge pairs, then place them to the right places to get a **cost** matrix. Finally, build
        the affinity matrix by :math:`-1\times` cost matrix.

        You can get your customized cost/affinity matrix by :func:`~pygmtools.utils.build_aff_mat`.

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: PyTorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)
            
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> nodes_num = 4
            >>> channel = 36

            >>> X_gt = torch.zeros(batch_size, nodes_num, nodes_num)
            >>> X_gt[:, torch.arange(0, nodes_num, dtype=torch.int64), torch.randperm(nodes_num)] = 1
            >>> A1 = 1. * (torch.rand(batch_size, nodes_num, nodes_num) > 0.5)
            >>> torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> feat1 = torch.rand(batch_size, nodes_num, channel) - 0.5
            >>> feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = torch.tensor([nodes_num] * batch_size)

            # Match by A* (load pretrained model)
            >>> X = pygm.astar(feat1, feat2, A1, A2, n1, n2)
            >>> (X * X_gt).sum() / X_gt.sum()# accuracy
            tensor(1.)
            
            # This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.
            >>> part_f1 = feat1[0]
            >>> part_f2 = feat2[0]
            >>> part_A1 = A1[0]
            >>> part_A2 = A2[0]
            >>> part_X_gt = X_gt[0]
            >>> part_X = pygm.astar(part_f1, part_f2, part_A1, part_A2)
            
            >>> part_X.shape
            torch.Size([4, 4])
            
            >>> (part_X * part_X_gt).sum() / part_X_gt.sum()# accuracy
            tensor(1.)

    .. note::
        If you find this graph matching solver useful in your research, please cite:

        ::

            @inproceedings{astar,
              title={Speeding Up Graph Edit Distance Computation with a Bipartite Heuristic.},
              author={Riesen, Kaspar and Fankhauser, Stefan and Bunke, Horst},
              booktitle={Mining and Learning with Graphs},
              pages={21--24},
              year={2007}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if feat1 is not None: # if feat1 is None, this function skips the forward pass and only returns a network object
        for _ in (feat1, feat2, A1, A2):
            _check_data_type(_, backend)

        if all([_check_shape(_, 2, backend) for _ in (feat1, feat2, A1, A2)]):
            feat1, feat2, A1, A2 = [_unsqueeze(_, 0, backend) for _ in (feat1, feat2, A1, A2)]
            if type(n1) is int: n1 = from_numpy(np.array([n1]), backend=backend)
            if type(n2) is int: n2 = from_numpy(np.array([n2]), backend=backend)
            non_batched_input = True
        elif all([_check_shape(_, 3, backend) for _ in (feat1, feat2, A1, A2)]):
            non_batched_input = False
        else:
            raise ValueError(
                f'the input arguments feat1, feat2, A1, A2 are expected to be all 2-dimensional or 3-dimensional, got '
                f'feat1:{len(_get_shape(feat1, backend))}dims, feat2:{len(_get_shape(feat2, backend))}dims, '
                f'A1:{len(_get_shape(A1, backend))}dims, A2:{len(_get_shape(A2, backend))}dims!')

        if not (_get_shape(feat1, backend)[0] == _get_shape(feat2, backend)[0] == _get_shape(A1, backend)[0] == _get_shape(A2, backend)[0])\
                or not (_get_shape(feat1, backend)[1] == _get_shape(A1, backend)[1] == _get_shape(A1, backend)[2])\
                or not (_get_shape(feat2, backend)[1] == _get_shape(A2, backend)[1] == _get_shape(A2, backend)[2])\
                or not (_get_shape(feat1, backend)[2] == _get_shape(feat2, backend)[2]):
            raise ValueError(
                f'the input dimensions do not match. Got feat1:{_get_shape(feat1, backend)}, '
                f'feat2:{_get_shape(feat2, backend)}, A1:{_get_shape(A1, backend)}, A2:{_get_shape(A2, backend)}!')
    if n1 is not None: _check_data_type(n1, 'n1', backend)
    if n2 is not None: _check_data_type(n2, 'n2', backend)

    args = (feat1, feat2, A1, A2, n1, n2, channel, beam_width)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.astar
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    return match_mat


def __check_gm_arguments(n1, n2, n1max, n2max):
    if n1 is None and n1max is None:
        raise ValueError('at least one of the following arguments are required: n1 and n1max.')
    if n2 is None and n2max is None:
        raise ValueError('at least one of the following arguments are required: n2 and n2max.')
