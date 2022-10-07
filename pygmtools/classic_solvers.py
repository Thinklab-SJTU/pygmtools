r"""
Classic (learning-free) **two-graph matching** solvers. These two-graph matching solvers are recommended to solve
matching problems with two explicit graphs, or problems formulated as Quadratic Assignment Problem (QAP).

The two-graph matching problem considers both nodes and edges, formulated as a QAP:

.. math::

    &\max_{\mathbf{X}} \ \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})\\
    s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}
"""

import importlib
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type


def sm(K, n1=None, n2=None, n1max=None, n2max=None, x0=None,
       max_iter: int=50,
       backend=None):
    r"""
    Spectral Graph Matching solver for graph matching (QAP).
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
    _check_data_type(K, backend)
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
    except ModuleNotFoundError and AttributeError:
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
    Reweighted Random Walk Matching (RRWM) solver for graph matching (QAP). This algorithm is implemented by power
    iteration with Sinkhorn reweighted jumps.

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
    _check_data_type(K, backend)
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
    except ModuleNotFoundError and AttributeError:
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
    Integer Projected Fixed Point (IPFP) method for graph matching (QAP).

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
    _check_data_type(K, backend)
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
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def __check_gm_arguments(n1, n2, n1max, n2max):
    if n1 is None and n1max is None:
        raise ValueError('at least one of the following arguments are required: n1 and n1max.')
    if n2 is None and n2max is None:
        raise ValueError('at least one of the following arguments are required: n2 and n2max.')
