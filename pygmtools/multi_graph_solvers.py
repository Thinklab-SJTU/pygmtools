r"""
Classic (learning-free) **multi-graph matching** solvers. These multi-graph matching solvers are recommended to solve
the joint matching problem of multiple graphs.
"""

import functools
import importlib
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type
import math


def cao(K, x0=None, qap_solver=None,
        mode='accu',
        max_iter=6, lambda_init=0.3, lambda_step=1.1, lambda_max=1.0, iter_boost=2,
        backend=None):
    r"""
    Composition based Affinity Optimization (CAO) solver for multi-graph matching. This solver builds a supergraph for
    matching update to incorporate the two aspects by optimizing the affinity score, meanwhile gradually
    infusing the consistency.

    Each update step is described as follows:

    .. math::

        \arg \max_{k} (1-\lambda) J(\mathbf{X}_{ik} \mathbf{X}_{kj}) + \lambda C_p(\mathbf{X}_{ik} \mathbf{X}_{kj})

    where :math:`J(\mathbf{X}_{ik} \mathbf{X}_{kj})` is the objective score, and
    :math:`C_p(\mathbf{X}_{ik} \mathbf{X}_{kj})` measures a consistency score compared to other matchings. These two
    terms are balanced by :math:`\lambda`, and :math:`\lambda` starts from a smaller number and gradually grows.

    :param K: :math:`(m\times m \times n^2 \times n^2)` the input affinity matrix, where ``K[i,j]`` is the affinity
              matrix of graph ``i`` and graph ``j`` (:math:`m`: number of nodes)
    :param x0: (optional) :math:`(m\times m \times n \times n)` the initial two-graph matching result, where ``X[i,j]``
               is the matching matrix result of graph ``i`` and graph ``j``. If this argument is not given,
               ``qap_solver`` will be used to compute the two-graph matching result.
    :param qap_solver: (default: pygm.rrwm) a function object that accepts a batched affinity matrix and returns the
                       matching matrices. It is suggested to use ``functools.partial`` and the QAP solvers provided in
                       the :mod:`~pygmtools.classic_solvers` module (see examples below).
    :param mode: (default: ``'accu'``) the operation mode of this algorithm. Options: ``'accu', 'c', 'fast', 'pc'``,
                 where ``'accu'`` is equivalent to ``'c'`` (accurate version) and ``'fast'`` is equivalent to ``'pc'``
                 (fast version).
    :param max_iter: (default: 6) max number of iterations
    :param lambda_init: (default: 0.3) initial value of :math:`\lambda`, with :math:`\lambda\in[0,1]`
    :param lambda_step: (default: 1.1) the increase step size of :math:`\lambda`, updated by ``lambda = step * lambda``
    :param lambda_max: (default: 1.0) the max value of lambda
    :param iter_boost: (default: 2) to boost the convergence of the CAO algorithm, :math:`\lambda` will be forced to
                       update every ``iter_boost`` iterations.
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(m\times m \times n \times n)` the multi-graph matching result

    .. note::

        The input graphs must have the same number of nodes for this algorithm to work correctly.

    .. note::

       Multi-graph matching methods process all graphs at once and do not support the additional batch dimension. Please
       note that this behavior is different from two-graph matching solvers in :mod:`~pygmtools.classic_solvers`.

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)

            # Generate 10 isomorphic graphs
            >>> graph_num = 10
            >>> As, X_gt = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=10)
            >>> As_1, As_2 = [], []
            >>> for i in range(graph_num):
            ...     for j in range(graph_num):
            ...         As_1.append(As[i])
            ...         As_2.append(As[j])
            >>> As_1 = torch.stack(As_1, dim=0)
            >>> As_2 = torch.stack(As_2, dim=0)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(As_1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(As_2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, None, None, None, None, edge_aff_fn=gaussian_aff)
            >>> K = K.reshape(graph_num, graph_num, 4*4, 4*4)
            >>> K.shape
            torch.Size([10, 10, 16, 16])

            # Solve the multi-matching problem
            >>> X = pygm.cao(K)
            >>> (X * X_gt).sum() / X_gt.sum()
            tensor(1.)

            # Use the IPFP solver for two-graph matching
            >>> ipfp_func = functools.partial(pygmtools.ipfp, n1max=4, n2max=4)
            >>> X = pygm.cao(K, qap_solver=ipfp_func)
            >>> (X * X_gt).sum() / X_gt.sum()
            tensor(1.)

            # Run the faster version of CAO algorithm
            >>> X = pygm.cao(K, mode='fast')
            >>> (X * X_gt).sum() / X_gt.sum()
            tensor(1.)

    .. note::
        If you find this graph matching solver useful in your research, please cite:

        ::

            @article{cao,
              title={Multi-graph matching via affinity optimization with graduated consistency regularization},
              author={Yan, Junchi and Cho, Minsu and Zha, Hongyuan and Yang, Xiaokang and Chu, Stephen M},
              journal={IEEE transactions on pattern analysis and machine intelligence},
              volume={38},
              number={6},
              pages={1228--1242},
              year={2015},
              publisher={IEEE}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    # check the correctness of input
    _check_data_type(K, backend)
    K_shape = _get_shape(K, backend)
    if not (len(K_shape) == 4 and K_shape[0] == K_shape[1] and K_shape[2] == K_shape[3]):
        raise ValueError(f"Unsupported input data shape: got K {K_shape}")
    num_graph, aff_size = K_shape[0], K_shape[2]
    num_node = int(math.sqrt(aff_size))
    if not num_node ** 2 == aff_size:
        raise ValueError("The input affinity matrix is not supported. Please note that this function "
                         "does not support matching with outliers or partial matching.")
    if not 0 <= lambda_init <= 1: raise ValueError(f"lambda_init must be in [0, 1], got lambda_init={lambda_init}")
    if not 0 <= lambda_max <= 1: raise ValueError(f"lambda_max must be in [0, 1], got lambda_max={lambda_max}")
    if not lambda_step > 1: raise ValueError(f"lambda_step must be >1, got lambda_step={lambda_step}")
    if x0 is not None:
        _check_data_type(x0, backend)
        x0_shape = _get_shape(x0, backend)
        if not len(x0_shape) == 4 and num_graph == x0_shape[0] == x0_shape[1] and num_node == x0_shape[2] == x0_shape[3]:
            raise ValueError(f"Unsupported input data shape: got K {K_shape} x0 {x0_shape}")
    else:
        if qap_solver is None:
            qap_solver = functools.partial(pygmtools.rrwm, n1max=num_node, n2max=num_node, backend=backend)
        x0 = qap_solver(K.reshape(num_graph ** 2, aff_size, aff_size))
        x0 = pygmtools.hungarian(x0, backend=backend)
        x0 = x0.reshape(num_graph, num_graph, num_node, num_node)

    args = (K, x0, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        if mode in ['pc', 'fast']:
            fn = mod.cao_fast_solver
        elif mode in ['c', 'accu']:
            fn = mod.cao_solver
        else:
            raise ValueError("Unknown value of mode: supported values ['c', 'accu', 'pc', 'fast']")
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    return fn(*args)


def mgm_floyd(K, x0=None, qap_solver=None,
              mode='accu',
              param_lambda=0.2,
              backend=None):
    r"""
    Multi-Graph Matching based on Floyd shortest path algorithm. A supergraph is considered by regarding each input
    graph as a node, and the matching between graphs are regraded as edges in the supergraph. Floyd algorithm is used
    to discover a shortest path on this supergraph for matching update.

    The length of edges on the supergraph is described as follows:

    .. math::

        \arg \max_{k} (1-\lambda) J(\mathbf{X}_{ik} \mathbf{X}_{kj}) + \lambda C_p(\mathbf{X}_{ik} \mathbf{X}_{kj})

    where :math:`J(\mathbf{X}_{ik} \mathbf{X}_{kj})` is the objective score, and
    :math:`C_p(\mathbf{X}_{ik} \mathbf{X}_{kj})` measures a consistency score compared to other matchings. These two
    terms are balanced by :math:`\lambda`.

    :param K: :math:`(m\times m \times n^2 \times n^2)` the input affinity matrix, where ``K[i,j]`` is the affinity
              matrix of graph ``i`` and graph ``j`` (:math:`m`: number of nodes)
    :param x0: (optional) :math:`(m\times m \times n \times n)` the initial two-graph matching result, where ``X[i,j]``
               is the matching matrix result of graph ``i`` and graph ``j``. If this argument is not given,
               ``qap_solver`` will be used to compute the two-graph matching result.
    :param qap_solver: (default: pygm.rrwm) a function object that accepts a batched affinity matrix and returns the
                       matching matrices. It is suggested to use ``functools.partial`` and the QAP solvers provided in
                       the :mod:`~pygmtools.classic_solvers` module (see examples below).
    :param mode: (default: ``'accu'``) the operation mode of this algorithm. Options: ``'accu', 'c', 'fast', 'pc'``,
                 where ``'accu'`` is equivalent to ``'c'`` (accurate version) and ``'fast'`` is equivalent to ``'pc'``
                 (fast version).
    :param param_lambda: (default: 0.3) value of :math:`\lambda`, with :math:`\lambda\in[0,1]`
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(m\times m \times n \times n)` the multi-graph matching result

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)

            # Generate 10 isomorphic graphs
            >>> graph_num = 10
            >>> As, X_gt = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=10)
            >>> As_1, As_2 = [], []
            >>> for i in range(graph_num):
            ...     for j in range(graph_num):
            ...         As_1.append(As[i])
            ...         As_2.append(As[j])
            >>> As_1 = torch.stack(As_1, dim=0)
            >>> As_2 = torch.stack(As_2, dim=0)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(As_1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(As_2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, None, None, None, None, edge_aff_fn=gaussian_aff)
            >>> K = K.reshape(graph_num, graph_num, 4*4, 4*4)
            >>> K.shape
            torch.Size([10, 10, 16, 16])

            # Solve the multi-matching problem
            >>> X = pygm.mgm_floyd(K)
            >>> (X * X_gt).sum() / X_gt.sum()
            tensor(1.)

            # Use the IPFP solver for two-graph matching
            >>> ipfp_func = functools.partial(pygmtools.ipfp, n1max=4, n2max=4)
            >>> X = pygm.mgm_floyd(K, qap_solver=ipfp_func)
            >>> (X * X_gt).sum() / X_gt.sum()
            tensor(1.)

            # Run the faster version of CAO algorithm
            >>> X = pygm.mgm_floyd(K, mode='fast')
            >>> (X * X_gt).sum() / X_gt.sum()
            tensor(1.)

    .. note::

        If you find this graph matching solver useful in your research, please cite:

        ::

            @article{mgm_floyd,
              title={Unifying offline and online multi-graph matching via finding shortest paths on supergraph},
              author={Jiang, Zetian and Wang, Tianzhe and Yan, Junchi},
              journal={IEEE transactions on pattern analysis and machine intelligence},
              volume={43},
              number={10},
              pages={3648--3663},
              year={2020},
              publisher={IEEE}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    # check the correctness of input
    _check_data_type(K, backend)
    K_shape = _get_shape(K, backend)
    if not (len(K_shape) == 4 and K_shape[0] == K_shape[1] and K_shape[2] == K_shape[3]):
        raise ValueError(f"Unsupported input data shape: got K {K_shape}")
    num_graph, aff_size = K_shape[0], K_shape[2]
    num_node = int(math.sqrt(aff_size))
    if not num_node ** 2 == aff_size:
        raise ValueError("The input affinity matrix is not supported. Please note that this function "
                         "does not support matching with outliers or partial matching.")
    if not 0 <= param_lambda <= 1: raise ValueError(f"param_lambda must be in [0, 1], got param_lambda={param_lambda}")
    if x0 is not None:
        _check_data_type(x0, backend)
        x0_shape = _get_shape(x0, backend)
        if not len(x0_shape) == 4 and num_graph == x0_shape[0] == x0_shape[1] and num_node == x0_shape[2] == x0_shape[3]:
            raise ValueError(f"Unsupported input data shape: got K {K_shape} x0 {x0_shape}")
    else:
        if qap_solver is None:
            qap_solver = functools.partial(pygmtools.rrwm, n1max=num_node, n2max=num_node, backend=backend)
        x0 = qap_solver(K.reshape(num_graph ** 2, aff_size, aff_size))
        x0 = pygmtools.hungarian(x0, backend=backend)
        x0 = x0.reshape(num_graph, num_graph, num_node, num_node)

    args = (K, x0, num_graph, num_node, param_lambda)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        if mode in ['pc', 'fast']:
            fn = mod.mgm_floyd_fast_solver
        elif mode in ['c', 'accu']:
            fn = mod.mgm_floyd_solver
        else:
            raise ValueError("Unknown value of mode: supported values ['c', 'accu', 'pc', 'fast']")
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    return fn(*args)


def gamgm(A, W,
          ns=None, n_univ=None, U0=None,
          sk_init_tau=0.5, sk_min_tau=0.1, sk_gamma=0.8, sk_iter=20, max_iter=100, param_lambda=1.,
          converge_thresh=1e-5, outlier_thresh=-1, bb_smooth=0.1,
          verbose=False,
          backend=None):
    r"""
    Graduated Assignment-based multi-graph matching solver. Graduated assignment is a classic approach for hard
    assignment problems like graph matching, based on graduated annealing of Sinkhorn's temperature :math:`\tau` to
    enforce the matching constraint.

    The objective score is described as

    .. math::

        \max_{\mathbf{X}_{i,j}, i,j\in [m]} \ \sum_{i,j\in [m]} \left( \lambda \ \mathrm{tr}(\mathbf{X}_{ij}^\top \mathbf{A}_{i} \mathbf{X}_{ij} \mathbf{A}_{j}) + \mathrm{tr}(\mathbf{X}_{ij}^\top \mathbf{W}_{ij})\right)

    Once the algorithm converges at a fixed :math:`\tau` value, :math:`\tau` shrinks as:

    .. math::

        \tau = \tau \times \gamma

    and the iteration continues. At last, Hungarian algorithm is applied to ensure the result is a permutation matrix.

    .. note::

        This algorithm is based on the Koopmans-Beckmann's QAP formulation and you should input the adjacency matrices
        ``A`` and node-wise similarity matrices ``W`` instead of the affinity matrices.

    :param A: :math:`(m\times n \times n)` the adjacency matrix (:math:`m`: number of nodes).
              The graphs may have different number of nodes (specified by the ``ns`` argument).
    :param W: :math:`(m\times m \times n \times n)` the node-wise similarity matrix, where ``W[i,j]`` is the similarity
              matrix
    :param ns: (optional) :math:`(m)` the number of nodes. If not given, it will be inferred based on the size of ``A``.
    :param n_univ: (optional) the size of the universe node set. If not given, it will be the largest number of nodes.
    :param U0: (optional) the initial multi-graph matching result. If not given, it will be randomly initialized.
    :param sk_init_tau: (default: 0.05) initial value of :math:`\tau` for Sinkhorn algorithm
    :param sk_min_tau: (default: 1.0e-3) minimal value of :math:`\tau` for Sinkhorn algorithm
    :param sk_gamma: (default: 0.8) the shrinking parameter of :math:`\tau`: :math:`\tau = \tau \times \gamma`
    :param sk_iter: (default: 200) max number of iterations for Sinkhorn algorithm
    :param max_iter: (default: 1000) max number of iterations for graduated assignment
    :param param_lambda: (default: 1) the weight :math:`\lambda` of the quadratic term
    :param converge_thresh: (default: 1e-5) if the Frobenius norm of the change of U is smaller than this, the iteration
                            is stopped.
    :param outlier_thresh: (default: -1) if > 0, pairs with node+edge similarity score smaller than this threshold will
                           be discarded. This threshold is designed to handle outliers.
    :param bb_smooth: (default: 0.1) the black-box differentiation smoothing parameter.
    :param verbose: (default: False) print verbose information for parameter tuning
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: the multi-graph matching result (a :mod:`~pygmtools.utils.MultiMatchingResult` object)

    .. note::

        In PyTorch backend, this function is differentiable through the black-box trick. See the following paper for
        details:

        ::

            Vlastelica M, Paulus A., Differentiation of Blackbox Combinatorial Solvers, ICLR 2020

        If you want to disable this differentiable feature, please detach the input tensors from the computational
        graph.

    .. note::

        Setting ``verbose=True`` may help you tune the parameters.

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> import itertools
            >>> import time
            >>> pygm.BACKEND = 'pytorch'
            >>> _ = torch.manual_seed(1)

            # Generate 10 isomorphic graphs
            >>> graph_num = 10
            >>> As, X_gt, Fs = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=10, node_feat_dim=20)

            # Compute node-wise similarity by inner-product and Sinkhorn
            >>> W = torch.matmul(Fs.unsqueeze(1), Fs.transpose(1, 2).unsqueeze(0))
            >>> W = pygm.sinkhorn(W.reshape(graph_num ** 2, 4, 4)).reshape(graph_num, graph_num, 4, 4)

            # Solve the multi-matching problem
            >>> X = pygm.gamgm(As, W)
            >>> matched = 0
            >>> for i, j in itertools.product(range(graph_num), repeat=2):
            ...     matched += (X[i,j] * X_gt[i,j]).sum()
            >>> acc = matched / X_gt.sum()
            >>> acc
            tensor(1.)

            # This function is differentiable by the black-box trick
            >>> W.requires_grad_(True)  # tell PyTorch to track the gradients
            >>> X = pygm.gamgm(As, W)
            >>> matched = 0
            >>> for i, j in itertools.product(range(graph_num), repeat=2):
            ...     matched += (X[i,j] * X_gt[i,j]).sum()
            >>> acc = matched / X_gt.sum()

            # Backward pass via black-box trick
            >>> acc.backward()
            >>> torch.sum(W.grad != 0)
            tensor(128)

            # This function supports graphs with different nodes (also known as partial matching)
            # In the following we ignore the last node from the last 5 graphs
            >>> ns = torch.tensor([4, 4, 4, 4, 4, 3, 3, 3, 3, 3], dtype=torch.int)
            >>> for i in range(graph_num):
            ...     As[i, ns[i]:, :] = 0
            ...     As[i, :, ns[i]:] = 0
            >>> for i, j in itertools.product(range(graph_num), repeat=2):
            ...     X_gt[i, j, ns[i]:, :] = 0
            ...     X_gt[i, j, :, ns[j]:] = 0
            ...     W[i, j, ns[i]:, :] = 0
            ...     W[i, j, :, ns[j]:] = 0
            >>> W = W.detach() # detach tensor if gradient is not needed

            # Partial matching is challenging and the following parameters are carefully tuned
            >>> X = pygm.gamgm(As, W, ns, n_univ=4, sk_init_tau=.1, sk_min_tau=0.01, param_lambda=0.3)

            # Check the partial matching result
            >>> matched = 0
            >>> for i, j in itertools.product(range(graph_num), repeat=2):
            ...     matched += (X[i,j] * X_gt[i, j, :ns[i], :ns[j]]).sum()
            >>> matched / X_gt.sum()
            tensor(1.)

    .. note::

        If you find this graph matching solver useful in your research, please cite:

        ::

            @article{gamgm1,
              title={Graduated assignment algorithm for multiple graph matching based on a common labeling},
              author={Sol{\'e}-Ribalta, Albert and Serratosa, Francesc},
              journal={International Journal of Pattern Recognition and Artificial Intelligence},
              volume={27},
              number={01},
              pages={1350001},
              year={2013},
              publisher={World Scientific}
            }

            @article{gamgm2,
              title={Graduated assignment for joint multi-graph matching and clustering with application to unsupervised graph matching network learning},
              author={Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
              journal={Advances in Neural Information Processing Systems},
              volume={33},
              pages={19908--19919},
              year={2020}
            }

        This algorithm is originally proposed by paper ``gamgm1``, and further improved by paper ``gamgm2`` to fit
        modern computing architectures like GPU.
    """
    if backend is None:
        backend = pygmtools.BACKEND
    # check the correctness of input
    _check_data_type(A, backend)
    A_shape = _get_shape(A, backend)
    if not (len(A_shape) == 3 and A_shape[1] == A_shape[2]):
        raise ValueError(f"Unsupported input data shape: got A {A_shape}")
    num_graph, max_node = A_shape[0], A_shape[1]
    _check_data_type(W, backend)
    W_shape = _get_shape(W, backend)
    if not (len(W_shape) == 4 and W_shape[0] == W_shape[1] == num_graph and W_shape[2] == W_shape[3] == max_node):
        raise ValueError(f"Unsupported input data shape: got A {A_shape}, W {W_shape}")
    if ns is not None:
        _check_data_type(ns, backend)
        ns_shape = _get_shape(ns, backend)
        if not (len(ns_shape) == 1 and ns_shape[0] == num_graph):
            raise ValueError(f"The size of ns mismatches the sizes of A and W: got ns {ns_shape}, A {A_shape}, W {W_shape}")
    if n_univ is None:
        n_univ = max_node
    if U0 is not None:
        _check_data_type(U0, backend)
    if not sk_init_tau > 0: raise ValueError(f"sk_init_tau must be >0, got sk_init_tau={sk_init_tau}")
    if not sk_min_tau > 0: raise ValueError(f"sk_min_tau must be >0, got sk_min_tau={sk_min_tau}")
    if not 0 < sk_gamma < 1: raise ValueError(f"sk_gamma must be in (0, 1), got sk_gamma={sk_gamma}")
    if not 0 < bb_smooth < 1: raise ValueError(f"bb_smooth must be in (0, 1), got bb_smooth={bb_smooth}")

    args = (A, W, ns, n_univ, U0, sk_init_tau, sk_min_tau, sk_gamma, sk_iter, max_iter, param_lambda,
            converge_thresh, outlier_thresh, bb_smooth, verbose)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.gamgm
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    return fn(*args)
