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
              matrix of graph ``i`` and graph ``j``
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

    Example for Pytorch backend::

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
        """
    if backend is None:
        backend = pygmtools.BACKEND
    # check the correctness of input
    _check_data_type(K, backend)
    K_shape = _get_shape(K, backend)
    assert len(K_shape) == 4 and K_shape[0] == K_shape[1] and K_shape[2] == K_shape[3], \
        f"Unsupported input data shape: got K {K_shape}"
    num_graph, aff_size = K_shape[0], K_shape[2]
    num_node = int(math.sqrt(aff_size))
    assert num_node ** 2 == aff_size, "The input affinity matrix is not supported. Please note that this function " \
                                      "does not support matching with outliers or partial matching."
    if x0 is not None:
        _check_data_type(x0, backend)
        x0_shape = _get_shape(x0, backend)
        assert len(x0_shape) == 4 and num_graph == x0_shape[0] == x0_shape[1] and num_node == x0_shape[2] == x0_shape[3], \
            f"Unsupported input data shape: got K {K_shape} x0 {x0_shape}"
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
