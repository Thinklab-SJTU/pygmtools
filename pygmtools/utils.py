import functools
import importlib

import pygmtools

NOT_IMPLEMENTED_MSG = \
    'The backend function for {} is not implemented. It will be truly appreciated if you could share your implementation ' \
    'with the community! See our Github: https://github.com/Thinklab-SJTU/pygmtools'


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

    Example for numpy backend::

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

    Example for Pytorch backend::

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
    """
    if backend is None:
        backend = pygmtools.BACKEND

    # check the correctness of input
    batch_size = None
    if node_feat1 is not None or node_feat2 is not None:
        assert all([_ is not None for _ in (node_feat1, node_feat2)]), \
            'The following arguments must all be given if you want to compute node-wise affinity: ' \
            'node_feat1, node_feat2'
        _check_data_type(node_feat1, backend)
        _check_data_type(node_feat2, backend)
        assert all([_check_shape(_, 3, backend) for _ in (node_feat1, node_feat2)]), \
            f'The shape of the following tensors are illegal, expected 3-dimensional, ' \
            f'got node_feat1={len(_get_shape(node_feat1))}d; node_feat2={len(_get_shape(node_feat2))}d!'
        if batch_size is None:
            batch_size = _get_shape(node_feat1)[0]
        assert _get_shape(node_feat1)[0] == _get_shape(node_feat2)[0] == batch_size, 'batch size mismatch'
    if edge_feat1 is not None or edge_feat2 is not None:
        assert all([_ is not None for _ in (edge_feat1, edge_feat2, connectivity1, connectivity2)]), \
            'The following arguments must all be given if you want to compute edge-wise affinity: ' \
            'edge_feat1, edge_feat2, connectivity1, connectivity2'
        assert all([_check_shape(_, 3, backend) for _ in (edge_feat1, edge_feat2, connectivity1, connectivity2)]), \
            f'The shape of the following tensors are illegal, expected 3-dimensional, ' \
            f'got edge_feat1:{len(_get_shape(edge_feat1))}d; edge_feat2:{len(_get_shape(edge_feat2))}d; ' \
            f'connectivity1:{len(_get_shape(connectivity1))}d; connectivity2:{len(_get_shape(connectivity2))}d!'
        assert _get_shape(connectivity1)[2] == _get_shape(connectivity1)[2] == 2, \
            'the 3rd dimension of connectivity1, connectivity2 must be 2-dimensional'
        if batch_size is None:
            batch_size = _get_shape(edge_feat1)[0]
        assert _get_shape(edge_feat1)[0] == _get_shape(edge_feat2)[0] == _get_shape(connectivity1)[0] == \
               _get_shape(connectivity2)[0] == batch_size, 'batch size mismatch'

    # assign the default affinity functions if not given
    if node_aff_fn is None:
        node_aff_fn = functools.partial(inner_prod_aff_fn, backend=backend)
    if edge_aff_fn is None:
        edge_aff_fn = functools.partial(inner_prod_aff_fn, backend=backend)

    node_aff = node_aff_fn(node_feat1, node_feat2) if node_feat1 is not None else None
    edge_aff = edge_aff_fn(edge_feat1, edge_feat2) if edge_feat1 is not None else None

    return _aff_mat_from_node_edge_aff(node_aff, edge_aff, connectivity1, connectivity2, n1, n2, ne1, ne2, backend=backend)


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
        return mod.inner_prod_aff_fn(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


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
        return mod.gaussian_aff_fn(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


def build_batch(input, return_ori_dim=False, backend=None):
    r"""
    Build a batched tensor from a list of tensors. If the list of tensors are with different sizes of dimensions, it
    will be padded to the largest dimension.

    The batched tensor and the number of original dimensions will be returned.

    :param input: list of input tensors
    :param return_ori_dim: (default: False) return the original dimension
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: batched tensor, (if ``return_ori_dim=True``) number of original dimensions...

    Example for numpy backend::

        >>> import numpy as np
        >>> import pygmtools as pygm
        >>> pygm.BACKEND = 'numpy'

        # batched adjacency matrices
        >>> A1 = np.random.rand(4, 4)
        >>> A2 = np.random.rand(5, 5)
        >>> A3 = np.random.rand(3, 3)
        >>> batched_A, n1, n2 = pygm.utils.build_batch([A1, A2, A3], return_ori_dim=True)
        >>> batched_A.shape
        (3, 5, 5)
        >>> n1
        [4, 5, 3]
        >>> n2
        [4, 5, 3]

        # batched node features (feature dimension=10)
        >>> F1 = np.random.rand(4, 10)
        >>> F2 = np.random.rand(5, 10)
        >>> F3 = np.random.rand(3, 10)
        >>> batched_F = pygm.utils.build_batch([F1, F2, F3])
        >>> batched_F.shape
        (3, 5, 10)

    Example for Pytorch backend::

        >>> import torch
        >>> import pygmtools as pygm
        >>> pygm.BACKEND = 'pytorch'

        # batched adjacency matrices
        >>> A1 = torch.rand(4, 4)
        >>> A2 = torch.rand(5, 5)
        >>> A3 = torch.rand(3, 3)
        >>> batched_A, n1, n2 = pygm.utils.build_batch([A1, A2, A3], return_ori_dim=True)
        >>> batched_A.shape
        torch.Size([3, 5, 5])
        >>> n1
        tensor([4, 5, 3])
        >>> n2
        tensor([4, 5, 3])

        # batched node features (feature dimension=10)
        >>> F1 = torch.rand(4, 10)
        >>> F2 = torch.rand(5, 10)
        >>> F3 = torch.rand(3, 10)
        >>> batched_F = pygm.utils.build_batch([F1, F2, F3])
        >>> batched_F.shape
        torch.Size([3, 5, 10])

    """
    if backend is None:
        backend = pygmtools.BACKEND
    for item in input:
        _check_data_type(item, backend)
    args = (input, return_ori_dim)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        return mod.build_batch(*args)
    except ImportError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


def dense_to_sparse(dense_adj, backend=None):
    r"""
    Convert a dense connectivity/adjacency matrix to a sparse connectivity/adjacency matrix and an edge weight tensor.

    :param dense_adj: :math:`(b\times n\times n)` the dense adjacency matrix. This function also supports non-batched
                      input where the batch dimension ``b`` is ignored
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times ne\times 2)` sparse connectivity matrix, :math:`(b\times ne\times 1)` edge weight tensor,
             :math:`(b)` number of edges

    Example for numpy backend::

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

    Example for Pytorch backend::

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
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

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
        raise ValueError(f'the input argument s is expected to be 2-dimensional or 3-dimensional, got '
                         f's:{len(_get_shape(dense_adj))}!')

    args = (dense_adj,)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        result = mod.dense_to_sparse(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    if non_batched_input:
        return _squeeze(result[0], 0, backend), _squeeze(result[1], 0, backend)
    else:
        return result


def to_numpy(input, backend=None):
    r"""
    Convert a tensor to a numpy ndarray.
    This is the helper function to convert tensors across different backends via numpy.

    :param input: input tensor
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: numpy ndarray
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input,)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        return mod.to_numpy(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


def from_numpy(input, backend=None):
    r"""
    Convert a numpy ndarray to a tensor.
    This is the helper function to convert tensors across different backends via numpy.

    :param input: input ndarray
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: tensor for the backend
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input,)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        return mod.from_numpy(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


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
        return mod._aff_mat_from_node_edge_aff(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


def _check_data_type(input, backend=None):
    r"""
    Check whether the input data meets the backend. If not met, it will raise an ValueError

    :param input: input data (must be Tensor/ndarray)
    :return: None
    """
    if backend is None:
        backend = pygmtools.BACKEND
    args = (input, )
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        return mod._check_data_type(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


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
        return mod._check_shape(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

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
        return mod._get_shape(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


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
        return mod._squeeze(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )


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
        return mod._unsqueeze(*args)
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
