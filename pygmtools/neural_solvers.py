"""
**Neural network-based** graph matching solvers. It is recommended to integrate these networks as modules into your
existing deep learning pipeline (either supervised, unsupervised or reinforcement learning).
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
import numpy as np
from pygmtools.utils import NOT_IMPLEMENTED_MSG, from_numpy, \
    _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type
from pygmtools.classic_solvers import __check_gm_arguments


def pca_gm(feat1, feat2, A1, A2, n1=None, n2=None,
           in_channel=1024, hidden_channel=2048, out_channel=2048, num_layers=2, sk_max_iter=20, sk_tau=0.05,
           network=None, return_network=False, pretrain='voc',
           backend=None):
    r"""
    The **PCA-GM** (Permutation loss and Cross-graph Affinity Graph Matching) neural network model for processing two
    individual graphs (KB-QAP).
    The graph matching module is composed of several intra-graph embedding layers, a cross-graph embedding layer, and
    a Sinkhorn matching layer. Only the second last layer has a cross-graph update layer.

    See the following pipeline for an example, with application to visual graph matching (layers in the gray box
    + Affinity Metric + Sinkhorn are implemented by pygmtools):

    .. image:: ../../images/pca_gm.png

    See the following paper for more technical details:
    `"Wang et al. Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach. TPAMI 2020."
    <https://ieeexplore.ieee.org/abstract/document/9128045/>`_

    You may be also interested in the extended version IPCA-GM (see :func:`~pygmtools.neural_solvers.ipca_gm`).

    :param feat1: :math:`(b\times n_1 \times d)` input feature of graph1
    :param feat2: :math:`(b\times n_2 \times d)` input feature of graph2
    :param A1: :math:`(b\times n_1 \times n_1)` input adjacency matrix of graph1
    :param A2: :math:`(b\times n_2 \times n_2)` input adjacency matrix of graph2
    :param n1: :math:`(b)` number of nodes in graph1. Optional if all equal to :math:`n_1`
    :param n2: :math:`(b)` number of nodes in graph2. Optional if all equal to :math:`n_2`
    :param in_channel: (default: 1024) Channel size of the input layer. It must match the feature dimension :math:`(d)`
        of ``feat1, feat2``. Ignored if the network object is given (ignored if ``network!=None``)
    :param hidden_channel: (default: 2048) Channel size of hidden layers. Ignored if the network object is given
        (ignored if ``network!=None``)
    :param out_channel: (default: 2048) Channel size of the output layer. Ignored if the network object is given
        (ignored if ``network!=None``)
    :param num_layers: (default: 2) Number of graph embedding layers. Must be >=2. Ignored if the network object is
        given (ignored if ``network!=None``)
    :param sk_max_iter: (default: 20) Max number of iterations of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param sk_tau: (default: 0.05) The temperature parameter of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param network: (default: None) The network object. If None, a new network object will be created, and load the
        model weights specified in ``pretrain`` argument.
    :param return_network: (default: False) Return the network object (saving model construction time if calling the
        model multiple times).
    :param pretrain: (default: 'voc') If ``network==None``, the pretrained model weights to be loaded. Available
        pretrained weights: ``voc`` (on Pascal VOC Keypoint dataset), ``willow`` (on Willow Object Class dataset),
        ``voc-all`` (on Pascal VOC Keypoint dataset, without filtering), or ``False`` (no pretraining).
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if ``return_network==False``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix

        if ``return_network==True``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix,
        the network object

    .. note::
        You may need a proxy to load the pretrained weights if Google drive is not accessible in your contry/region.
        You may also download the pretrained models manually and put them at ``~/.cache/pygmtools`` (for Linux).

        `[google drive] <https://drive.google.com/drive/folders/1O7vkIW8QXBJsNsHUIRiSw91HJ_0FAzu_?usp=sharing>`_
        `[baidu drive] <https://pan.baidu.com/s/1MvzfM52NJeLWx2JXbbc6HA?pwd=x8bv>`_

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.set_backend('numpy')
            >>> np.random.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype='i4'), np.random.permutation(4)] = 1
            >>> A1 = 1. * (np.random.rand(batch_size, 4, 4) > 0.5)
            >>> for i in np.arange(4): # discard self-loop edges
            ...    for j in np.arange(batch_size):
            ...        A1[j][i][i] = 0
            >>> A2 = np.matmul(np.matmul(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> feat1 = np.random.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = np.matmul(X_gt.swapaxes(1, 2), feat1)
            >>> n1 = n2 = np.array([4] * batch_size)

            # Match by PCA-GM (load pretrained model)
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/pca_gm_voc_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may also load other pretrained weights
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/pca_gm_willow_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.pca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0


    .. dropdown:: PyTorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
            >>> _ = torch.manual_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = torch.zeros(batch_size, 4, 4)
            >>> X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
            >>> A1 = 1. * (torch.rand(batch_size, 4, 4) > 0.5)
            >>> torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> feat1 = torch.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = torch.tensor([4] * batch_size)

            # Match by PCA-GM (load pretrained model)
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/pca_gm_voc_pytorch.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            tensor(1.)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/pca_gm_willow_pytorch.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.pca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()

        
    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.set_backend('jittor')
            >>> _ = jt.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = jt.zeros((batch_size, 4, 4))
            >>> X_gt[:, jt.arange(0, 4, dtype=jt.int64), jt.randperm(4)] = 1
            >>> A1 = 1. * (jt.rand(batch_size, 4, 4) > 0.5)
            >>> for i in range(batch_size):
            >>>     for j in range(4):
            >>>         A1.data[i][j][j] = 0  # discard self-loop edges
            >>> A2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> feat1 = jt.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = jt.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = jt.Var([4] * batch_size)

            # Match by PCA-GM (load pretrained model)
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/pca_gm_voc_jittor.pt...

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/pca_gm_willow_jittor.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            jt.Var([1.], dtype=float32)

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.pca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = jt.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> optimizer.backward(loss)
            >>> optimizer.step()

            
    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.set_backend('paddle')
            >>> _ = paddle.seed(4)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = 1. * (paddle.rand((batch_size, 4, 4)) > 0.5)
            >>> paddle.diagonal(A1, axis1=1, axis2=2)[:] = 0 # discard self-loop edges
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> feat1 = paddle.rand((batch_size, 4, 1024)) - 0.5
            >>> feat2 = paddle.bmm(X_gt.transpose((0, 2, 1)), feat1)
            >>> n1 = n2 = paddle.to_tensor([4] * batch_size)

            # Match by PCA-GM (load pretrained model)
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/pca_gm_voc_paddle.pdparams...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [1.])

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/pca_gm_willow_paddle.pdparams...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.pca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=0.001)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.pca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()

    .. note::

        If you find this model useful in your research, please cite:

        ::

            @article{WangPAMI20,
              author = {Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
              title = {Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach},
              journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
              year = {2020}
            }
    """
    if not num_layers >= 2: raise ValueError(f'num_layers must be >=2, got {num_layers}!')

    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if feat1 is not None: # if feat1 is None, this function skips the forward pass and only returns a network object
        for var, name in ((feat1, 'feat1'), (feat2, 'feat2'), (A1, 'A1'), (A2, 'A2')):
            _check_data_type(var, name, backend)

        if all([_check_shape(_, 2, backend) for _ in (feat1, feat2, A1, A2)]):
            feat1, feat2, A1, A2 = [_unsqueeze(_, 0, backend) for _ in (feat1, feat2, A1, A2)]
            if isinstance(n1, (int, np.integer)): n1 = from_numpy(np.array([n1]), backend=backend)
            if isinstance(n2, (int, np.integer)): n2 = from_numpy(np.array([n2]), backend=backend)
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

    args = (feat1, feat2, A1, A2, n1, n2, in_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
           network, pretrain)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.pca_gm
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    if return_network:
        return match_mat, result[1]
    else:
        return match_mat


def ipca_gm(feat1, feat2, A1, A2, n1=None, n2=None,
            in_channel=1024, hidden_channel=2048, out_channel=2048, num_layers=2, cross_iter=3,
            sk_max_iter=20, sk_tau=0.05,
            network=None, return_network=False, pretrain='voc',
            backend=None):
    r"""
    The **IPCA-GM** (Iterative Permutation loss and Cross-graph Affinity Graph Matching) neural network model for
    processing two individual graphs (KB-QAP).
    The graph matching module is composed of several intra-graph embedding layers, a cross-graph embedding layer, and
    a Sinkhorn matching layer. The weight matrix of the cross-graph embedding layer is updated iteratively.
    Only the second last layer has a cross-graph update layer.
    IPCA-GM is the extended version of PCA-GM (see :func:`~pygmtools.neural_solvers.pca_gm`). The dfference is that
    the cross-graph weight in PCA-GM is computed in one shot, and in IPCA-GM it is updated iteratively.

    See the following pipeline for an example, with application to visual graph matching (layers in gray box are
    implemented by pygmtools):

    .. image:: ../../images/ipca_gm.png

    See the following paper for more technical details:
    `"Wang et al. Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach. TPAMI 2020."
    <https://ieeexplore.ieee.org/abstract/document/9128045/>`_

    :param feat1: :math:`(b\times n_1 \times d)` input feature of graph1
    :param feat2: :math:`(b\times n_2 \times d)` input feature of graph2
    :param A1: :math:`(b\times n_1 \times n_1)` input adjacency matrix of graph1
    :param A2: :math:`(b\times n_2 \times n_2)` input adjacency matrix of graph2
    :param n1: :math:`(b)` number of nodes in graph1. Optional if all equal to :math:`n_1`
    :param n2: :math:`(b)` number of nodes in graph2. Optional if all equal to :math:`n_2`
    :param in_channel: (default: 1024) Channel size of the input layer. It must match the feature dimension :math:`(d)`
        of ``feat1, feat2``. Ignored if the network object is given (ignored if ``network!=None``)
    :param hidden_channel: (default: 2048) Channel size of hidden layers. Ignored if the network object is given
        (ignored if ``network!=None``)
    :param out_channel: (default: 2048) Channel size of the output layer. Ignored if the network object is given
        (ignored if ``network!=None``)
    :param num_layers: (default: 2) Number of graph embedding layers. Must be >=2. Ignored if the network object is
        given (ignored if ``network!=None``)
    :param cross_iter: (default: 3) Number of iterations for the cross-graph embedding layer.
    :param sk_max_iter: (default: 20) Max number of iterations of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param sk_tau: (default: 0.05) The temperature parameter of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param network: (default: None) The network object. If None, a new network object will be created, and load the
        model weights specified in ``pretrain`` argument.
    :param return_network: (default: False) Return the network object (saving model construction time if calling the
        model multiple times).
    :param pretrain: (default: 'voc') If ``network==None``, the pretrained model weights to be loaded. Available
        pretrained weights: ``voc`` (on Pascal VOC Keypoint dataset), ``willow`` (on Willow Object Class dataset),
        or ``False`` (no pretraining).
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if ``return_network==False``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix

        if ``return_network==True``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix,
        the network object

    .. note::
        You may need a proxy to load the pretrained weights if Google drive is not accessible in your contry/region.
        You may also download the pretrained models manually and put them at ``~/.cache/pygmtools`` (for Linux).

        `[google drive] <https://drive.google.com/drive/folders/1O7vkIW8QXBJsNsHUIRiSw91HJ_0FAzu_?usp=sharing>`_
        `[baidu drive] <https://pan.baidu.com/s/1MvzfM52NJeLWx2JXbbc6HA?pwd=x8bv>`_

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.set_backend('numpy')
            >>> np.random.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype='i4'), np.random.permutation(4)] = 1
            >>> A1 = 1. * (np.random.rand(batch_size, 4, 4) > 0.5)
            >>> for i in np.arange(4): # discard self-loop edges
            ...    for j in np.arange(batch_size):
            ...        A1[j][i][i] = 0
            >>> A2 = np.matmul(np.matmul(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> feat1 = np.random.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = np.matmul(X_gt.swapaxes(1, 2), feat1)
            >>> n1 = n2 = np.array([4] * batch_size)

            # Match by IPCA-GM (load pretrained model)
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ipca_gm_voc_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may also load other pretrained weights
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/ipca_gm_willow_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ipca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, cross_iter=10, pretrain=False)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0


    .. dropdown:: PyTorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
            >>> _ = torch.manual_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = torch.zeros(batch_size, 4, 4)
            >>> X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
            >>> A1 = 1. * (torch.rand(batch_size, 4, 4) > 0.5)
            >>> torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> feat1 = torch.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = torch.tensor([4] * batch_size)

            # Match by IPCA-GM (load pretrained model)
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ipca_gm_voc_pytorch.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            tensor(1.)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/ipca_gm_willow_pytorch.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ipca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, cross_iter=10, pretrain=False)
            >>> optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()


    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.set_backend('jittor')
            >>> _ = jt.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = jt.zeros((batch_size, 4, 4))
            >>> X_gt[:, jt.arange(0, 4, dtype=jt.int64), jt.randperm(4)] = 1
            >>> A1 = 1. * (jt.rand(batch_size, 4, 4) > 0.5)
            >>> for i in range(batch_size):
            >>>     for j in range(4):
            >>>         A1.data[i][j][j] = 0  # discard self-loop edges
            >>> A2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> feat1 = jt.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = jt.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = jt.Var([4] * batch_size)

            # Match by IPCA-GM (load pretrained model)
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ipca_gm_voc_jitttor.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            jt.Var([1.], dtype=float32)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.ca/che/pygmtools/ipca_gm_willow_jittor.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ipca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, cross_iter=10, pretrain=False)
            >>> optimizer = jt.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> optimizer.backward(loss)
            >>> optimizer.step()
    
    
    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.set_backend('paddle')
            >>> _ = paddle.seed(5)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = 1. * (paddle.rand((batch_size, 4, 4)) > 0.5)
            >>> paddle.diagonal(A1, axis1=1, axis2=2)[:] = 0 # discard self-loop edges
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> feat1 = paddle.rand((batch_size, 4, 1024)) - 0.5
            >>> feat2 = paddle.bmm(X_gt.transpose((0, 2, 1)), feat1)
            >>> n1 = n2 = paddle.to_tensor([4] * batch_size)

            # Match by IPCA-GM (load pretrained model)
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ipca_gm_voc_paddle.pdparams...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [1.])

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/ipca_gm_willow_paddle.pdparams...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ipca_gm, in_channel=1024, hidden_channel=2048, out_channel=512, num_layers=3, cross_iter=10, pretrain=False)
            >>> optimizer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=0.001)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.ipca_gm(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()

    .. note::

        If you find this model useful in your research, please cite:

        ::

            @article{WangPAMI20,
              author = {Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
              title = {Combinatorial Learning of Robust Deep Graph Matching: an Embedding based Approach},
              journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
              year = {2020}
            }
    """
    if not num_layers >= 2: raise ValueError(f'num_layers must be >=2, got {num_layers}!')
    if not cross_iter >= 1: raise ValueError(f'cross_iter must be >=1, got {cross_iter}!')

    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if feat1 is not None:  # if feat1 is None, this function skips the forward pass and only returns a network object
        for var, name in ((feat1, 'feat1'), (feat2, 'feat2'), (A1, 'A1'), (A2, 'A2')):
            _check_data_type(var, name, backend)

        if all([_check_shape(_, 2, backend) for _ in (feat1, feat2, A1, A2)]):
            feat1, feat2, A1, A2 = [_unsqueeze(_, 0, backend) for _ in (feat1, feat2, A1, A2)]
            if isinstance(n1, (int, np.integer)): n1 = from_numpy(np.array([n1]), backend=backend)
            if isinstance(n2, (int, np.integer)): n2 = from_numpy(np.array([n2]), backend=backend)
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

    args = (feat1, feat2, A1, A2, n1, n2, in_channel, hidden_channel, out_channel, num_layers, cross_iter,
            sk_max_iter, sk_tau, network, pretrain)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.ipca_gm
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    if return_network:
        return match_mat, result[1]
    else:
        return match_mat


def cie(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1=None, n2=None,
        in_node_channel=1024, in_edge_channel=1, hidden_channel=2048, out_channel=2048, num_layers=2,
        sk_max_iter=20, sk_tau=0.05,
        network=None, return_network=False, pretrain='voc',
        backend=None):
    r"""
    The **CIE** (Channel Independent Embedding) graph matching neural network model for processing two individual graphs
    (KB-QAP).
    The graph matching module is composed of several intra-graph embedding layers, a cross-graph embedding layer, and
    a Sinkhorn matching layer. Only the second last layer has a cross-graph update layer. The graph embedding layers
    are based on channel-independent embedding, under the assumption that such a message passing scheme may offer higher
    model capacity especially with high-dimensional edge features.

    See the following pipeline for an example, with application to visual graph matching:

    .. image:: ../../images/cie_framework.png

    The graph embedding layer (CIE layer) involves both node embeding and edge embedding:

    .. image:: ../../images/cie_layer.png

    See the following paper for more technical details:
    `"Yu et al. Learning Deep Graph Matching with Channel-Independent Embedding and Hungarian Attention. ICLR 2020."
    <https://openreview.net/pdf?id=rJgBd2NYPH>`_

    :param feat_node1: :math:`(b\times n_1 \times d_n)` input node feature of graph1
    :param feat_node2: :math:`(b\times n_2 \times d_n)` input node feature of graph2
    :param A1: :math:`(b\times n_1 \times n_1)` input adjacency matrix of graph1
    :param A2: :math:`(b\times n_2 \times n_2)` input adjacency matrix of graph2
    :param feat_edge1: :math:`(b\times n_1 \times n_1 \times d_e)` input edge feature of graph1
    :param feat_edge2: :math:`(b\times n_2 \times n_2 \times d_e)` input edge feature of graph2
    :param n1: :math:`(b)` number of nodes in graph1. Optional if all equal to :math:`n_1`
    :param n2: :math:`(b)` number of nodes in graph2. Optional if all equal to :math:`n_2`
    :param in_node_channel: (default: 1024) Node channel size of the input layer. It must match the feature dimension
        :math:`(d_n)` of ``feat_node1, feat_node2``. Ignored if the network object is given (ignored if ``network!=None``)
    :param in_edge_channel: (default: 1) Edge channel size of the input layer. It must match the feature dimension
        :math:`(d_e)` of ``feat_edge1, feat_edge2``. Ignored if the network object is given (ignored if ``network!=None``)
    :param hidden_channel: (default: 2048) Channel size of hidden layers (node channel == edge channel).
        Ignored if the network object is given (ignored if ``network!=None``)
    :param out_channel: (default: 2048) Channel size of the output layer (node channel == edge channel).
        Ignored if the network object is given (ignored if ``network!=None``)
    :param num_layers: (default: 2) Number of graph embedding layers. Must be >=2. Ignored if the network object is
        given (ignored if ``network!=None``)
    :param sk_max_iter: (default: 20) Max number of iterations of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param sk_tau: (default: 0.05) The temperature parameter of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param network: (default: None) The network object. If None, a new network object will be created, and load the
        model weights specified in ``pretrain`` argument.
    :param return_network: (default: False) Return the network object (saving model construction time if calling the
        model multiple times).
    :param pretrain: (default: 'voc') If ``network==None``, the pretrained model weights to be loaded. Available
        pretrained weights: ``voc`` (on Pascal VOC Keypoint dataset), ``willow`` (on Willow Object Class dataset),
        or ``False`` (no pretraining).
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if ``return_network==False``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix

        if ``return_network==True``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix,
        the network object

    .. note::
        You may need a proxy to load the pretrained weights if Google drive is not accessible in your contry/region.
        You may also download the pretrained models manually and put them at ``~/.cache/pygmtools`` (for Linux).

        `[google drive] <https://drive.google.com/drive/folders/1O7vkIW8QXBJsNsHUIRiSw91HJ_0FAzu_?usp=sharing>`_
        `[baidu drive] <https://pan.baidu.com/s/1MvzfM52NJeLWx2JXbbc6HA?pwd=x8bv>`_

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.set_backend('numpy')
            >>> np.random.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype='i4'), np.random.permutation(4)] = 1
            >>> A1 = 1. * (np.random.rand(batch_size, 4, 4) > 0.5)
            >>> for i in np.arange(4): # discard self-loop edges
            ...    for j in np.arange(batch_size):
            ...        A1[j][i][i] = 0
            >>> e_feat1 = np.expand_dims(np.random.rand(batch_size, 4, 4) * A1,axis=-1) # shape: (10, 4, 4, 1)
            >>> A2 = np.matmul(np.matmul(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> e_feat2 = np.expand_dims(np.matmul(np.matmul(X_gt.swapaxes(1, 2),np.squeeze(e_feat1,axis=-1)), X_gt),axis=-1)
            >>> feat1 = np.random.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = np.matmul(X_gt.swapaxes(1, 2), feat1)
            >>> n1 = n2 = np.array([4] * batch_size)

            # Match by CIE (load pretrained model)
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/cie_voc_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may also load other pretrained weights
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/cie_willow_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.cie, in_node_channel=1024, in_edge_channel=1, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            # feat1/feat2/e_feat1/e_feat2 may be outputs by other neural networks
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0
            

    .. dropdown:: PyTorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
            >>> _ = torch.manual_seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = torch.zeros(batch_size, 4, 4)
            >>> X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
            >>> A1 = 1. * (torch.rand(batch_size, 4, 4) > 0.5)
            >>> torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
            >>> e_feat1 = (torch.rand(batch_size, 4, 4) * A1).unsqueeze(-1) # shape: (10, 4, 4, 1)
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> e_feat2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), e_feat1.squeeze(-1)), X_gt).unsqueeze(-1)
            >>> feat1 = torch.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = torch.tensor([4] * batch_size)

            # Match by CIE (load pretrained model)
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/cie_voc_pytorch.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            tensor(1.)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/cie_willow_pytorch.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.cie, in_node_channel=1024, in_edge_channel=1, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2/e_feat1/e_feat2 may be outputs by other neural networks
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.set_backend('jittor')
            >>> _ = jt.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = jt.zeros((batch_size, 4, 4))
            >>> X_gt[:, jt.arange(0, 4, dtype=jt.int64), jt.randperm(4)] = 1
            >>> A1 = 1. * (jt.rand(batch_size, 4, 4) > 0.5)
            >>> for i in range(batch_size):
            >>>     for j in range(4):
            >>>        A1.data[i][j][j] = 0  # discard self-loop edges
            >>> e_feat1 = (jt.rand(batch_size, 4, 4) * A1).unsqueeze(-1) # shape: (10, 4, 4, 1)
            >>> A2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> e_feat2 = jt.bmm(jt.bmm(X_gt.transpose(1, 2), e_feat1.squeeze(-1)), X_gt).unsqueeze(-1)
            >>> feat1 = jt.rand(batch_size, 4, 1024) - 0.5
            >>> feat2 = jt.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = jt.Var([4] * batch_size)

            # Match by CIE (load pretrained model)
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/cie_voc_jittor.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            jt.Var([1.], dtype=float32)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/cie_willow_jittor.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.cie, in_node_channel=1024, in_edge_channel=1, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = jt.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2/e_feat1/e_feat2 may be outputs by other neural networks
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> optimizer.backward(loss)
            >>> optimizer.step()

    
    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.set_backend('paddle')
            >>> _ = paddle.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = 1. * (paddle.rand((batch_size, 4, 4)) > 0.5)
            >>> paddle.diagonal(A1, axis1=1, axis2=2)[:] = 0 # discard self-loop edges
            >>> e_feat1 = (paddle.rand((batch_size, 4, 4)) * A1).unsqueeze(-1) # shape: (10, 4, 4, 1)
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), A1), X_gt)
            >>> e_feat2 = paddle.bmm(paddle.bmm(X_gt.transpose((0, 2, 1)), e_feat1.squeeze(-1)), X_gt).unsqueeze(-1)
            >>> feat1 = paddle.rand((batch_size, 4, 1024)) - 0.5
            >>> feat2 = paddle.bmm(X_gt.transpose((0, 2, 1)), feat1)
            >>> n1 = n2 = paddle.to_tensor([4] * batch_size)

            # Match by CIE (load pretrained model)
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/cie_voc_paddle.pdparams...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [1.])

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/cie_willow_paddle.pdparams...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.cie, in_node_channel=1024, in_edge_channel=1, hidden_channel=2048, out_channel=512, num_layers=3, pretrain=False)
            >>> optimizer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=0.001)
            # feat1/feat2/e_feat1/e_feat2 may be outputs by other neural networks
            >>> X = pygm.cie(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()
    
    .. note::

        If you find this model useful in your research, please cite:

        ::

            @inproceedings{YuICLR20,
              title={Learning deep graph matching with channel-independent embedding and Hungarian attention},
              author={Yu, Tianshu and Wang, Runzhong and Yan, Junchi and Li, Baoxin},
              booktitle={International Conference on Learning Representations},
              year={2020}
            }
    """
    if not num_layers >= 2: raise ValueError(f'num_layers must be >=2, got {num_layers}!')

    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if feat_node1 is not None:  # if feat_node1 is None, this function skips the forward pass and only returns a network object
        for var, name in ((feat_node1, 'feat_node1'), (feat_node2, 'feat_node2'), (A1, 'A1'), (A2, 'A2'),
                          (feat_edge1, 'feat_edge1'), (feat_edge2, 'feat_edge2')):
            _check_data_type(var, name, backend)

        if all([_check_shape(_, 2, backend) for _ in (feat_node1, feat_node2, A1, A2)]) \
                and all([_check_shape(_, 3, backend) for _ in (feat_edge1, feat_edge2)]):
            feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2 =\
                [_unsqueeze(_, 0, backend) for _ in (feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2)]
            if isinstance(n1, (int, np.integer)): n1 = from_numpy(np.array([n1]), backend=backend)
            if isinstance(n2, (int, np.integer)): n2 = from_numpy(np.array([n2]), backend=backend)
            non_batched_input = True
        elif all([_check_shape(_, 3, backend) for _ in (feat_node1, feat_node2, A1, A2)]) \
                and all([_check_shape(_, 4, backend) for _ in (feat_edge1, feat_edge2)]):
            non_batched_input = False
        else:
            raise ValueError(
                f'the dimensions of the input arguments are illegal. Got '
                f'feat_node1:{len(_get_shape(feat_node1, backend))}dims, feat_node2:{len(_get_shape(feat_node2, backend))}dims, '
                f'A1:{len(_get_shape(A1, backend))}dims, A2:{len(_get_shape(A2, backend))}dims, '
                f'feat_edge1:{len(_get_shape(feat_edge1, backend))}dims, feat_edge2:{len(_get_shape(feat_edge2, backend))}dims. '
                f'Read the doc for more details!')

        if not (_get_shape(feat_node1, backend)[0] == _get_shape(feat_node2, backend)[0] == _get_shape(A1, backend)[0] ==
                _get_shape(A2, backend)[0] == _get_shape(feat_edge1, backend)[0] == _get_shape(feat_edge2, backend)[0])\
                or not (_get_shape(feat_node1, backend)[1] == _get_shape(A1, backend)[1] == _get_shape(A1, backend)[2] ==
                        _get_shape(feat_edge1, backend)[1] == _get_shape(feat_edge1, backend)[2])\
                or not (_get_shape(feat_node2, backend)[1] == _get_shape(A2, backend)[1] == _get_shape(A2, backend)[2] ==
                        _get_shape(feat_edge2, backend)[1] == _get_shape(feat_edge2, backend)[2])\
                or not (_get_shape(feat_node1, backend)[2] == _get_shape(feat_node2, backend)[2])\
                or not (_get_shape(feat_edge1, backend)[3] == _get_shape(feat_edge2, backend)[3]):
            raise ValueError(
                f'the input dimensions do not match. Got feat_node1:{_get_shape(feat_node1, backend)}, '
                f'feat_node2:{_get_shape(feat_node2, backend)}, A1:{_get_shape(A1, backend)}, A2:{_get_shape(A2, backend)},'
                f'feat_edge1:{_get_shape(feat_edge1, backend)}, feat_edge2:{_get_shape(feat_edge2, backend)}!')
    if n1 is not None: _check_data_type(n1, 'n1', backend)
    if n2 is not None: _check_data_type(n2, 'n2', backend)

    args = (feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2,
            in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers,
            sk_max_iter, sk_tau, network, pretrain)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.cie
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    if return_network:
        return match_mat, result[1]
    else:
        return match_mat


def ngm(K, n1=None, n2=None, n1max=None, n2max=None, x0=None,
        gnn_channels=(16, 16, 16), sk_emb=1,
        sk_max_iter=20, sk_tau=0.05,
        network=None, return_network=False, pretrain='voc',
        backend=None):
    r"""
    The **NGM** (Neural Graph Matching) model for processing the affinity matrix (the most general form of Lawler's QAP).
    The math form of graph matching (Lawler's QAP) is equivalent to a vertex classification problem on the
    **association graph**, which is an equivalent formulation based on the affinity matrix :math:`\mathbf{K}`.
    The graph matching module is composed of several graph convolution layers, Sinkhorn embedding layers and finally
    a Sinkhorn layer to output a doubly-stochastic matrix.

    See the following pipeline for an example:

    .. image:: ../../images/ngm.png

    See the following paper for more technical details:
    `"Wang et al. Neural Graph Matching Network: Learning Lawler’s Quadratic Assignment Problem With Extension to
    Hypergraph and Multiple-Graph Matching. TPAMI 2022."
    <https://ieeexplore.ieee.org/abstract/document/9426408/>`_

    :param K: :math:`(b\times n_1n_2 \times n_1n_2)` the input affinity matrix, :math:`b`: batch size.
    :param n1: :math:`(b)` number of nodes in graph1 (optional if n1max is given, and all n1=n1max).
    :param n2: :math:`(b)` number of nodes in graph2 (optional if n2max is given, and all n2=n2max).
    :param n1max: :math:`(b)` max number of nodes in graph1 (optional if n1 is given, and n1max=max(n1)).
    :param n2max: :math:`(b)` max number of nodes in graph2 (optional if n2 is given, and n2max=max(n2)).
    :param x0: :math:`(b\times n_1 \times n_2)` an initial matching solution to warm-start the vertex embedding.
        If not given, the vertex embedding is initialized as a vector of all 1s.
    :param gnn_channels: (default: ``(16, 16, 16)``) A list/tuple of channel sizes of the GNN.
        Ignored if the network object is given (ignored if ``network!=None``)
    :param sk_emb: (default: 1) Number of Sinkhorn embedding channels. Sinkhorn embedding is designed to encode the
        matching constraints inside GNN layers. How it works: a Sinkhorn embedding channel accepts the vertex feature
        from the current layer and computes a doubly-stochastic matrix, which is then concatenated to the vertex feature.
        Ignored if the network object is given (ignored if ``network!=None``)
    :param sk_max_iter: (default: 20) Max number of iterations of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param sk_tau: (default: 0.05) The temperature parameter of Sinkhorn. See
        :func:`~pygmtools.classic_solvers.sinkhorn` for more details about this argument.
    :param network: (default: None) The network object. If None, a new network object will be created, and load the
        model weights specified in ``pretrain`` argument.
    :param return_network: (default: False) Return the network object (saving model construction time if calling the
        model multiple times).
    :param pretrain: (default: 'voc') If ``network==None``, the pretrained model weights to be loaded. Available
        pretrained weights: ``voc`` (on Pascal VOC Keypoint dataset), ``willow`` (on Willow Object Class dataset),
        or ``False`` (no pretraining).
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if ``return_network==False``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix

        if ``return_network==True``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix,
        the network object

    .. note::
        You may need a proxy to load the pretrained weights if Google drive is not accessible in your contry/region.
        You may also download the pretrained models manually and put them at ``~/.cache/pygmtools`` (for Linux).

        `[google drive] <https://drive.google.com/drive/folders/1O7vkIW8QXBJsNsHUIRiSw91HJ_0FAzu_?usp=sharing>`_
        `[baidu drive] <https://pan.baidu.com/s/1MvzfM52NJeLWx2JXbbc6HA?pwd=x8bv>`_

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.set_backend('numpy')
            >>> np.random.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = np.zeros((batch_size, 4, 4))
            >>> X_gt[:, np.arange(0, 4, dtype='i4'), np.random.permutation(4)] = 1
            >>> A1 = np.random.rand(batch_size, 4, 4)
            >>> A2 = np.matmul(np.matmul(X_gt.swapaxes(1, 2), A1), X_gt)
            >>> n1 = n2 = np.array([4] * batch_size)

            # Build affinity matrix
            >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
            >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
            >>> import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by NGM
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ngm_voc_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ngm(K, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may also load other pretrained weights
            >>> X, net = pygm.ngm(feat1, feat2, A1, A2, e_feat1, e_feat2, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/ngm_willow_numpy.npy...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ngm, gnn_channels=(32, 64, 128, 64, 32), sk_emb=8, pretrain=False)
            # K may be outputs by other neural networks (constructed K from node/edge features by pygm.utils.build_aff_mat)
            >>> X = pygm.ngm(K, n1, n2, network=net)
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            1.0


    .. dropdown:: PyTorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
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

            # Solve by NGM
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ngm_voc_pytorch.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            tensor(1.)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ngm(K, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/ngm_willow_pytorch.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ngm, gnn_channels=(32, 64, 128, 64, 32), sk_emb=8, pretrain=False)
            >>> optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # K may be outputs by other neural networks (constructed K from node/edge features by pygm.utils.build_aff_mat)
            >>> X = pygm.ngm(K, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.set_backend('jittor')
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
            import functools
            >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
            >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

            # Solve by NGM
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True)
            # Downloading to ~/.cache/pygmtools/ngm_voc_jittor.pt...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            jt.Var([1.], dtype=float32)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ngm(K, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True, pretrain='willow')
            # Downloading to ~/.cache/pygmtools/ngm_willow_jittor.pt...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ngm, gnn_channels=(32, 64, 128, 64, 32), sk_emb=8, pretrain=False)
            >>> optimizer = jt.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # K may be outputs by other neural networks (constructed K from node/edge features by pygm.utils.build_aff_mat)
            >>> X = pygm.ngm(K, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> optimizer.backward(loss)
            >>> optimizer.step()        

    
    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.set_backend('paddle')
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

            # Solve by NGM
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/ngm_voc_paddle.pdparams...
            >>> (pygm.hungarian(X) * X_gt).sum() / X_gt.sum() # accuracy
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                   [1.])

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.ngm(K, n1, n2, network=net)

            # You may also load other pretrained weights
            >>> X, net = pygm.ngm(K, n1, n2, return_network=True, pretrain='willow')
            Downloading to ~/.cache/pygmtools/ngm_willow_paddle.pdparams...

            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.ngm, gnn_channels=(32, 64, 128, 64, 32), sk_emb=8, pretrain=False)
            >>> optimizer = paddle.optimizer.SGD(parameters=net.parameters(), learning_rate=0.001)
            # K may be outputs by other neural networks (constructed K from node/edge features by pygm.utils.build_aff_mat)
            >>> X = pygm.ngm(K, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()
    
    .. note::

        If you find this model useful in your research, please cite:

        ::

            @ARTICLE{WangPAMI22,
              author={Wang, Runzhong and Yan, Junchi and Yang, Xiaokang},
              journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
              title={Neural Graph Matching Network: Learning Lawler’s Quadratic Assignment Problem With Extension to Hypergraph and Multiple-Graph Matching},
              year={2022},
              volume={44},
              number={9},
              pages={5261-5279},
              doi={10.1109/TPAMI.2021.3078053}
            }
    """
    if not len(gnn_channels) >= 1: raise ValueError(f'gnn_channels should not be empty!')
    if not sk_emb >= 0: raise ValueError(f'sk_emb must be >=0. Got sk_emb={sk_emb}!')

    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if K is not None: # if K is None, this function skips the forward pass and only returns a network object
        _check_data_type(K, 'K', backend)
        if _check_shape(K, 2, backend):
            K = _unsqueeze(K, 0, backend)
            non_batched_input = True
            if isinstance(n1, (int, np.integer)) and n1max is None:
                n1max = n1
                n1 = None
            if isinstance(n2, (int, np.integer)) and n2max is None:
                n2max = n2
                n2 = None
        elif _check_shape(K, 3, backend):
            non_batched_input = False
        else:
            raise ValueError(f'the input argument K is expected to be 2-dimensional or 3-dimensional, got '
                             f'K:{len(_get_shape(K, backend))}dims!')
        __check_gm_arguments(n1, n2, n1max, n2max)

    args = (K, n1, n2, n1max, n2max, x0, gnn_channels, sk_emb, sk_max_iter, sk_tau, network, return_network, pretrain)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.ngm
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    result = fn(*args)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    if return_network:
        return match_mat, result[1]
    else:
        return match_mat


def genn_astar(feat1, feat2, A1, A2, n1=None, n2=None, channel=None, filters_1=64, filters_2=32, filters_3=16,
           tensor_neurons=16, beam_width=0, trust_fact=1, no_pred_size=0,
           network=None, return_network=False, pretrain='AIDS700nef', backend=None):
    r"""
    The **GENN-A\*** (Graph Edit Neural Network A\*) solver for graph matching (and graph edit distance)
    based on the fusion of traditional A\* and Neural Network.
    This algorithm replaces the heuristic prediction module in the traditional A\* algorithm with **GENN** (Graph Edit
    Neural Network) model, greatly improving the efficiency of A\* algorithm with negligible loss in accuracy.
    During the search process, the algorithm chooses the next state with the lowest edit cost, which is predicted by
    GENN.
    
    See the following picture to better understand the workflow of the algorithm:

    .. image:: ../../images/astar.png

    See the following paper for more technical details:
    `"Combinatorial Learning of Graph Edit Distance via Dynamic Embedding"
    <https://ieeexplore.ieee.org/document/9578389/>`_

 
    :param feat1: :math:`(b\times n_1 \times d)` input feature of graph1
    :param feat2: :math:`(b\times n_2 \times d)` input feature of graph2
    :param A1: :math:`(b\times n_1 \times n_1)` input adjacency matrix of graph1
    :param A2: :math:`(b\times n_2 \times n_2)` input adjacency matrix of graph2
    :param n1: :math:`(b)` number of nodes in graph1. Optional if all equal to :math:`n_1`
    :param n2: :math:`(b)` number of nodes in graph2. Optional if all equal to :math:`n_2`
    :param channel: (default: None)  Channel size of the input layer. If given, it must match the feature dimension (d) of feat1, feat2. 
        If not given, it will be defined by the feature dimension (d) of feat1, feat2. 
        Ignored if the network object isgiven (ignored if network!=None)
    :param filters_1: (default: 64) Filters (neurons) in 1st convolution.
    :param filters_2: (default: 32) Filters (neurons) in 2nd convolution.
    :param filters_3: (default: 16) Filters (neurons) in 2nd convolution.
    :param tensor_neurons: (default: 16) Neurons in tensor network layer.
    :param beam_width: (default: 0) Size of beam-search witdh (0 = no beam).
    :param trust_fact: (default: 1) The trust factor on GNN prediction (0 = no GNN).
    :param no_pred_size: (default: 0) If the smaller graph has no more than ``no_pred_size`` nodes, stop using heuristics.
    :param network: (default: None) The network object. If None, a new network object will be created, and load the
        model weights specified in ``pretrain`` argument.
    :param return_network: (default: False) Return the network object (saving model construction time if calling the
        model multiple times).
    :param pretrain: (default: 'AIDS700nef') If ``network==None``, the pretrained model weights to be loaded. Available
        pretrained weights: ``AIDS700nef`` (channel=36), ``LINUX`` (channel=8),
        or ``False`` (no pretraining).
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: if ``return_network==False``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix

        if ``return_network==True``, :math:`(b\times n_1 \times n_2)` the doubly-stochastic matching matrix,
        the network object

    .. note::
        Graph matching problem (Lawler's QAP) and graph edit distance problem are two sides of the same coin. If you
        want to transform your graph edit distance problem into Lawler's QAP, first compute the edit cost of each
        node pairs and edge pairs, then place them to the right places to get a **cost** matrix. Finally, build
        the affinity matrix by :math:`-1\times` cost matrix.

        You can get your customized cost/affinity matrix by :func:`~pygmtools.utils.build_aff_mat`.

    .. note::
        You may need a proxy to load the pretrained weights if Google drive is not accessible in your contry/region.
        You may also download the pretrained models manually and put them at ``~/.cache/pygmtools`` (for Linux).

        `[google drive] <https://drive.google.com/drive/folders/1mUpwHeW1RbMHaNxX_PZvD5HrWvyCQG8y>`_

    .. note::
        This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.

    .. dropdown:: PyTorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
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

            # Match by GENN-A* (load pretrained model)
            >>> X, net = pygm.genn_astar(feat1, feat2, A1, A2, n1, n2, return_network=True)
            Downloading to ~/.cache/pygmtools/best_genn_AIDS700nef_gcn_astar.pt...
            >>> (X * X_gt).sum() / X_gt.sum()# accuracy
            tensor(1.)

            # Pass the net object to avoid rebuilding the model agian
            >>> X = pygm.genn_astar(feat1, feat2, A1, A2, n1, n2, network=net)
            
            # This function also supports non-batched input, by ignoring all batch dimensions in the input tensors.
            >>> part_f1 = feat1[0]
            >>> part_f2 = feat2[0]
            >>> part_A1 = A1[0]
            >>> part_A2 = A2[0]
            >>> part_X_gt = X_gt[0]
            >>> part_X = pygm.genn_astar(part_f1, part_f2, part_A1, part_A2, return_network=False)
            
            >>> part_X.shape
            torch.Size([4, 4])
            
            >>> (part_X * part_X_gt).sum() / part_X_gt.sum()# accuracy
            tensor(1.)
            
            # You may also load other pretrained weights
            # However, it should be noted that each pretrained set supports different node feature dimensions
            # AIDS700nef(Default): channel = 36
            # LINUX: channel = 8
            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> nodes_num = 4
            >>> channel = 8

            >>> X_gt = torch.zeros(batch_size, nodes_num, nodes_num)
            >>> X_gt[:, torch.arange(0, nodes_num, dtype=torch.int64), torch.randperm(nodes_num)] = 1
            >>> A1 = 1. * (torch.rand(batch_size, nodes_num, nodes_num) > 0.5)
            >>> torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
            >>> A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> feat1 = torch.rand(batch_size, nodes_num, channel) - 0.5
            >>> feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
            >>> n1 = n2 = torch.tensor([nodes_num] * batch_size)
            
            >>> X, net = pygm.genn_astar(feat1, feat2, A1, A2, n1, n2, pretrain='LINUX', return_network=True)
            Downloading to ~/.cache/pygmtools/best_genn_LINUX_gcn_astar.pt...

            >>> (X * X_gt).sum() / X_gt.sum()# accuracy
            tensor(1.)
            
            # When the input node feature dimension is different from the one supported by pre training, 
            # you can still use the solver, but the solver will provide a warning
            >>> X, net = pygm.genn_astar(feat1, feat2, A1, A2, n1, n2, return_network=True, pretrain='AIDS700nef')
            Warning: Skip key(s) in state_dict: "convolution_1.weight". 
            Warning: Pretrain AIDS700nef does not support the parameters you entered, 
                     Supported parameters: ( channel:36, filters:(64,32,16) ), 
                     Input parameters: ( channel:8, filters:(64,32,16) )
            
            # You may configure your own model and integrate the model into a deep learning pipeline. For example:
            >>> net = pygm.utils.get_network(pygm.genn_astar, channel = 1000, filters_1 = 1024, filters_2 = 256, filters_3 = 128, pretrain=False)
            >>> optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            # feat1/feat2 may be outputs by other neural networks
            >>> X = pygm.genn_astar(feat1, feat2, A1, A2, n1, n2, network=net)
            >>> loss = pygm.utils.permutation_loss(X, X_gt)
            >>> loss.backward()
            >>> optimizer.step()

    .. note::

        If you find this model useful in your research, please cite:

        ::

            @inproceedings{WangCVPR21,
              author={Runzhong Wang, Tianqi Zhang, Tianshu Yu, Junchi Yan, Xiaokang Yang},
              booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
              title={Combinatorial Learning of Graph Edit Distance via Dynamic Embedding},
              year={2021},
            }
    """

    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if feat1 is not None: # if feat1 is None, this function skips the forward pass and only returns a network object
        for var, name in ((feat1, 'feat1'), (feat2, 'feat2'), (A1, 'A1'), (A2, 'A2')):
            _check_data_type(var, name, backend)

        if all([_check_shape(_, 2, backend) for _ in (feat1, feat2, A1, A2)]):
            feat1, feat2, A1, A2 = [_unsqueeze(_, 0, backend) for _ in (feat1, feat2, A1, A2)]
            if isinstance(n1, (int, np.integer)): n1 = from_numpy(np.array([n1]), backend=backend)
            if isinstance(n2, (int, np.integer)): n2 = from_numpy(np.array([n2]), backend=backend)
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

    args = (feat1, feat2, A1, A2, n1, n2, channel, filters_1, filters_2, filters_3, 
            tensor_neurons, beam_width, trust_fact, no_pred_size, network, pretrain)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.genn_astar
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    if return_network:
        return match_mat, result[1]
    else:
        return match_mat
    