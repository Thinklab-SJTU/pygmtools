r"""
Classic (learning-free) **linear assignment problem** solvers. These linear assignment solvers are recommended to solve
matching problems with only nodes (i.e. linear matching problems), or large-scale graph matching problems where the cost
of QAP formulation is too high.

The linear assignment problem only considers nodes, and is also known as bipartite graph matching and linear matching:

.. math::

    &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S})\\
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
import numpy as np
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, from_numpy, \
    _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type


def sinkhorn(s, n1=None, n2=None, unmatch1=None, unmatch2=None,
             dummy_row: bool = False, max_iter: int = 10, tau: float = 1., batched_operation: bool = False,
             backend=None):
    r"""
    Sinkhorn algorithm turns the input matrix into a doubly-stochastic matrix.

    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:

    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)

    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:

    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_1} \mathbf{1}_{n_1}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)

    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size. Non-batched input is also
              supported if ``s`` is of size :math:`(n_1 \times n_2)`
    :param n1: (optional) :math:`(b)` number of objects in dim1
    :param n2: (optional) :math:`(b)` number of objects in dim2
    :param unmatch1: (optional, new in ``0.3.0``) :math:`(b\times n_1)` the scores indicating the objects in dim1 is unmatched
    :param unmatch2: (optional, new in ``0.3.0``) :math:`(b\times n_2)` the scores indicating the objects in dim2 is unmatched
    :param dummy_row: (default: False) whether to add dummy rows (rows whose elements are all 0) to pad the matrix
                      to square matrix.
    :param max_iter: (default: 10) maximum iterations
    :param tau: (default: 1) the hyper parameter :math:`\tau` controlling the temperature
    :param batched_operation: (default: False) apply batched_operation for better efficiency (but may cause issues
     for back-propagation)
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1 \times n_2)` the computed doubly-stochastic matrix

    You need not dive too deep into the math details if you are simply using Sinkhorn. However, you should
    be aware of one important hyper parameter. ``tau`` controls the distance between the predicted doubly-
    stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
    :func:`~pygmtools.linear_solvers.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
    Hungarian, at the cost of slower convergence speed and reduced numerical stability.

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded and all elements in ``n1`` are equal, all in ``n2`` are equal.

    .. note::
        The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
        matched have different number of nodes, it is a common practice to add dummy rows to construct a square
        matrix. After the row and column normalizations, the padded rows are discarded.

    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient. It is assumed that ``row number <= column number``. If not, the input matrix will be
        transposed.

    .. warning::
        This function can work with or without the maximal inlier matching:

        * **With maximal inlier matching** (the default mode). If ``unmatch1=None`` and ``unmatch2=None``,
          the solver aims to match as many nodes as possible. The corresponding linear assignment problem is

          .. math::

              &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S})\\
              s.t. \quad &\mathbf{X} \in [0, 1]^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}

          where the constraint :math:`\mathbf{X}\mathbf{1} = \mathbf{1}` urges the solver to match as many inlier
          nodes as possible. :math:`\mathbf{X}` is relaxed to continuous value in Sinkhorn.
        * **Without maximal inlier matching** (new in ``0.3.0``). If ``unmatch1`` and ``unmatch2`` are not ``None``,
          the solver is allowed to match nodes to void nodes, and the corresponding scores for matching to void nodes
          are specified by ``unmatch1`` and ``unmatch2``. The following (modified) linear assignment problem is
          considered:

          .. math::
              &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S}^\prime)\\
              s.t. \quad &\mathbf{X} \in [0, 1]^{n_1+1\times n_2+1}, \ \mathbf{X}_{[0:n_1, :]}\mathbf{1} = \mathbf{1}, \ \mathbf{X}_{[:, 0:n_2]}^\top\mathbf{1} \leq \mathbf{1}

          where the last column and last row of :math:`\mathbf{S}^\prime` are ``unmatch1`` and ``unmatch2``,
          respectively.

          For example, if you want to solve the following problem (note that both consrtraints are :math:`\leq`)

          .. math::

              &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S})\\
              s.t. \quad &\mathbf{X} \in [0, 1]^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} \leq \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}

          you can simply set ``unmatch1`` and ``unmatch2`` as zero vectors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.set_backend('numpy')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = np.random.rand(5, 5)
            >>> s_2d
            array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ],
                   [0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152],
                   [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                   [0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215],
                   [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]])
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            array([[0.18880224, 0.24990915, 0.19202217, 0.16034278, 0.20892366],
                   [0.18945066, 0.17240445, 0.23345011, 0.22194762, 0.18274716],
                   [0.23713583, 0.204348  , 0.18271243, 0.23114583, 0.1446579 ],
                   [0.11731039, 0.1229692 , 0.23823909, 0.19961588, 0.32186549],
                   [0.26730088, 0.2503692 , 0.15357619, 0.18694789, 0.1418058 ]])

            # 3-dimensional (batched) input
            >>> s_3d = np.random.rand(3, 5, 5)
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', x.sum(2))
            row_sum: [[1.         1.         1.         1.         1.        ]
             [0.99999998 1.00000002 0.99999999 1.00000003 0.99999999]
             [1.         1.         1.         1.         1.        ]]
            >>> print('col_sum:', x.sum(1))
            col_sum: [[1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1.]]

             # If the 3-d tensor are with different number of nodes
            >>> n1 = np.array([3, 4, 5])
            >>> n2 = np.array([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            array([[0.36665934, 0.21498158, 0.41835906, 0.        , 0.        ],
                   [0.27603621, 0.44270207, 0.28126175, 0.        , 0.        ],
                   [0.35730445, 0.34231636, 0.3003792 , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.        ]])
            >>> x[1] # non-zero size: 4x4
            array([[0.28847831, 0.20583051, 0.34242091, 0.16327021, 0.        ],
                   [0.22656752, 0.30153021, 0.19407969, 0.27782262, 0.        ],
                   [0.25346378, 0.19649853, 0.32565049, 0.22438715, 0.        ],
                   [0.23149039, 0.29614075, 0.13784891, 0.33452002, 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.        ]])
            >>> x[2] # non-zero size: 5x5
            array([[0.20147352, 0.19541986, 0.24942798, 0.17346397, 0.18021467],
                   [0.21050732, 0.17620948, 0.18645469, 0.20384684, 0.22298167],
                   [0.18319623, 0.18024007, 0.17619871, 0.1664133 , 0.29395169],
                   [0.20754376, 0.2236443 , 0.19658101, 0.20570847, 0.16652246],
                   [0.19727917, 0.22448629, 0.19133762, 0.25056742, 0.13632951]])

            # non-squared input
            >>> s_non_square = np.random.rand(4, 5)
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: [1. 1. 1. 1.] col_sum: [0.78239609 0.80485526 0.80165627 0.80004254 0.81104984]

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = np.random.randn(5, 5)
            >>> s_2d
            array([[ 0.01050002,  1.78587049,  0.12691209,  0.40198936,  1.8831507 ],
                   [-1.34775906, -1.270485  ,  0.96939671, -1.17312341,  1.94362119],
                   [-0.41361898, -0.74745481,  1.92294203,  1.48051479,  1.86755896],
                   [ 0.90604466, -0.86122569,  1.91006495, -0.26800337,  0.8024564 ],
                   [ 0.94725197, -0.15501009,  0.61407937,  0.92220667,  0.37642553]])
            >>> unmatch1 = np.random.randn(5)
            >>> unmatch1
            array([-1.09940079,  0.29823817,  1.3263859 , -0.69456786, -0.14963454])
            >>> unmatch2 = np.random.randn(5)
            >>> unmatch2
            array([-0.43515355,  1.84926373,  0.67229476,  0.40746184, -0.76991607])
            >>> x = pygm.sinkhorn(s_2d, unmatch1=unmatch1, unmatch2=unmatch2, max_iter=40)
            >>> x
            array([[0.12434101, 0.23913991, 0.05663597, 0.13943479, 0.31811425],
                   [0.03084473, 0.01085787, 0.12689067, 0.02784578, 0.3260589 ],
                   [0.03192548, 0.00745004, 0.13391025, 0.16087345, 0.12289304],
                   [0.29820536, 0.01659601, 0.32997174, 0.06988242, 0.10573396],
                   [0.29787774, 0.0322356 , 0.08654936, 0.22023996, 0.06619393]])
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: [0.87766593 0.52249794 0.45705226 0.82038949 0.70309659] col_sum: [0.78319431 0.30627943 0.733958   0.61827641 0.93899407]

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = torch.from_numpy(np.random.rand(5, 5))
            >>> s_2d
            tensor([[0.5488, 0.7152, 0.6028, 0.5449, 0.4237],
                    [0.6459, 0.4376, 0.8918, 0.9637, 0.3834],
                    [0.7917, 0.5289, 0.5680, 0.9256, 0.0710],
                    [0.0871, 0.0202, 0.8326, 0.7782, 0.8700],
                    [0.9786, 0.7992, 0.4615, 0.7805, 0.1183]], dtype=torch.float64)
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            tensor([[0.1888, 0.2499, 0.1920, 0.1603, 0.2089],
                    [0.1895, 0.1724, 0.2335, 0.2219, 0.1827],
                    [0.2371, 0.2043, 0.1827, 0.2311, 0.1447],
                    [0.1173, 0.1230, 0.2382, 0.1996, 0.3219],
                    [0.2673, 0.2504, 0.1536, 0.1869, 0.1418]], dtype=torch.float64)
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float64)
            col_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float64)

            # 3-dimensional (batched) input
            >>> s_3d = torch.from_numpy(np.random.rand(3, 5, 5))
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', x.sum(2))
            row_sum: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]], dtype=torch.float64)
            >>> print('col_sum:', x.sum(1))
            col_sum: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]], dtype=torch.float64)

            # If the 3-d tensor are with different number of nodes
            >>> n1 = torch.tensor([3, 4, 5])
            >>> n2 = torch.tensor([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            tensor([[0.3667, 0.2150, 0.4184, 0.0000, 0.0000],
                    [0.2760, 0.4427, 0.2813, 0.0000, 0.0000],
                    [0.3573, 0.3423, 0.3004, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=torch.float64)
            >>> x[1] # non-zero size: 4x4
            tensor([[0.2885, 0.2058, 0.3424, 0.1633, 0.0000],
                    [0.2266, 0.3015, 0.1941, 0.2778, 0.0000],
                    [0.2535, 0.1965, 0.3257, 0.2244, 0.0000],
                    [0.2315, 0.2961, 0.1378, 0.3345, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=torch.float64)
            >>> x[2] # non-zero size: 5x5
            tensor([[0.2015, 0.1954, 0.2494, 0.1735, 0.1802],
                    [0.2105, 0.1762, 0.1865, 0.2038, 0.2230],
                    [0.1832, 0.1802, 0.1762, 0.1664, 0.2940],
                    [0.2075, 0.2236, 0.1966, 0.2057, 0.1665],
                    [0.1973, 0.2245, 0.1913, 0.2506, 0.1363]], dtype=torch.float64)

            # non-squared input
            >>> s_non_square = torch.from_numpy(np.random.rand(4, 5))
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float64) col_sum: tensor([0.7824, 0.8049, 0.8017, 0.8000, 0.8110], dtype=torch.float64)

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = torch.from_numpy(np.random.randn(5, 5))
            >>> s_2d
            tensor([[ 0.0105,  1.7859,  0.1269,  0.4020,  1.8832],
                    [-1.3478, -1.2705,  0.9694, -1.1731,  1.9436],
                    [-0.4136, -0.7475,  1.9229,  1.4805,  1.8676],
                    [ 0.9060, -0.8612,  1.9101, -0.2680,  0.8025],
                    [ 0.9473, -0.1550,  0.6141,  0.9222,  0.3764]], dtype=torch.float64)
            >>> unmatch1 = torch.from_numpy(np.random.randn(5))
            >>> unmatch1
            tensor([-1.0994,  0.2982,  1.3264, -0.6946, -0.1496], dtype=torch.float64)
            >>> unmatch2 = torch.from_numpy(np.random.randn(5))
            >>> unmatch2
            tensor([-0.4352,  1.8493,  0.6723,  0.4075, -0.7699], dtype=torch.float64)
            >>> x = pygm.sinkhorn(s_2d, unmatch1=unmatch1, unmatch2=unmatch2, max_iter=40)
            >>> x
            tensor([[0.1243, 0.2391, 0.0566, 0.1394, 0.3181],
                    [0.0308, 0.0109, 0.1269, 0.0278, 0.3261],
                    [0.0319, 0.0075, 0.1339, 0.1609, 0.1229],
                    [0.2982, 0.0166, 0.3300, 0.0699, 0.1057],
                    [0.2979, 0.0322, 0.0865, 0.2202, 0.0662]], dtype=torch.float64)
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: tensor([0.8777, 0.5225, 0.4571, 0.8204, 0.7031], dtype=torch.float64) col_sum: tensor([0.7832, 0.3063, 0.7340, 0.6183, 0.9390], dtype=torch.float64)

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.set_backend('paddle')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = paddle.to_tensor(np.random.rand(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.54881350, 0.71518937, 0.60276338, 0.54488318, 0.42365480],
                    [0.64589411, 0.43758721, 0.89177300, 0.96366276, 0.38344152],
                    [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                    [0.08712930, 0.02021840, 0.83261985, 0.77815675, 0.87001215],
                    [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]])
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.18880224, 0.24990915, 0.19202217, 0.16034278, 0.20892366],
                    [0.18945066, 0.17240445, 0.23345011, 0.22194762, 0.18274716],
                    [0.23713583, 0.20434800, 0.18271243, 0.23114583, 0.14465790],
                    [0.11731039, 0.12296920, 0.23823909, 0.19961588, 0.32186549],
                    [0.26730088, 0.25036920, 0.15357619, 0.18694789, 0.14180580]])
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                            [1.00000000, 1.00000001, 0.99999998, 1.00000005, 0.99999997])
            col_sum: Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                            [1.00000000, 1.00000000, 1.00000000, 1.        , 1.00000000])

            # 3-dimensional (batched) input
            >>> s_3d = paddle.to_tensor(np.random.rand(3, 5, 5))
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', x.sum(2))
            row_sum: Tensor(shape=[3, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                            [[1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000],
                             [0.99999998, 1.00000002, 0.99999999, 1.00000003, 0.99999999],
                             [1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000]])
            >>> print('col_sum:', x.sum(1))
            col_sum: Tensor(shape=[3, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                            [[1.00000000, 1.        , 1.        , 1.00000000, 1.00000000],
                             [1.        , 1.        , 1.        , 1.        , 1.        ],
                             [1.        , 1.        , 1.        , 1.        , 1.00000000]])

            # If the 3-d tensor are with different number of nodes
            >>> n1 = paddle.to_tensor([3, 4, 5])
            >>> n2 = paddle.to_tensor([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.36665934, 0.21498158, 0.41835906, 0.00000000, 0.00000000],
                    [0.27603621, 0.44270207, 0.28126175, 0.00000000, 0.00000000],
                    [0.35730445, 0.34231636, 0.30037920, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]])
            >>> x[1] # non-zero size: 4x4
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.28847831, 0.20583051, 0.34242091, 0.16327021, 0.00000000],
                    [0.22656752, 0.30153021, 0.19407969, 0.27782262, 0.00000000],
                    [0.25346378, 0.19649853, 0.32565049, 0.22438715, 0.00000000],
                    [0.23149039, 0.29614075, 0.13784891, 0.33452002, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]])
            >>> x[2] # non-zero size: 5x5
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.20147352, 0.19541986, 0.24942798, 0.17346397, 0.18021467],
                    [0.21050732, 0.17620948, 0.18645469, 0.20384684, 0.22298167],
                    [0.18319623, 0.18024007, 0.17619871, 0.16641330, 0.29395169],
                    [0.20754376, 0.22364430, 0.19658101, 0.20570847, 0.16652246],
                    [0.19727917, 0.22448629, 0.19133762, 0.25056742, 0.13632951]])

            # non-squared input
            >>> s_non_square = paddle.to_tensor(np.random.rand(4, 5))
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: Tensor(shape=[4], dtype=float64, place=Place(cpu), stop_gradient=True,
                            [1.00000000, 1.00000000, 1.00000000, 1.00000000])
            col_sum: Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                            [0.78239609, 0.80485526, 0.80165627, 0.80004254, 0.81104984])

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = paddle.to_tensor(np.random.randn(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[ 0.01050002,  1.78587049,  0.12691209,  0.40198936,  1.88315070],
                    [-1.34775906, -1.27048500,  0.96939671, -1.17312341,  1.94362119],
                    [-0.41361898, -0.74745481,  1.92294203,  1.48051479,  1.86755896],
                    [ 0.90604466, -0.86122569,  1.91006495, -0.26800337,  0.80245640],
                    [ 0.94725197, -0.15501009,  0.61407937,  0.92220667,  0.37642553]])
            >>> unmatch1 = paddle.to_tensor(np.random.randn(5))
            >>> unmatch1
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [-1.09940079,  0.29823817,  1.32638590, -0.69456786, -0.14963454])
            >>> unmatch2 = paddle.to_tensor(np.random.randn(5))
            >>> unmatch2
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [-0.43515355,  1.84926373,  0.67229476,  0.40746184, -0.76991607])
            >>> x = pygm.sinkhorn(s_2d, unmatch1=unmatch1, unmatch2=unmatch2, max_iter=40)
            >>> x
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.12434101, 0.23913991, 0.05663597, 0.13943479, 0.31811425],
                    [0.03084473, 0.01085787, 0.12689067, 0.02784578, 0.32605890],
                    [0.03192548, 0.00745004, 0.13391025, 0.16087345, 0.12289304],
                    [0.29820536, 0.01659601, 0.32997174, 0.06988242, 0.10573396],
                    [0.29787774, 0.03223560, 0.08654936, 0.22023996, 0.06619393]])
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [0.87766593, 0.52249794, 0.45705226, 0.82038949, 0.70309659])
            col_sum: Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [0.78319431, 0.30627943, 0.73395800, 0.61827641, 0.93899407])

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.set_backend('jittor')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = pygm.utils.from_numpy(np.random.rand(5, 5))
            >>> s_2d
            jt.Var([[0.5488135  0.71518934 0.60276335 0.5448832  0.4236548 ]
                    [0.6458941  0.4375872  0.891773   0.96366274 0.3834415 ]
                    [0.79172504 0.5288949  0.56804454 0.92559665 0.07103606]
                    [0.0871293  0.0202184  0.83261985 0.77815676 0.87001216]
                    [0.9786183  0.7991586  0.46147937 0.7805292  0.11827443]], dtype=float32)
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            jt.Var([[0.18880227 0.24990915 0.19202219 0.1603428  0.20892365]
                    [0.18945065 0.17240447 0.23345011 0.22194763 0.18274714]
                    [0.23713583 0.20434798 0.18271242 0.23114584 0.1446579 ]
                    [0.11731039 0.1229692  0.23823905 0.19961584 0.3218654 ]
                    [0.2673009  0.2503692  0.1535762  0.1869479  0.1418058 ]], dtype=float32)
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: jt.Var([1.0000001  0.99999994 1.        0.9999999  1.       ], dtype=float32)
            col_sum: jt.Var([1.         1.         1.        1.         0.9999999], dtype=float32)

            # 3-dimensional (batched) input
            >>> s_3d = pygm.utils.from_numpy(np.random.rand(3, 5, 5))
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', x.sum(2))
            row_sum: jt.Var([[1.0000001  0.9999999  0.99999994 1.         0.99999994]
                            [1.         1.0000001  1.         0.99999994 1.        ]
                            [1.         1.         0.99999994 0.99999994 1.        ]], dtype=float32)
            >>> print('col_sum:', x.sum(1))
            col_sum: jt.Var([[1.         0.99999994 1.         0.99999994 1.        ]
                            [1.         1.         1.0000001  1.         0.9999999 ]
                            [0.99999994 1.0000001  0.9999999  1.         1.        ]], dtype=float32)

            # If the 3-d tensor are with different number of nodes
            >>> n1 = jt.Var([3, 4, 5])
            >>> n2 = jt.Var([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            jt.Var([[0.3666593  0.21498157 0.41835907 0.         0.        ]
                    [0.2760362  0.44270205 0.28126174 0.         0.        ]
                    [0.35730445 0.34231633 0.30037922 0.         0.        ]
                    [0.         0.         0.         0.         0.        ]
                    [0.         0.         0.         0.         0.        ]], dtype=float32)
            >>> x[1] # non-zero size: 4x4
            jt.Var([[0.28847834 0.20583051 0.34242094 0.16327024 0.        ]
                    [0.22656752 0.3015302  0.1940797  0.2778226  0.        ]
                    [0.2534638  0.1964985  0.32565048 0.22438715 0.        ]
                    [0.23149039 0.2961407  0.13784888 0.33452    0.        ]
                    [0.         0.         0.         0.         0.        ]], dtype=float32)
            >>> x[2] # non-zero size: 5x5
            jt.Var([[0.20147353 0.19541988 0.24942797 0.17346397 0.18021466]
                    [0.21050733 0.1762095  0.18645467 0.20384683 0.22298168]
                    [0.18319622 0.18024008 0.17619869 0.16641329 0.2939517 ]
                    [0.20754376 0.2236443  0.19658099 0.20570846 0.16652244]
                    [0.19727917 0.2244863  0.1913376  0.25056744 0.13632952]], dtype=float32)

            # non-squared input
            >>> s_non_square = pygm.utils.from_numpy(np.random.rand(4, 5))
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: jt.Var([1.         1.         1.         0.99999994], dtype=float32)
            col_sum: jt.Var([0.78239614 0.8048552  0.80165625 0.8000425  0.8110498], dtype=float32)

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = pygm.utils.from_numpy(np.random.randn(5, 5))
            >>> s_2d
            jt.Var([[ 0.01050002  1.7858706   0.12691209  0.40198937  1.8831507 ]
                    [-1.347759   -1.270485    0.9693967  -1.1731234   1.9436212 ]
                    [-0.41361898 -0.7474548   1.922942    1.4805148   1.867559  ]
                    [ 0.90604466 -0.86122566  1.9100649  -0.26800337  0.8024564 ]
                    [ 0.947252   -0.15501009  0.61407936  0.9222067   0.37642553]], dtype=float32)
            >>> unmatch1 = pygm.utils.from_numpy(np.random.randn(5))
            >>> unmatch1
            jt.Var([-1.0994008   0.2982382   1.3263859  -0.69456786 -0.14963454], dtype=float32)
            >>> unmatch2 = pygm.utils.from_numpy(np.random.randn(5))
            >>> unmatch2
            jt.Var([-0.43515354  1.8492638   0.67229474  0.40746182 -0.76991606], dtype=float32)
            >>> x = pygm.sinkhorn(s_2d, unmatch1=unmatch1, unmatch2=unmatch2, max_iter=40)
            >>> x
            jt.Var([[0.12434097 0.23913991 0.05663597 0.13943481 0.3181142 ]
                    [0.03084473 0.01085788 0.12689069 0.02784578 0.32605886]
                    [0.03192548 0.00745005 0.13391027 0.16087341 0.12289305]
                    [0.2982054  0.01659602 0.32997176 0.06988242 0.10573398]
                    [0.29787776 0.0322356  0.08654935 0.22023994 0.06619392]], dtype=float32)
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: jt.Var([0.8776659  0.52249795 0.45705223 0.8203896  0.70309657], dtype=float32) col_sum: jt.Var([0.7831943  0.30627945 0.73395807 0.61827636 0.938994  ], dtype=float32)

    .. dropdown:: MindSpore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.set_backend('mindspore')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = mindspore.Tensor(np.random.rand(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[5.48813504e-001, 7.15189366e-001, 6.02763376e-001, 5.44883183e-001, 4.23654799e-001],
                 [6.45894113e-001, 4.37587211e-001, 8.91773001e-001, 9.63662761e-001, 3.83441519e-001],
                 [7.91725038e-001, 5.28894920e-001, 5.68044561e-001, 9.25596638e-001, 7.10360582e-002],
                 [8.71292997e-002, 2.02183974e-002, 8.32619846e-001, 7.78156751e-001, 8.70012148e-001],
                 [9.78618342e-001, 7.99158564e-001, 4.61479362e-001, 7.80529176e-001, 1.18274426e-001]])
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[1.88802237e-001, 2.49909146e-001, 1.92022173e-001, 1.60342782e-001, 2.08923658e-001],
                 [1.89450662e-001, 1.72404455e-001, 2.33450110e-001, 2.21947620e-001, 1.82747159e-001],
                 [2.37135825e-001, 2.04348002e-001, 1.82712427e-001, 2.31145830e-001, 1.44657896e-001],
                 [1.17310392e-001, 1.22969199e-001, 2.38239095e-001, 1.99615882e-001, 3.21865485e-001],
                 [2.67300884e-001, 2.50369198e-001, 1.53576195e-001, 1.86947886e-001, 1.41805802e-001]])
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: [1.         1.00000001 0.99999998 1.00000005 0.99999997] col_sum: [1. 1. 1. 1. 1.]

            # 3-dimensional (batched) input
            >>> s_3d = mindspore.Tensor(np.random.rand(3, 5, 5))
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', x.sum(2))
            row_sum: [[1.         1.         1.         1.         1.        ]
                      [0.99999998 1.00000002 0.99999999 1.00000003 0.99999999]
                      [1.         1.         1.         1.         1.        ]]
            >>> print('col_sum:', x.sum(1))
            col_sum: [[1. 1. 1. 1. 1.]
                      [1. 1. 1. 1. 1.]
                      [1. 1. 1. 1. 1.]]

            # If the 3-d tensor are with different number of nodes
            >>> n1 = mindspore.Tensor([3, 4, 5])
            >>> n2 = mindspore.Tensor([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[3.66659344e-001, 2.14981580e-001, 4.18359055e-001, 0.00000000e+000, 0.00000000e+000],
                 [2.76036207e-001, 4.42702065e-001, 2.81261746e-001, 0.00000000e+000, 0.00000000e+000],
                 [3.57304449e-001, 3.42316355e-001, 3.00379198e-001, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]])
            >>> x[1] # non-zero size: 4x4
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[2.88478308e-001, 2.05830510e-001, 3.42420911e-001, 1.63270208e-001, 0.00000000e+000],
                 [2.26567517e-001, 3.01530213e-001, 1.94079686e-001, 2.77822621e-001, 0.00000000e+000],
                 [2.53463783e-001, 1.96498526e-001, 3.25650495e-001, 2.24387154e-001, 0.00000000e+000],
                 [2.31490392e-001, 2.96140751e-001, 1.37848909e-001, 3.34520016e-001, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]])
            >>> x[2] # non-zero size: 5x5
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[2.01473521e-001, 1.95419860e-001, 2.49427981e-001, 1.73463970e-001, 1.80214669e-001],
                 [2.10507324e-001, 1.76209477e-001, 1.86454688e-001, 2.03846840e-001, 2.22981672e-001],
                 [1.83196232e-001, 1.80240070e-001, 1.76198709e-001, 1.66413296e-001, 2.93951694e-001],
                 [2.07543757e-001, 2.23644304e-001, 1.96581006e-001, 2.05708473e-001, 1.66522460e-001],
                 [1.97279167e-001, 2.24486289e-001, 1.91337616e-001, 2.50567421e-001, 1.36329506e-001]])

            # non-squared input
            >>> s_non_square = mindspore.Tensor(np.random.rand(4, 5))
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: [1. 1. 1. 1.] col_sum: [0.78239609 0.80485526 0.80165627 0.80004254 0.81104984]

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = mindspore.Tensor(np.random.randn(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[1.05000207e-002, 1.78587049e+000, 1.26912093e-001, 4.01989363e-001, 1.88315070e+000],
                 [-1.34775906e+000, -1.27048500e+000, 9.69396708e-001, -1.17312341e+000, 1.94362119e+000],
                 [-4.13618981e-001, -7.47454811e-001, 1.92294203e+000, 1.48051479e+000, 1.86755896e+000],
                 [9.06044658e-001, -8.61225685e-001, 1.91006495e+000, -2.68003371e-001, 8.02456396e-001],
                 [9.47251968e-001, -1.55010093e-001, 6.14079370e-001, 9.22206672e-001, 3.76425531e-001]])
            >>> unmatch1 = mindspore.Tensor(np.random.randn(5))
            >>> unmatch1
            Tensor(shape=[5], dtype=Float64, value= [-1.09940079e+000, 2.98238174e-001, 1.32638590e+000, -6.94567860e-001, -1.49634540e-001])
            >>> unmatch2 = mindspore.Tensor(np.random.randn(5))
            >>> unmatch2
            Tensor(shape=[5], dtype=Float64, value= [-4.35153552e-001, 1.84926373e+000, 6.72294757e-001, 4.07461836e-001, -7.69916074e-001])
            >>> x = pygm.sinkhorn(s_2d, unmatch1=unmatch1, unmatch2=unmatch2, max_iter=40)
            >>> x
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[1.24341010e-001, 2.39139910e-001, 5.66359709e-002, 1.39434794e-001, 3.18114246e-001],
                 [3.08447251e-002, 1.08578709e-002, 1.26890674e-001, 2.78457752e-002, 3.26058897e-001],
                 [3.19254796e-002, 7.45003720e-003, 1.33910250e-001, 1.60873453e-001, 1.22893042e-001],
                 [2.98205355e-001, 1.65960124e-002, 3.29971742e-001, 6.98824247e-002, 1.05733960e-001],
                 [2.97877737e-001, 3.22356021e-002, 8.65493605e-002, 2.20239960e-001, 6.61939270e-002]])
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: [0.87766593 0.52249794 0.45705226 0.82038949 0.70309659] col_sum: [0.78319431 0.30627943 0.733958   0.61827641 0.93899407]

    .. dropdown:: Tensorflow Example

        ::

            >>> import tensorflow as tf
            >>> import pygmtools as pygm
            >>> pygm.set_backend('tensorflow')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = tf.constant(np.random.rand(5, 5))
            >>> s_2d
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ],
                   [0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152],
                   [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                   [0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215],
                   [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]])>
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.18880224, 0.24990915, 0.19202217, 0.16034278, 0.20892366],
                   [0.18945066, 0.17240445, 0.23345011, 0.22194762, 0.18274716],
                   [0.23713583, 0.204348  , 0.18271243, 0.23114583, 0.1446579 ],
                   [0.11731039, 0.1229692 , 0.23823909, 0.19961588, 0.32186549],
                   [0.26730088, 0.2503692 , 0.15357619, 0.18694789, 0.1418058 ]])>
            >>> print('row_sum:', tf.reduce_sum(x,axis=1), 'col_sum:', tf.reduce_sum(x, axis=0))
            row_sum: tf.Tensor([1.         1.00000001 0.99999998 1.00000005 0.99999997], shape=(5,), dtype=float64) col_sum: tf.Tensor([1. 1. 1. 1. 1.], shape=(5,), dtype=float64)

            # 3-dimensional (batched) input
            >>> s_3d = tf.constant(np.random.rand(3, 5, 5))
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', tf.reduce_sum(x, axis=2))
            row_sum: tf.Tensor(
            [[1.         1.         1.         1.         1.        ]
             [0.99999998 1.00000002 0.99999999 1.00000003 0.99999999]
             [1.         1.         1.         1.         1.        ]], shape=(3, 5), dtype=float64)
            >>> print('col_sum:', tf.reduce_sum(x, axis=1))
            col_sum: tf.Tensor(
            [[1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1.]
             [1. 1. 1. 1. 1.]], shape=(3, 5), dtype=float64)

            # If the 3-d tensor are with different number of nodes
            >>> n1 = tf.constant([3, 4, 5])
            >>> n2 = tf.constant([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.36665934, 0.21498158, 0.41835906, 0.        , 0.        ],
                   [0.27603621, 0.44270207, 0.28126175, 0.        , 0.        ],
                   [0.35730445, 0.34231636, 0.3003792 , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.        ]])>
            >>> x[1] # non-zero size: 4x4
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.28847831, 0.20583051, 0.34242091, 0.16327021, 0.        ],
                   [0.22656752, 0.30153021, 0.19407969, 0.27782262, 0.        ],
                   [0.25346378, 0.19649853, 0.32565049, 0.22438715, 0.        ],
                   [0.23149039, 0.29614075, 0.13784891, 0.33452002, 0.        ],
                   [0.        , 0.        , 0.        , 0.        , 0.        ]])>
            >>> x[2] # non-zero size: 5x5
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.20147352, 0.19541986, 0.24942798, 0.17346397, 0.18021467],
                   [0.21050732, 0.17620948, 0.18645469, 0.20384684, 0.22298167],
                   [0.18319623, 0.18024007, 0.17619871, 0.1664133 , 0.29395169],
                   [0.20754376, 0.2236443 , 0.19658101, 0.20570847, 0.16652246],
                   [0.19727917, 0.22448629, 0.19133762, 0.25056742, 0.13632951]])>

            # non-squared input
            >>> s_non_square = tf.constant(np.random.rand(4, 5))
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', tf.reduce_sum(x,axis=1),  'col_sum:', tf.reduce_sum(x,axis=0))
            row_sum: tf.Tensor([1. 1. 1. 1.], shape=(4,), dtype=float64) col_sum: tf.Tensor([0.78239609 0.80485526 0.80165627 0.80004254 0.81104984], shape=(5,), dtype=float64)

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = tf.constant(np.random.randn(5, 5))
            >>> s_2d
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[ 0.01050002,  1.78587049,  0.12691209,  0.40198936,  1.8831507 ],
                   [-1.34775906, -1.270485  ,  0.96939671, -1.17312341,  1.94362119],
                   [-0.41361898, -0.74745481,  1.92294203,  1.48051479,  1.86755896],
                   [ 0.90604466, -0.86122569,  1.91006495, -0.26800337,  0.8024564 ],
                   [ 0.94725197, -0.15501009,  0.61407937,  0.92220667,  0.37642553]])>
            >>> unmatch1 = tf.constant(np.random.randn(5))
            >>> unmatch1
            <tf.Tensor: shape=(5,), dtype=float64, numpy=array([-1.09940079,  0.29823817,  1.3263859 , -0.69456786, -0.14963454])>
            >>> unmatch2 = tf.constant(np.random.randn(5))
            >>> unmatch2
            <tf.Tensor: shape=(5,), dtype=float64, numpy=array([-0.43515355,  1.84926373,  0.67229476,  0.40746184, -0.76991607])>
            >>> x = pygm.sinkhorn(s_2d, unmatch1=unmatch1, unmatch2=unmatch2, max_iter=40)
            >>> x
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.12434101, 0.23913991, 0.05663597, 0.13943479, 0.31811425],
                   [0.03084473, 0.01085787, 0.12689067, 0.02784578, 0.3260589 ],
                   [0.03192548, 0.00745004, 0.13391025, 0.16087345, 0.12289304],
                   [0.29820536, 0.01659601, 0.32997174, 0.06988242, 0.10573396],
                   [0.29787774, 0.0322356 , 0.08654936, 0.22023996, 0.06619393]])>
            >>> print('row_sum:', tf.reduce_sum(x, axis=1), 'col_sum:', tf.reduce_sum(x, axis=0))
            row_sum: tf.Tensor([0.87766593 0.52249794 0.45705226 0.82038949 0.70309659], shape=(5,), dtype=float64) col_sum: tf.Tensor([0.78319431 0.30627943 0.733958   0.61827641 0.93899407], shape=(5,), dtype=float64)

    .. note::

        If you find this graph matching solver useful for your research, please cite:

        ::

            @article{sinkhorn,
              title={Concerning nonnegative matrices and doubly stochastic matrices},
              author={Sinkhorn, Richard and Knopp, Paul},
              journal={Pacific Journal of Mathematics},
              volume={21},
              number={2},
              pages={343--348},
              year={1967},
              publisher={Mathematical Sciences Publishers}
            }
    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(s, 's', backend)
    if _check_shape(s, 2, backend):
        s = _unsqueeze(s, 0, backend)
        if isinstance(n1, (int, np.integer)): n1 = from_numpy(np.array([n1]), backend=backend)
        if isinstance(n2, (int, np.integer)): n2 = from_numpy(np.array([n2]), backend=backend)
        non_batched_input = True
    elif _check_shape(s, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument s is expected to be 2-dimensional or 3-dimensional, got '
                         f's:{len(_get_shape(s, backend))}dims!')
    if n1 is not None: _check_data_type(n1, 'n1', backend)
    if n2 is not None: _check_data_type(n2, 'n2', backend)
    if unmatch1 is not None and unmatch2 is not None:
        _check_data_type(unmatch1, 'unmatch1', backend)
        _check_data_type(unmatch2, 'unmatch2', backend)
        if non_batched_input:
            unmatch1 = _unsqueeze(unmatch1, 0, backend)
            unmatch2 = _unsqueeze(unmatch2, 0, backend)
        if not _check_shape(unmatch1, 2, backend) or not _check_shape(unmatch2, 2, backend):
            raise ValueError(f'the input arguments unmatch1 and unmatch2 are illegal. They should be 2-dim'
                             f'for batched input, and 1-dim for non-batched input.')
        if not all((_get_shape(unmatch1, backend)[1] == _get_shape(s, backend)[1],
                    _get_shape(unmatch2, backend)[1] == _get_shape(s, backend)[2],
                    _get_shape(unmatch1, backend)[0] == _get_shape(unmatch2, backend)[0] == _get_shape(s, backend)[0])):
            raise ValueError(f'the shapes of the following arguments mismatch. '
                             f'Please read the doc for the correct shape.\n'
                             f'Got s:{_get_shape(s, backend)}, unmatch1:{_get_shape(unmatch1, backend)}, '
                             f'unmatch2:{_get_shape(unmatch2, backend)}!')
    elif unmatch1 is None and unmatch2 is None:
        pass
    else:
        raise ValueError('The arguments unmatch1 and unmatch2 must be specified together.')
    args = (s, n1, n2, unmatch1, unmatch2, dummy_row, max_iter, tau, batched_operation)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.sinkhorn
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def hungarian(s, n1=None, n2=None, unmatch1=None, unmatch2=None,
              nproc: int = 1,
              backend=None):
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size. Non-batched input is also
              supported if ``s`` is of size :math:`(n_1 \times n_2)`
    :param n1: :math:`(b)` (optional) number of objects in dim1
    :param n2: :math:`(b)` (optional) number of objects in dim2
    :param unmatch1: (optional, new in ``0.3.0``) :math:`(b\times n_1)` the scores indicating the objects in dim1 is unmatched
    :param unmatch2: (optional, new in ``0.3.0``) :math:`(b\times n_2)` the scores indicating the objects in dim2 is unmatched
    :param nproc: (default: 1, i.e. no parallel) number of parallel processes
    :param backend: (default: ``pygmtools.BACKEND`` variable) the backend for computation.
    :return: :math:`(b\times n_1 \times n_2)` optimal permutation matrix

    .. note::
        The parallelization is based on multi-processing workers that run on multiple CPU cores.

    .. note::
        For all backends, ``scipy.optimize.linear_sum_assignment`` is called to solve the LAP, therefore the
        computation is based on ``numpy`` and ``scipy``. The ``backend`` argument of this function only affects
        the input-output data type.

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded and all elements in ``n1`` are equal, all in ``n2`` are equal.

    .. warning::
        This function can work with or without maximal inlier matching:

        * **With maximal inlier matching** (the default mode). If ``unmatch1=None`` and ``unmatch2=None``,
          the solver aims to match as many nodes as possible. The corresponding linear assignment problem is

          .. math::

              &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S})\\
              s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}

          where the constraint :math:`\mathbf{X}\mathbf{1} = \mathbf{1}` urges the solver to match as many inlier
          nodes as possible.
        * **Without maximal inlier matching** (new in ``0.3.0``). If ``unmatch1`` and ``unmatch2`` are not ``None``,
          the solver is allowed to match nodes to void nodes, and the corresponding scores for matching to void nodes
          are specified by ``unmatch1`` and ``unmatch2``. The following (modified) linear assignment problem is
          considered:

          .. math::
              &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S}^\prime)\\
              s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1+1\times n_2+1}, \ \mathbf{X}_{[0:n_1, :]}\mathbf{1} = \mathbf{1}, \ \mathbf{X}_{[:, 0:n_2]}^\top\mathbf{1} \leq \mathbf{1}

          where the last column and last row of :math:`\mathbf{S}^\prime` are ``unmatch1`` and ``unmatch2``,
          respectively.

          For example, if you want to solve the following problem (note that both consrtraints are :math:`\leq`)

          .. math::

              &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{S})\\
              s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} \leq \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}

          you can simply set ``unmatch1`` and ``unmatch2`` as zero vectors.

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.set_backend('numpy')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = np.random.rand(5, 5)
            >>> s_2d
            array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ],
                   [0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152],
                   [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                   [0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215],
                   [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]])

            >>> x = pygm.hungarian(s_2d)
            >>> x
            array([[0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., 0.],
                   [0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 0.]])

            # 3-dimensional (batched) input
            >>> s_3d = np.random.rand(3, 5, 5)
            >>> n1 = n2 = np.array([3, 4, 5])
            >>> x = pygm.hungarian(s_3d, n1, n2)
            >>> x
            array([[[0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                   [[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                   [[0., 0., 1., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0.]]])

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = np.random.randn(5, 5)
            >>> s_2d
            array([[-1.16514984,  0.90082649,  0.46566244, -1.53624369,  1.48825219],
                   [ 1.89588918,  1.17877957, -0.17992484, -1.07075262,  1.05445173],
                   [-0.40317695,  1.22244507,  0.20827498,  0.97663904,  0.3563664 ],
                   [ 0.70657317,  0.01050002,  1.78587049,  0.12691209,  0.40198936],
                   [ 1.8831507 , -1.34775906, -1.270485  ,  0.96939671, -1.17312341]])
            >>> unmatch1 = np.random.randn(5)
            >>> unmatch1
            array([ 1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479])
            >>> unmatch2 = np.random.randn(5)
            >>> unmatch2
            array([ 1.86755896,  0.90604466, -0.86122569,  1.91006495, -0.26800337])
            >>> x = pygm.hungarian(s_2d, unmatch1=unmatch1, unmatch2=unmatch2)
            >>> x
            array([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1.],
                   [0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.set_backend('pytorch')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = torch.from_numpy(np.random.rand(5, 5))
            >>> s_2d
            tensor([[0.5488, 0.7152, 0.6028, 0.5449, 0.4237],
                    [0.6459, 0.4376, 0.8918, 0.9637, 0.3834],
                    [0.7917, 0.5289, 0.5680, 0.9256, 0.0710],
                    [0.0871, 0.0202, 0.8326, 0.7782, 0.8700],
                    [0.9786, 0.7992, 0.4615, 0.7805, 0.1183]], dtype=torch.float64)
            >>> x = pygm.hungarian(s_2d)
            >>> x
            tensor([[0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.],
                    [1., 0., 0., 0., 0.]], dtype=torch.float64)

            # 3-dimensional (batched) input
            >>> s_3d = torch.from_numpy(np.random.rand(3, 5, 5))
            >>> n1 = n2 = torch.tensor([3, 4, 5])
            >>> x = pygm.hungarian(s_3d, n1, n2)
            >>> x
            tensor([[[0., 0., 1., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[1., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [0., 0., 1., 0., 0.],
                     [0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[0., 0., 1., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1.],
                     [0., 1., 0., 0., 0.],
                     [0., 0., 0., 1., 0.]]], dtype=torch.float64)

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = torch.from_numpy(np.random.randn(5, 5))
            >>> s_2d
            tensor([[-1.1651,  0.9008,  0.4657, -1.5362,  1.4883],
                    [ 1.8959,  1.1788, -0.1799, -1.0708,  1.0545],
                    [-0.4032,  1.2224,  0.2083,  0.9766,  0.3564],
                    [ 0.7066,  0.0105,  1.7859,  0.1269,  0.4020],
                    [ 1.8832, -1.3478, -1.2705,  0.9694, -1.1731]], dtype=torch.float64)
            >>> unmatch1 = torch.from_numpy(np.random.randn(5))
            >>> unmatch1
            tensor([ 1.9436, -0.4136, -0.7475,  1.9229,  1.4805], dtype=torch.float64)
            >>> unmatch2 = torch.from_numpy(np.random.randn(5))
            >>> unmatch2
            tensor([ 1.8676,  0.9060, -0.8612,  1.9101, -0.2680], dtype=torch.float64)
            >>> x = pygm.hungarian(s_2d, unmatch1=unmatch1, unmatch2=unmatch2)
            >>> x
            tensor([[0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]], dtype=torch.float64)

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.set_backend('paddle')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = paddle.to_tensor(np.random.rand(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0.54881350, 0.71518937, 0.60276338, 0.54488318, 0.42365480],
                    [0.64589411, 0.43758721, 0.89177300, 0.96366276, 0.38344152],
                    [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                    [0.08712930, 0.02021840, 0.83261985, 0.77815675, 0.87001215],
                    [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]])
            >>> x = pygm.hungarian(s_2d)
            >>> x
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.],
                    [1., 0., 0., 0., 0.]])

            # 3-dimensional (batched) input
            >>> s_3d = paddle.to_tensor(np.random.rand(3, 5, 5))
            >>> n1 = n2 = paddle.to_tensor([3, 4, 5])
            >>> x = pygm.hungarian(s_3d, n1, n2)
            >>> x
            Tensor(shape=[3, 5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[[0., 0., 1., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[1., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0.],
                     [0., 0., 1., 0., 0.],
                     [0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                    [[0., 0., 1., 0., 0.],
                     [1., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 1.],
                     [0., 1., 0., 0., 0.],
                     [0., 0., 0., 1., 0.]]])


            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = paddle.to_tensor(np.random.randn(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[-1.16514984,  0.90082649,  0.46566244, -1.53624369,  1.48825219],
                    [ 1.89588918,  1.17877957, -0.17992484, -1.07075262,  1.05445173],
                    [-0.40317695,  1.22244507,  0.20827498,  0.97663904,  0.35636640],
                    [ 0.70657317,  0.01050002,  1.78587049,  0.12691209,  0.40198936],
                    [ 1.88315070, -1.34775906, -1.27048500,  0.96939671, -1.17312341]])
            >>> unmatch1 = paddle.to_tensor(np.random.randn(5))
            >>> unmatch1
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [ 1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479])
            >>> unmatch2 = paddle.to_tensor(np.random.randn(5))
            >>> unmatch2
            Tensor(shape=[5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [ 1.86755896,  0.90604466, -0.86122569,  1.91006495, -0.26800337])
            >>> x = pygm.hungarian(s_2d, unmatch1=unmatch1, unmatch2=unmatch2)
            >>> x
            Tensor(shape=[5, 5], dtype=float64, place=Place(cpu), stop_gradient=True,
                   [[0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]])

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.set_backend('jittor')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = pygm.utils.from_numpy(np.random.rand(5, 5))
            >>> s_2d
            jt.Var([[0.5488135  0.71518934 0.60276335 0.5448832  0.4236548 ]
                    [0.6458941  0.4375872  0.891773   0.96366274 0.3834415 ]
                    [0.79172504 0.5288949  0.56804454 0.92559665 0.07103606]
                    [0.0871293  0.0202184  0.83261985 0.77815676 0.87001216]
                    [0.9786183  0.7991586  0.46147937 0.7805292  0.11827443]], dtype=float32)
            >>> x = pygm.hungarian(s_2d)
            >>> x
            jt.Var([[0. 1. 0. 0. 0.]
                    [0. 0. 1. 0. 0.]
                    [0. 0. 0. 1. 0.]
                    [0. 0. 0. 0. 1.]
                    [1. 0. 0. 0. 0.]], dtype=float32)

            # 3-dimensional (batched) input
            >>> s_3d = pygm.utils.from_numpy(np.random.rand(3, 5, 5))
            >>> n1 = n2 = jt.Var([3, 4, 5])
            >>> x = pygm.hungarian(s_3d, n1, n2)
            >>> x
            jt.Var([[[0. 0. 1. 0. 0.]
                     [0. 1. 0. 0. 0.]
                     [1. 0. 0. 0. 0.]
                     [0. 0. 0. 0. 0.]
                     [0. 0. 0. 0. 0.]]
            <BLANKLINE>
                    [[1. 0. 0. 0. 0.]
                     [0. 1. 0. 0. 0.]
                     [0. 0. 1. 0. 0.]
                     [0. 0. 0. 1. 0.]
                     [0. 0. 0. 0. 0.]]
            <BLANKLINE>
                    [[0. 0. 1. 0. 0.]
                     [1. 0. 0. 0. 0.]
                     [0. 0. 0. 0. 1.]
                     [0. 1. 0. 0. 0.]
                     [0. 0. 0. 1. 0.]]], dtype=float32)

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = pygm.utils.from_numpy(np.random.randn(5, 5))
            >>> s_2d
            jt.Var([[-1.1651498   0.9008265   0.46566245 -1.5362437   1.4882522 ]
                    [ 1.8958892   1.1787796  -0.17992483 -1.0707526   1.0544517 ]
                    [-0.40317693  1.222445    0.20827498  0.97663903  0.3563664 ]
                    [ 0.7065732   0.01050002  1.7858706   0.12691209  0.40198937]
                    [ 1.8831507  -1.347759   -1.270485    0.9693967  -1.1731234 ]], dtype=float32)
            >>> unmatch1 = pygm.utils.from_numpy(np.random.randn(5))
            >>> unmatch1
            jt.Var([ 1.9436212  -0.41361898 -0.7474548   1.922942    1.4805148 ], dtype=float32)
            >>> unmatch2 = pygm.utils.from_numpy(np.random.randn(5))
            >>> unmatch2
            jt.Var([ 1.867559    0.90604466 -0.86122566  1.9100649  -0.26800337], dtype=float32)
            >>> x = pygm.hungarian(s_2d, unmatch1=unmatch1, unmatch2=unmatch2)
            >>> x
            jt.Var([[0. 0. 0. 0. 0.]
                    [0. 0. 0. 0. 1.]
                    [0. 0. 1. 0. 0.]
                    [0. 0. 0. 0. 0.]
                    [0. 0. 0. 0. 0.]], dtype=float32)

    .. dropdown:: MindSpore Example

        ::

            >>> import mindspore
            >>> import pygmtools as pygm
            >>> pygm.set_backend('mindspore')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = mindspore.Tensor(np.random.rand(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[5.48813504e-001, 7.15189366e-001, 6.02763376e-001, 5.44883183e-001, 4.23654799e-001],
                 [6.45894113e-001, 4.37587211e-001, 8.91773001e-001, 9.63662761e-001, 3.83441519e-001],
                 [7.91725038e-001, 5.28894920e-001, 5.68044561e-001, 9.25596638e-001, 7.10360582e-002],
                 [8.71292997e-002, 2.02183974e-002, 8.32619846e-001, 7.78156751e-001, 8.70012148e-001],
                 [9.78618342e-001, 7.99158564e-001, 4.61479362e-001, 7.80529176e-001, 1.18274426e-001]])
            >>> x = pygm.hungarian(s_2d)
            >>> x
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000],
                 [1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]])

            # 3-dimensional (batched) input
            >>> s_3d = mindspore.Tensor(np.random.rand(3, 5, 5))
            >>> n1 = n2 = mindspore.Tensor([3, 4, 5])
            >>> x = pygm.hungarian(s_3d, n1, n2)
            >>> x
            Tensor(shape=[3, 5, 5], dtype=Float64, value=
                [[[0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]],
                 [[1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]],
                 [[0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000],
                  [0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                  [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000]]])

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = mindspore.Tensor(np.random.randn(5, 5))
            >>> s_2d
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[-1.16514984e+000, 9.00826487e-001, 4.65662440e-001, -1.53624369e+000, 1.48825219e+000],
                 [1.89588918e+000, 1.17877957e+000, -1.79924836e-001, -1.07075262e+000, 1.05445173e+000],
                 [-4.03176947e-001, 1.22244507e+000, 2.08274978e-001, 9.76639036e-001, 3.56366397e-001],
                 [7.06573168e-001, 1.05000207e-002, 1.78587049e+000, 1.26912093e-001, 4.01989363e-001],
                 [1.88315070e+000, -1.34775906e+000, -1.27048500e+000, 9.69396708e-001, -1.17312341e+000]])
            >>> unmatch1 = mindspore.Tensor(np.random.randn(5))
            >>> unmatch1
            Tensor(shape=[5], dtype=Float64, value= [1.94362119e+000, -4.13618981e-001, -7.47454811e-001, 1.92294203e+000, 1.48051479e+000])
            >>> unmatch2 = mindspore.Tensor(np.random.randn(5))
            >>> unmatch2
            Tensor(shape=[5], dtype=Float64, value= [1.86755896e+000, 9.06044658e-001, -8.61225685e-001, 1.91006495e+000, -2.68003371e-001])
            >>> x = pygm.hungarian(s_2d, unmatch1=unmatch1, unmatch2=unmatch2)
            >>> x
            Tensor(shape=[5, 5], dtype=Float64, value=
                [[0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 1.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                 [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000]])

    .. dropdown:: Tensorflow Example

        ::

            >>> import tensorflow as tf
            >>> import pygmtools as pygm
            >>> pygm.set_backend('tensorflow')
            >>> np.random.seed(0)

            # 2-dimensional (non-batched) input
            >>> s_2d = tf.constant(np.random.rand(5, 5))
            >>> s_2d
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ],
                   [0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152],
                   [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                   [0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215],
                   [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]])>
            >>> x = pygm.hungarian(s_2d)
            >>> x
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., 0.],
                   [0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 1.],
                   [1., 0., 0., 0., 0.]])>

            # 3-dimensional (batched) input
            >>> s_3d = tf.constant(np.random.rand(3, 5, 5))
            >>> n1 = n2 = tf.constant([3, 4, 5])
            >>> x = pygm.hungarian(s_3d, n1, n2)
            >>> x
            <tf.Tensor: shape=(3, 5, 5), dtype=float64, numpy=
            array([[[0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                   [[1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0.]],
            <BLANKLINE>
                   [[0., 0., 1., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0.]]])>

            # allow matching to void nodes by setting unmatch1 and unmatch2
            >>> s_2d = tf.constant(np.random.randn(5, 5))
            >>> s_2d
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[-1.16514984,  0.90082649,  0.46566244, -1.53624369,  1.48825219],
                   [ 1.89588918,  1.17877957, -0.17992484, -1.07075262,  1.05445173],
                   [-0.40317695,  1.22244507,  0.20827498,  0.97663904,  0.3563664 ],
                   [ 0.70657317,  0.01050002,  1.78587049,  0.12691209,  0.40198936],
                   [ 1.8831507 , -1.34775906, -1.270485  ,  0.96939671, -1.17312341]])>
            >>> unmatch1 = tf.constant(np.random.randn(5))
            >>> unmatch1
            <tf.Tensor: shape=(5,), dtype=float64, numpy=array([ 1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479])>
            >>> unmatch2 = tf.constant(np.random.randn(5))
            >>> unmatch2
            <tf.Tensor: shape=(5,), dtype=float64, numpy=array([ 1.86755896,  0.90604466, -0.86122569,  1.91006495, -0.26800337])>
            >>> x = pygm.hungarian(s_2d, unmatch1=unmatch1, unmatch2=unmatch2)
            >>> x
            <tf.Tensor: shape=(5, 5), dtype=float64, numpy=
            array([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 1.],
                   [0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])>

    .. note::

        If you find this graph matching solver useful for your research, please cite:

        ::

            @article{hungarian,
              title={Algorithms for the assignment and transportation problems},
              author={Munkres, James},
              journal={Journal of the society for industrial and applied mathematics},
              volume={5},
              number={1},
              pages={32--38},
              year={1957},
              publisher={SIAM}
            }

    """
    if backend is None:
        backend = pygmtools.BACKEND
    _check_data_type(s, backend)
    if _check_shape(s, 2, backend):
        s = _unsqueeze(s, 0, backend)
        if isinstance(n1, (int, np.integer)): n1 = from_numpy(np.array([n1]), backend=backend)
        if isinstance(n2, (int, np.integer)): n2 = from_numpy(np.array([n2]), backend=backend)
        non_batched_input = True
    elif _check_shape(s, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument s is expected to be 2-dimensional or 3-dimensional, got '
                         f's:{len(_get_shape(s, backend))}dims!')
    if unmatch1 is not None and unmatch2 is not None:
        _check_data_type(unmatch1, 'unmatch1', backend)
        _check_data_type(unmatch2, 'unmatch2', backend)
        if non_batched_input:
            unmatch1 = _unsqueeze(unmatch1, 0, backend)
            unmatch2 = _unsqueeze(unmatch2, 0, backend)
        if not _check_shape(unmatch1, 2, backend) or not _check_shape(unmatch2, 2, backend):
            raise ValueError(f'the input arguments unmatch1 and unmatch2 are illegal. They should be 2-dim'
                             f'for batched input, and 1-dim for non-batched input.')
        if not all((_get_shape(unmatch1, backend)[1] == _get_shape(s, backend)[1],
                    _get_shape(unmatch2, backend)[1] == _get_shape(s, backend)[2],
                    _get_shape(unmatch1, backend)[0] == _get_shape(unmatch2, backend)[0] == _get_shape(s, backend)[0])):
            raise ValueError(f'the shapes of the following arguments mismatch. '
                             f'Please read the doc for the correct shape.\n'
                             f'Got s:{_get_shape(s, backend)}, unmatch1:{_get_shape(unmatch1, backend)}, '
                             f'unmatch2:{_get_shape(unmatch2, backend)}!')
    elif unmatch1 is None and unmatch2 is None:
        pass
    else:
        raise ValueError('The arguments unmatch1 and unmatch2 must be specified together.')
    args = (s, n1, n2, unmatch1, unmatch2, nproc)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.hungarian
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result
