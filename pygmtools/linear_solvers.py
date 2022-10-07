r"""
Classic (learning-free) **linear assignment problem** solvers. These linear assignment solvers are recommended to solve
matching problems with only nodes (i.e. linear matching problems), or large-scale graph matching problems where the cost
of QAP formulation is too high.

The linear assignment problem only considers nodes, and is also known as bipartite graph matching and linear matching:

.. math::

    &\max_{\mathbf{X}} \ \texttt{tr}(\mathbf{X}^\top \mathbf{M})\\
    s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}
"""

import importlib
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type


def sinkhorn(s, n1=None, n2=None,
             dummy_row: bool = False, max_iter: int = 10, tau: float = 1., batched_operation: bool = False,
             backend=None):
    r"""
    Sinkhorn algorithm turns the input matrix into a doubly-stochastic matrix.

    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:

    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)

    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:

    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)

    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size. Non-batched input is also
              supported if ``s`` is of size :math:`(n_1 \times n_2)`
    :param n1: (optional) :math:`(b)` number of objects in dim1
    :param n2: (optional) :math:`(b)` number of objects in dim2
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

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'
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

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
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
            row_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float64) col_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float64)

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

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'
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

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'jittor'
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
    _check_data_type(s, backend)
    if _check_shape(s, 2, backend):
        s = _unsqueeze(s, 0, backend)
        non_batched_input = True
    elif _check_shape(s, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument s is expected to be 2-dimensional or 3-dimensional, got '
                         f's:{len(_get_shape(s, backend))}dims!')

    args = (s, n1, n2, dummy_row, max_iter, tau, batched_operation)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.sinkhorn
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result


def hungarian(s, n1=None, n2=None,
              nproc: int = 1,
              backend=None):
    r"""
    Solve optimal LAP permutation by hungarian algorithm. The time cost is :math:`O(n^3)`.

    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size. Non-batched input is also
              supported if ``s`` is of size :math:`(n_1 \times n_2)`
    :param n1: :math:`(b)` (optional) number of objects in dim1
    :param n2: :math:`(b)` (optional) number of objects in dim2
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

    .. dropdown:: Numpy Example

        ::

            >>> import numpy as np
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'numpy'
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

    .. dropdown:: Pytorch Example

        ::

            >>> import torch
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'pytorch'
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

    .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'
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

    .. dropdown:: Jittor Example

        ::

            >>> import jittor as jt
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'jittor'
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
        non_batched_input = True
    elif _check_shape(s, 3, backend):
        non_batched_input = False
    else:
        raise ValueError(f'the input argument s is expected to be 2-dimensional or 3-dimensional, got '
                         f's:{len(_get_shape(s, backend))}dims!')

    args = (s, n1, n2, nproc)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.hungarian
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args)
    if non_batched_input:
        return _squeeze(result, 0, backend)
    else:
        return result
