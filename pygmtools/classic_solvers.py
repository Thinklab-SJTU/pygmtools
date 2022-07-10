import importlib
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type


def sinkhorn(s, n1=None, n2=None,
             dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False,
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

    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~pygmtools.classic_solvers.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.

    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient. It is assumed that ``row number <= column number``. If not, the input matrix will be
        transposed.

    .. note::
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.

    .. note::
        The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
        matched have different number of nodes, it is a common practice to add dummy rows to construct a square
        matrix. After the row and column normalizations, the padded rows are discarded.

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

            # 2-dimensional (non-batched) input
            >>> s_2d = torch.from_numpy(s_2d)
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
            >>> s_3d = torch.from_numpy(s_3d)
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
            >>> s_non_square = torch.from_numpy(s_non_square)
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000], dtype=torch.float64) col_sum: tensor([0.7824, 0.8049, 0.8017, 0.8000, 0.8110], dtype=torch.float64)

     .. dropdown:: Paddle Example

        ::

            >>> import paddle
            >>> import pygmtools as pygm
            >>> pygm.BACKEND = 'paddle'

            # 2-dimensional (non-batched) input
            >>> s_2d = paddle.to_tensor(s_2d)
            >>> s_2d
            tensor([[0.5488, 0.7152, 0.6028, 0.5449, 0.4237],
                    [0.6459, 0.4376, 0.8918, 0.9637, 0.3834],
                    [0.7917, 0.5289, 0.5680, 0.9256, 0.0710],
                    [0.0871, 0.0202, 0.8326, 0.7782, 0.8700],
                    [0.9786, 0.7992, 0.4615, 0.7805, 0.1183]], dtype=paddle.float64)
            >>> x = pygm.sinkhorn(s_2d)
            >>> x
            tensor([[0.1888, 0.2499, 0.1920, 0.1603, 0.2089],
                    [0.1895, 0.1724, 0.2335, 0.2219, 0.1827],
                    [0.2371, 0.2043, 0.1827, 0.2311, 0.1447],
                    [0.1173, 0.1230, 0.2382, 0.1996, 0.3219],
                    [0.2673, 0.2504, 0.1536, 0.1869, 0.1418]], dtype=paddle.float64)
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], dtype=paddle.float64) col_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], dtype=paddle.float64)

            # 3-dimensional (batched) input
            >>> s_3d = paddle.to_tensor(s_3d)
            >>> x = pygm.sinkhorn(s_3d)
            >>> print('row_sum:', x.sum(2))
            row_sum: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]], dtype=paddle.float64)
            >>> print('col_sum:', x.sum(1))
            col_sum: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]], dtype=paddle.float64)

            # If the 3-d tensor are with different number of nodes
            >>> n1 = paddle.to_tensor([3, 4, 5])
            >>> n2 = paddle.to_tensor([3, 4, 5])
            >>> x = pygm.sinkhorn(s_3d, n1, n2)
            >>> x[0] # non-zero size: 3x3
            tensor([[0.3667, 0.2150, 0.4184, 0.0000, 0.0000],
                    [0.2760, 0.4427, 0.2813, 0.0000, 0.0000],
                    [0.3573, 0.3423, 0.3004, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=paddle.float64)
            >>> x[1] # non-zero size: 4x4
            tensor([[0.2885, 0.2058, 0.3424, 0.1633, 0.0000],
                    [0.2266, 0.3015, 0.1941, 0.2778, 0.0000],
                    [0.2535, 0.1965, 0.3257, 0.2244, 0.0000],
                    [0.2315, 0.2961, 0.1378, 0.3345, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=paddle.float64)
            >>> x[2] # non-zero size: 5x5
            tensor([[0.2015, 0.1954, 0.2494, 0.1735, 0.1802],
                    [0.2105, 0.1762, 0.1865, 0.2038, 0.2230],
                    [0.1832, 0.1802, 0.1762, 0.1664, 0.2940],
                    [0.2075, 0.2236, 0.1966, 0.2057, 0.1665],
                    [0.1973, 0.2245, 0.1913, 0.2506, 0.1363]], dtype=paddle.float64)

            # non-squared input
            >>> s_non_square = paddle.to_tensor(s_non_square)
            >>> x = pygm.sinkhorn(s_non_square, dummy_row=True) # set dummy_row=True for non-squared cases
            >>> print('row_sum:', x.sum(1), 'col_sum:', x.sum(0))
            row_sum: tensor([1.0000, 1.0000, 1.0000, 1.0000], dtype=paddle.float64) col_sum: tensor([0.7824, 0.8049, 0.8017, 0.8000, 0.8110], dtype=paddle.float64)


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
                         f's:{len(_get_shape(s))}!')

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
              nproc: int=1,
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
        the batched matrices are not padded.

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
    
            # 2-dimensional (non-batched) input
            >>> s_2d = torch.from_numpy(s_2d)
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
            >>> s_3d = torch.from_numpy(s_3d)
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

            # 2-dimensional (non-batched) input
            >>> s_2d = paddle.to_tensor(s_2d)
            >>> s_2d
            tensor([[0.5488, 0.7152, 0.6028, 0.5449, 0.4237],
                    [0.6459, 0.4376, 0.8918, 0.9637, 0.3834],
                    [0.7917, 0.5289, 0.5680, 0.9256, 0.0710],
                    [0.0871, 0.0202, 0.8326, 0.7782, 0.8700],
                    [0.9786, 0.7992, 0.4615, 0.7805, 0.1183]], dtype=paddle.float64)
            >>> x = pygm.hungarian(s_2d)
            >>> x
            tensor([[0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.],
                    [1., 0., 0., 0., 0.]], dtype=paddle.float64)

            # 3-dimensional (batched) input
            >>> s_3d = paddle.to_tensor(s_3d)
            >>> n1 = n2 = paddle.tensor([3, 4, 5])
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
                    [0., 0., 0., 1., 0.]]], dtype=paddle.float64)

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
                         f's:{len(_get_shape(s))}!')

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
    __check_gm_arguments(n1, n2, n1max, n2max)
    args = (K, n1, n2, n1max, n2max, x0, max_iter)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.sm
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


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
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.

    .. note::
        This solver is differentiable and supports gradient back-propagation.

    .. warning::
        The solver's output is normalized with a sum of 1, which is in line with the original implementation. If a doubly-
        stochastic matrix is required, please call :func:`~pygmtools.classic_solvers.sinkhorn` after this. If a discrete
        permutation matrix is required, please call :func:`~pygmtools.classic_solvers.hungarian`. Note that the
        Hungarian algorithm will truncate the gradient.

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
            >>> _ = torch.seed(1)

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
    return fn(*args)


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
        We support batched instances with different number of nodes, therefore ``n1`` and ``n2`` are
        required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
        the batched matrices are not padded.

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
            >>> _ = torch.seed(1)

            # Generate a batch of isomorphic graphs
            >>> batch_size = 10
            >>> X_gt = paddle.zeros((batch_size, 4, 4))
            >>> X_gt[:, paddle.arange(0, 4, dtype=paddle.int64), paddle.randperm(4)] = 1
            >>> A1 = paddle.rand((batch_size, 4, 4))
            >>> A2 = paddle.bmm(paddle.bmm(X_gt.transpose(1, 2), A1), X_gt)
            >>> n1 = paddle.tensor([4] * batch_size)
            >>> n2 = paddle.tensor([4] * batch_size)

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
    __check_gm_arguments(n1, n2, n1max, n2max)

    args = (K, n1, n2, n1max, n2max, x0, max_iter)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.ipfp
    except ModuleNotFoundError and AttributeError:
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )
    return fn(*args)


def __check_gm_arguments(n1, n2, n1max, n2max):
    assert n1 is not None or n1max is not None, 'at least one of the following arguments are required: n1 and n1max.'
    assert n2 is not None or n2max is not None, 'at least one of the following arguments are required: n2 and n2max.'
