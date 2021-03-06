=============
Get Started
=============

Basic Install
______________

Install ``pygmtools`` is easy:
::

    pip install pygmtools


Now the pygmtools is available with the ``numpy`` backend. You may jump to :ref:`Example: Matching Isomorphic Graphs`
if you do not need other backends.

The following packages are required, and shall be automatically downloaded by ``pip install``:

* Python >= 3.5
* requests >= 2.25.1
* scipy >= 1.4.1
* Pillow >= 7.2.0
* numpy >= 1.18.5
* easydict >= 1.7


Install Other Backends
_________________________

Currently, we also support deep learning architectures ``pytorch`` and ``paddle`` which are GPU-friendly and deep learning-friendly.
The support of the following backends are also planned: ``tensorflow``, ``mindspore``, ``jittor``.

Please follow the install instructions on your backend.

Once the backend is ready, you may switch to the backend globally by the following command:

::

    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'pytorch'  # replace 'pytorch' by other backend names


Example: Matching Isomorphic Graphs
______________________________________

Here we provide a basic example of matching two isomorphic graphs (i.e. two graphs have the same nodes and edges, but
the node permutations are unknown).

Step 0: Import packages and set backend

::

    >>> import numpy as np
    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'numpy'
    >>> np.random.seed(1)

Step 1: Generate a batch of isomorphic graphs

::

    >>> batch_size = 3
    >>> X_gt = np.zeros((batch_size, 4, 4))
    >>> X_gt[:, np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
    >>> A1 = np.random.rand(batch_size, 4, 4)
    >>> A2 = np.matmul(np.matmul(X_gt.transpose((0, 2, 1)), A1), X_gt)
    >>> n1 = n2 = np.repeat([4], batch_size)

Step 2: Build affinity matrix and select an affinity function

::

    >>> conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
    >>> conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
    >>> import functools
    >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
    >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, ne1, n2, ne2, edge_aff_fn=gaussian_aff)

Step 3: Solve graph matching by RRWM

::

    >>> X = pygm.rrwm(K, n1, n2, beta=100)
    >>> X = pygm.hungarian(X)
    >>> X # X is the permutation matrix
    [[[0. 0. 0. 1.]
      [0. 0. 1. 0.]
      [1. 0. 0. 0.]
      [0. 1. 0. 0.]]

     [[0. 0. 0. 1.]
      [0. 0. 1. 0.]
      [1. 0. 0. 0.]
      [0. 1. 0. 0.]]

     [[0. 0. 0. 1.]
      [0. 0. 1. 0.]
      [1. 0. 0. 0.]
      [0. 1. 0. 0.]]]

Final Step: Evaluate the accuracy

::

    >>> (X * X_gt).sum() / X_gt.sum()
    1.0
