=============
Get Started
=============

Basic Install by pip
----------------------

You can install the stable release on PyPI:

::

    $ pip install pygmtools

or get the latest version by running:

::

    $ pip install -U https://github.com/Thinklab-SJTU/pygmtools/archive/master.zip # with --user for user install (no root)


The following packages are required, and shall be automatically installed by ``pip``:

::

    Python >= 3.5
    requests >= 2.25.1
    scipy >= 1.4.1
    Pillow >= 7.2.0
    numpy >= 1.18.5
    easydict >= 1.7
    appdirs >= 1.4.4
    tqdm >= 4.64.1

Note that we support different deep learning architectures ``pytorch``, ``paddle``, ``jittor``. You may see
:doc:`./numerical_backends` for the introduction and examples on different backends.

Example: Matching Isomorphic Graphs with ``numpy`` backend
---------------------------------------------------------------

Here we provide a basic example of matching two isomorphic graphs (i.e. two graphs have the same nodes and edges, but
the node permutations are unknown) with the default ``numpy`` backend to show the usage of pygmtools.

Step 0: Import packages and set backend

::

    >>> import numpy as np
    >>> import pygmtools as pygm
    >>> pygm.set_backend('numpy')
    >>> np.random.seed(1)

Step 1: Generate a batch of isomorphic graphs

::

    >>> batch_size = 3
    >>> X_gt = np.zeros((batch_size, 4, 4))
    >>> X_gt[:, np.arange(0, 4, dtype=np.int64), np.random.permutation(4)] = 1
    >>> A1 = np.random.rand(batch_size, 4, 4)
    >>> A2 = np.matmul(np.matmul(X_gt.transpose((0, 2, 1)), A1), X_gt)
    >>> n1 = n2 = np.repeat([4], batch_size)

Step 2: Build an affinity matrix and select an affinity function

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


What's Next
------------
Please checkout :doc:`../auto_examples/index` to see how to apply ``pygmtools`` to tackle real-world problems.
You may see :doc:`../api/pygmtools` for the API documentation.
