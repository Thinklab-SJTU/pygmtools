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




Now the pygmtools is available with the ``numpy`` backend:

.. image:: ../images/numpy_logo.png
    :width: 200

You may jump to :ref:`Example: Matching Isomorphic Graphs` if you do not need other backends.

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


Install Other Backends
------------------------

Currently, we also support deep learning architectures ``pytorch``, ``paddle``, ``jittor`` which are GPU-friendly and deep learning-friendly.

Once the backend is ready, you may switch to the backend globally by the following command:

::

    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'pytorch'  # replace 'pytorch' by other backend names

PyTorch Backend
^^^^^^^^^^^^^^^^

.. image:: ../images/pytorch_logo.png
    :width: 250

PyTorch is an open-source machine learning framework developed and maintained by Meta Inc./Linux Foundation.
PyTorch is popular, especially among the deep learning research community.
The PyTorch backend of ``pygmtools`` is designed to support GPU devices and facilitate deep learning research.

Please follow `the official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_.

This package is developed with ``torch==1.6.0`` and shall work with any PyTorch versions ``>=1.6.0``.

How to enable PyTorch backend:

::

    >>> import pygmtools as pygm
    >>> import torch
    >>> pygm.BACKEND = 'pytorch'

Paddle Backend
^^^^^^^^^^^^^^^^

.. image:: ../images/paddle_logo.png
    :width: 300

PaddlePaddle is an open-source deep learning platform originated from industrial practice, which is developed and
maintained by Baidu Inc.
The Paddle backend of ``pygmtools`` is designed to support GPU devices and deep learning applications.

Please follow `the official PaddlePaddle installation guide <https://www.paddlepaddle.org.cn/en/install/quick>`_.

This package is developed with ``paddlepaddle==2.3.1`` and shall work with any PaddlePaddle versions ``>=2.3.1``.

How to enable Paddle backend:

::

    >>> import pygmtools as pygm
    >>> import paddle
    >>> pygm.BACKEND = 'paddle'

Jittor Backend
^^^^^^^^^^^^^^^^

.. image:: ../images/jittor_logo.png
    :width: 300

Jittor is an open-source deep learning platform based on just-in-time (JIT) for high performance, which is developed
and maintained by the `CSCG group <https://cg.cs.tsinghua.edu.cn/>`_ from Tsinghua University.
The Jittor backend of ``pygmtools`` is designed to support GPU devices and deep learning applications.

Please follow `the official Jittor installation guide <https://github.com/Jittor/Jittor#install>`_.

This package is developed with ``jittor==1.3.4.16`` and shall work with any Jittor versions ``>=1.3.4.16``.

How to enable Jittor backend:

::

    >>> import pygmtools as pygm
    >>> import jittor
    >>> pygm.BACKEND = 'jittor'

Tensorflow Backend
^^^^^^^^^^^^^^^^^^^

.. image:: ../images/tensorflow_logo.png
    :width: 300

TensorFlow is an end-to-end open source platform for machine learning, which is developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research organization
The TensorFlow backend of ``pygmtools`` is designed to support GPU devices and deep learning applications.

Please follow `the official Tensorflow installation guide <https://tensorflow.google.cn/install>`_.

This package is developed with ``Tensorflow==2.9.1`` and shall work with any Tensorflow versions ``>=2.9.1``.

How to enable Tensorflow backend:

::

    >>> import pygmtools as pygm
    >>> import tensorflow
    >>> pygm.BACKEND = 'tensorflow'

Example: Matching Isomorphic Graphs
------------------------------------

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
