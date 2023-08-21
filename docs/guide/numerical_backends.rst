======================
Numerical Backends
======================

Now the pygmtools is available with the default ``numpy`` backend:

.. image:: ../images/numpy_logo.png
    :width: 200

We show an example of Matching Isomorphic Graphs with ``numpy`` backend in :doc:`./get_started`.

Currently, we also support deep learning architectures ``pytorch``, ``paddle``, ``jittor`` which are GPU-friendly and deep learning-friendly.

Once the backend is ready, you may switch to the backend globally by the following command:

::

    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'pytorch'  # replace 'pytorch' by other backend names

PyTorch Backend
------------------------

Introduction
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

Example: Matching Isomorphic Graphs with ``pytorch`` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we provide a basic example of matching two isomorphic graphs (i.e. two graphs have the same nodes and edges, but
the node permutations are unknown) with ``pytorch`` backend.

Step 0: Import packages and set backend

::

    >>> import torch # pytorch backend
    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'pytorch'
    >>> torch.manual_seed(1) # fix random seed

Step 1: Generate two isomorphic graphs

::

    >>> num_nodes = 10
    >>> X_gt = torch.zeros(num_nodes, num_nodes)
    >>> X_gt[torch.arange(0, num_nodes, dtype=torch.int64), torch.randperm(num_nodes)] = 1
    >>> A1 = torch.rand(num_nodes, num_nodes)
    >>> A1 = (A1 + A1.t() > 1.) * (A1 + A1.t()) / 2
    >>> torch.diagonal(A1)[:] = 0
    >>> A2 = torch.mm(torch.mm(X_gt.t(), A1), X_gt)
    >>> n1 = torch.tensor([num_nodes])
    >>> n2 = torch.tensor([num_nodes])

Step 2: Build an affinity matrix and select an affinity function

::

    >>> conn1, edge1 = pygm.utils.dense_to_sparse(A1)
    >>> conn2, edge2 = pygm.utils.dense_to_sparse(A2)
    >>> import functools
    >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
    >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

Step 3: Solve graph matching by RRWM

::

    >>> X = pygm.rrwm(K, n1, n2, beta=100)
    >>> X = pygm.hungarian(X)
    >>> X # X is the permutation matrix
    [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
      [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
      [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]

Final Step: Evaluate the accuracy

::

    >>> (X * X_gt).sum() / X_gt.sum()
    1.0

Jittor Backend
------------------------

Introduction
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


Example: Matching Isomorphic Graphs with ``jittor`` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we provide a basic example of matching two isomorphic graphs (i.e. two graphs have the same nodes and edges, but
the node permutations are unknown) with ``jittor`` backend.

Step 0: Import packages and set backend

::

    >>> import jittor as jt # jittor backend
    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'jittor'
    >>> jt.set_seed(1) # fix random seed
    >>> jt.flags.use_cuda = jt.has_cuda # detect cuda

Step 1: Generate two isomorphic graphs

::

    >>> num_nodes = 10
    >>> X_gt = jt.zeros((num_nodes, num_nodes))
    >>> X_gt[jt.arange(0, num_nodes, dtype=jt.int64), jt.randperm(num_nodes)] = 1
    >>> A1 = jt.rand(num_nodes, num_nodes)
    >>> A1 = (A1 + A1.t() > 1.) * (A1 + A1.t()) / 2
    >>> A1[jt.arange(A1.shape[0]), jt.arange(A1.shape[0])] = 0
    >>> A2 = jt.matmul(jt.matmul(X_gt.t(), A1), X_gt)
    >>> n1 = jt.Var([num_nodes])
    >>> n2 = jt.Var([num_nodes])

Step 2: Build an affinity matrix and select an affinity function

::

    >>> conn1, edge1 = pygm.utils.dense_to_sparse(A1)
    >>> conn2, edge2 = pygm.utils.dense_to_sparse(A2)
    >>> import functools
    >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
    >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

Step 3: Solve graph matching by RRWM

::

    >>> X = pygm.rrwm(K, n1, n2, beta=100)
    >>> X = pygm.hungarian(X)
    >>> X # X is the permutation matrix
    [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
      [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
      [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
      [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]

Final Step: Evaluate the accuracy

::

    >>> (X * X_gt).sum() / X_gt.sum()
    1.0

Paddle Backend
------------------------

Introduction
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

Example: Matching Isomorphic Graphs with ``paddle`` backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we provide a basic example of matching two isomorphic graphs (i.e. two graphs have the same nodes and edges, but
the node permutations are unknown) with ``paddle`` backend.

Step 0: Import packages and set backend

::

    >>> import paddle # paddle backend
    >>> import pygmtools as pygm
    >>> pygm.BACKEND = 'paddle'
    >>> paddle.seed(1) # fix random seed
    >>> paddle.device.set_device('cpu') # set cpu

Step 1: Generate two isomorphic graphs

::

    >>> num_nodes = 10
    >>> X_gt = paddle.zeros((num_nodes, num_nodes))
    >>> X_gt[paddle.arange(0, num_nodes, dtype=paddle.int64), paddle.randperm(num_nodes)] = 1
    >>> A1 = paddle.rand((num_nodes, num_nodes))
    >>> A1 = (A1 + A1.t() > 1.) / 2 * (A1 + A1.t())
    >>> A1[paddle.arange(A1.shape[0]), paddle.arange(A1.shape[1])] = 0  # paddle.diagonal(A1)[:] = 0
    >>> A2 = paddle.mm(paddle.mm(X_gt.t(), A1), X_gt)
    >>> n1 = paddle.to_tensor([num_nodes])
    >>> n2 = paddle.to_tensor([num_nodes])

Step 2: Build an affinity matrix and select an affinity function

::

    >>> conn1, edge1 = pygm.utils.dense_to_sparse(A1)
    >>> conn2, edge2 = pygm.utils.dense_to_sparse(A2)
    >>> import functools
    >>> gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) # set affinity function
    >>> K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

Step 3: Solve graph matching by RRWM

::

    >>> X = pygm.rrwm(K, n1, n2, beta=100)
    >>> X = pygm.hungarian(X)
    >>> X # X is the permutation matrix
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
      [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
      [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
      [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

Final Step: Evaluate the accuracy

::

    >>> (X * X_gt).sum() / X_gt.sum()
    1.0

Tensorflow Backend
------------------------

Introduction
^^^^^^^^^^^^^^^^

.. image:: ../images/tensorflow_logo.png
    :width: 300

TensorFlow is an end-to-end open source platform for machine learning, which is developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research organization.
The TensorFlow backend of ``pygmtools`` is designed to support GPU devices and deep learning applications.

Please follow `the official Tensorflow installation guide <https://www.tensorflow.org/install>`_.

This package is developed with ``Tensorflow==2.9.3`` and please mind the API compatibility among different Tensorflow
versions.

How to enable Tensorflow backend:

::

    >>> import pygmtools as pygm
    >>> import tensorflow
    >>> pygm.BACKEND = 'tensorflow'

Examples with Tensorflow backend are coming soon.

Mindspore Backend
------------------------

Introduction
^^^^^^^^^^^^^^^^

.. image:: ../images/mindspore_logo.png
    :width: 300

Mindspore is an open source deep learning platform developed and maintained by Huawei.
The Mindspore backend of ``pygmtools`` is designed to support GPU devices and deep learning applications.

Please follow `the official Mindspore installation guide <https://www.mindspore.cn/install>`_.

This package is developed with ``mindspore==1.10.0`` and shall work with any Mindspore versions ``>=1.10.0``.

How to enable Mindspore backend:

::

    >>> import pygmtools as pygm
    >>> import mindspore
    >>> pygm.BACKEND = 'mindspore'

Examples with Mindspore backend are coming soon.

What's Next
------------
Please checkout :doc:`../auto_examples/index` to see how to apply ``pygmtools`` to tackle real-world problems.
You may see :doc:`../api/pygmtools` for the API documentation.
