Welcome to pygmtools documentation!
======================================

**pygmtools** provides graph matching solvers in Python and is easily accessible via the following command:

    ::

        pip install pygmtools

By default the solvers are executed on the ``numpy`` backend, and the required packages will be automatically
downloaded.

For advanced and professional users, the ``pytorch`` backend is also available if you have installed and configured
a pytorch runtime. The ``pytorch`` backend exploits the underlying GPU-acceleration feature, and also supports
integrating graph matching modules into your deep learning pipeline.

To highlight, **pygmtools** has the following features:

* *Support various backends*, including ``numpy`` which is universally accessible, and the state-of-the-art
  deep learning architecture ``pytorch`` with GPU-support. The support of the following backends are also planned:
  ``tensorflow``, ``mindspore``, ``paddle``, ``jittor``;
* *Include various solvers*, including traditional combinatorial solvers and novel deep learning-based solvers;
* *Deep learning friendly*, the operations are designed to best preserve the gradient during computation and with
  special treatments on the performance.

**pygmtools** is also featured with standard data interface of several graph matching benchmarks. We also maintain a
repository containing non-trivial implementation of deep graph matching models, please check out
`ThinkMatch <https://thinkmatch.readthedocs.io/>`_ if you are interested!

**pygmtools** is currently developed and maintained by members from `ThinkLab <http://thinklab.sjtu.edu.cn>`_ at
Shanghai Jiao Tong University.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/introduction
   guide/installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/_autosummary/pygmtools.benchmark
   api/_autosummary/pygmtools.dataset
   api/_autosummary/pygmtools.classic_solvers
   api/_autosummary/pygmtools.utils
