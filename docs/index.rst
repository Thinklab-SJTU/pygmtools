Welcome to pygmtools documentation!
======================================

.. image:: https://badgen.net/pypi/v/pygmtools/
  :target: https://pypi.org/project/pygmtools/
  :alt: PyPi version
.. image:: https://img.shields.io/pypi/pyversions/pygmtools.svg
  :target: https://pypi.org/project/pygmtools/
  :alt: PyPI pyversions
.. image:: https://readthedocs.org/projects/pygmtools/badge/?version=latest
  :target: https://pygmtools.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://img.shields.io/github/stars/Thinklab-SJTU/pygmtools.svg?style=social&label=Star&maxAge=2592000
  :target: https://GitHub.com/Thinklab-SJTU/pygmtools/
  :alt: GitHub stars

**pygmtools** provides graph matching solvers in Python and is easily accessible via the following command:

::

    pip install pygmtools

Backends
-----------

By default the solvers are executed on the ``numpy`` backend, and the required packages will be automatically
downloaded.

For advanced and professional users, the ``pytorch`` backend is also available if you have installed and configured
a pytorch runtime. The ``pytorch`` backend exploits the underlying GPU-acceleration feature, and also supports
integrating graph matching modules into your deep learning pipeline.

Features
----------

To highlight, **pygmtools** has the following features:

* *Support various backends*, including ``numpy`` which is universally accessible, and the state-of-the-art
  deep learning architecture ``pytorch`` with GPU-support. The support of the following backends are also planned:
  ``tensorflow``, ``mindspore``, ``paddle``, ``jittor``;
* *Support various solvers*, including traditional combinatorial solvers and novel deep learning-based solvers;
* *Deep learning friendly*, the operations are designed to best preserve the gradient during computation and batched
  operations support for the best performance.

Benchmarks
--------------

**pygmtools** is also featured with standard data interface of several graph matching benchmarks. We also maintain a
repository containing non-trivial implementation of deep graph matching models, please check out
`ThinkMatch <https://thinkmatch.readthedocs.io/>`_ if you are interested!

Developers and Maintainers
------------------------------

**pygmtools** is currently developed and maintained by members from `ThinkLab <http://thinklab.sjtu.edu.cn>`_ at
Shanghai Jiao Tong University.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/introduction
   guide/get_started
   guide/benchmark

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/_autosummary/pygmtools.benchmark
   api/_autosummary/pygmtools.dataset
   api/_autosummary/pygmtools.classic_solvers
   api/_autosummary/pygmtools.utils
