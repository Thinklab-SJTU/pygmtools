API and Modules
======================================

.. currentmodule:: pygmtools

.. autosummary::
   :toctree: _autosummary
   :template: module-template.rst
   :recursive:

   linear_solvers
   classic_solvers
   multi_graph_solvers
   neural_solvers
   utils
   benchmark
   dataset

.. warning::

    By default the API functions and modules run on ``numpy`` backend. You could set the default backend by calling
    ``pygm.set_backend('new_backend')``. If you are using other backends, the corresponding package (such as
    PyTorch) should be installed. See :doc:`the numerical backend guide <../guide/numerical_backends>` for details.
