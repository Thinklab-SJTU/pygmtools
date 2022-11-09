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

    By default the API functions and modules run on ``numpy`` backend. You could set the default backend by setting
    ``pygm.BACKEND``. If you enable other backends than ``numpy``, the corresponding package should be installed. See
    :doc:`the installation guide <../guide/get_started>` for details.
