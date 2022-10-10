# coding: utf-8
"""
========================================
Image Matching by Graph Matching Solvers
========================================
This example shows how to match images by graph matching solvers.
"""

# Author: Runzhong Wang
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 1

import pygmtools as pygm
pygm.BACKEND = 'pytorch'

##############################################################################
# POT Python Optimal Transport Toolbox
# ------------------------------------
#
# POT installation
# ```````````````````
#
# * Install with pip::
#
#     pip install pot
# * Install with conda::
#
#     conda install -c conda-forge pot
#
# Import the toolbox
# ```````````````````
#
