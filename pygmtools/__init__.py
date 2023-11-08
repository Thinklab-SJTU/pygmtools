# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from .benchmark import Benchmark
from .linear_solvers import sinkhorn, hungarian
from .classic_solvers import rrwm, sm, ipfp, astar
from .multi_graph_solvers import cao, mgm_floyd, gamgm
from .neural_solvers import pca_gm, ipca_gm, cie, ngm, genn_astar
import pygmtools.utils as utils
BACKEND = 'numpy'
__version__ = '0.4.2a2'
__author__ = 'ThinkLab at SJTU'


def env_report():
    """
    Print environment report
    """
    import platform
    print(platform.platform())

    import sys
    print("Python", sys.version)

    import numpy
    print("NumPy", numpy.__version__)

    import scipy
    print("SciPy", scipy.__version__)

    from pygmtools import __version__
    print("pygmtools", __version__)

    try:
        import torch
        print("Torch", torch.__version__)
    except ImportError:
        print("Torch not installed")

    try:
        import paddle
        print("Paddle", paddle.__version__)
    except ImportError:
        print("Paddle not installed")

    try:
        import jittor
        print("Jittor", jittor.__version__)
    except ImportError:
        print("Jittor not installed")

    try:
        import pynvml
        pynvml.nvmlInit()
        print("NVIDIA Driver Version:", pynvml.nvmlSystemGetDriverVersion())
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            print("GPU", i, ":", pynvml.nvmlDeviceGetName(handle))
    except ImportError:
        print('No GPU found. If you are using GPU, make sure to install pynvml: pip install pynvml')
