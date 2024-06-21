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
import importlib.util
set_backend = utils.set_backend

BACKEND = 'numpy'
__version__ = '0.5.3'
__author__ = 'ThinkLab at SJTU'

SUPPORTED_BACKENDS = [
    'numpy',
    'pytorch',
    'jittor',
    'paddle',
    'mindspore',
    'tensorflow'
]


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

    found_torch = importlib.util.find_spec("torch")
    if found_torch is not None:
        import torch
        print("Torch", torch.__version__)
    else:
        print("Torch not installed")

    found_paddle = importlib.util.find_spec("paddle")
    if found_paddle is not None:
        import paddle
        print("Paddle", paddle.__version__)
    else:
        print("Paddle not installed")

    found_jittor = importlib.util.find_spec("jittor")
    if found_jittor is not None:
        import jittor
        print("Jittor", jittor.__version__)
    else:
        print("Jittor not installed")

    found_pynvml = importlib.util.find_spec("pynvml")
    if found_pynvml is not None:
        import pynvml
        pynvml.nvmlInit()
        print("NVIDIA Driver Version:", pynvml.nvmlSystemGetDriverVersion())
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            print("GPU", i, ":", pynvml.nvmlDeviceGetName(handle))
    else:
        print('No GPU found. If you are using GPU, make sure to install pynvml: pip install pynvml')
