from .benchmark import Benchmark
from .linear_solvers import sinkhorn, hungarian
from .classic_solvers import rrwm, sm, ipfp
from .multi_graph_solvers import cao, mgm_floyd, gamgm
from .neural_solvers import pca_gm, ipca_gm, cie, ngm
import pygmtools.utils as utils
BACKEND = 'numpy'
__version__ = '0.2.10'
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
