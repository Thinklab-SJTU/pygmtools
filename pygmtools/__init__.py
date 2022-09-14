from .benchmark import Benchmark
from .linear_solvers import sinkhorn, hungarian
from .classic_solvers import rrwm, sm, ipfp
from .multi_graph_solvers import cao, mgm_floyd, gamgm
from .neural_solvers import pca_gm, ipca_gm, cie, ngm
import pygmtools.utils as utils
BACKEND = 'numpy'
