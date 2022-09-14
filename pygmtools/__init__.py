from .benchmark import Benchmark
from .classic_solvers import sinkhorn, hungarian, rrwm, sm, ipfp
from .multi_graph_solvers import cao, mgm_floyd, gamgm
from .neural_solvers import pca_gm, ipca_gm, cie, ngm
import pygmtools.utils as utils
BACKEND = 'numpy'
