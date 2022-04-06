from .benchmark import Benchmark
from .classic_solvers import sinkhorn, hungarian, rrwm, sm, ipfp
from .multi_graph_solvers import cao, mgm_floyd, gamgm
import pygmtools.utils as utils
BACKEND = 'numpy'
