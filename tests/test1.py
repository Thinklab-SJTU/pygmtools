import mindspore
import sys

sys.path.insert(0, '.')

import numpy as np
import torch
import functools
import itertools
from tqdm import tqdm

from test_utils import *
a = mindspore.Tensor([0])
print('a=', a)
