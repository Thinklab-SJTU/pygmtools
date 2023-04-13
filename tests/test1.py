# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

sys.path.insert(0, '.')

import numpy as np
import torch
import functools
import itertools
from tqdm import tqdm

from test_utils import *

pygm.BACKEND = 'mindspore'
a = np.array([0])
b = data_from_numpy(a)
print(type(a))
print(type(b))
