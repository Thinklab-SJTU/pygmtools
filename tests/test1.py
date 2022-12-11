import sys
import threading
sys.path.insert(0, '.')
sys.setrecursionlimit(5000)
import numpy as np
import functools
import itertools
from tqdm import tqdm

from test_utils import *
import importlib
import pygmtools
from pygmtools.utils import NOT_IMPLEMENTED_MSG, _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type
import mindspore
import pygmtools as pygm
import mindspore.nn as nn
from mindspore.ops.composite import GradOperation
from pygmtools.mindspore_backend import sm

# def main():

pygm.BACKEND = 'mindspore'
_ = mindspore.set_seed(1)
mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

# Generate a batch of isomorphic graphs
batch_size = 10
X_gt = mindspore.numpy.zeros((batch_size, 4, 4))
X_gt[:, mindspore.numpy.arange(0, 4, dtype=mindspore.int64),
[2, 3, 1, 0]] = 1
A1 = mindspore.numpy.rand((batch_size, 4, 4))
A2 = mindspore.ops.BatchMatMul()(mindspore.ops.BatchMatMul()(X_gt.swapaxes(1, 2), A1), X_gt)
n1 = n2 = mindspore.Tensor([4] * batch_size)

# Build affinity matrix
conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
import functools

gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.)  # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

# Solve by SM. Note that X is normalized with a sum of 1
X = pygm.rrwm(K, n1, n2, beta=100)
print(X.sum(axis=(1, 2)))

# Accuracy
print((pygm.hungarian(X) * X_gt).sum() / X_gt.sum())


def fn(K, n1, n2, b):
    res = pygm.rrwm(K, n1, n2, beta=b).sum()
    return res


g = mindspore.ops.grad(fn)(K, n1, n2, 100)
print(mindspore.ops.count_nonzero(g))


# if __name__ == '__main__':
#     sys.setrecursionlimit(100000)
#     threading.stack_size(200000000)
#     thread = threading.Thread(target=main())
#     thread.start()