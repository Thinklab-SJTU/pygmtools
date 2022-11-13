import mindspore
import numpy as np
import torch
from mindspore.common.initializer import One, Normal
#
# a = mindspore.Tensor(np.ones(shape=[2,3]), mindspore.float32)
# b = mindspore.Tensor(np.ones(shape=[3, 4]), mindspore.float32)
# print(mindspore.ops.expand_dims(a,axis=1).shape)
# print(mindspore.ops.expand_dims(a,axis=2).shape)

# print(b.shape)
# c = mindspore.ops.MatMul(a,b)
# print(c.shape)

a=(0,0,0,0)
b=list((a[2 * i], a[2 * i + 1]) for i in range(int(len(a) / 2)))
b=tuple(b)
print(b)