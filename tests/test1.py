import mindspore
import numpy as np
import torch
from mindspore.common.initializer import One, Normal

a = mindspore.Tensor(np.ones(shape=[2, 3]), mindspore.float32)
b = mindspore.Tensor(np.ones(shape=[3, 4]), mindspore.float32)
print(a.shape)

print(b.shape)
c = mindspore.ops.MatMul(a,b)
print(c.shape)
