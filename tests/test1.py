import mindspore
import numpy as np
import torch
from mindspore.common.initializer import One, Normal
#
a = mindspore.Tensor(np.ones(shape=[10,10,10]), mindspore.float32)
slice1=slice(0,5)
b=a[1,slice1,slice1]
print(b)
# b = mindspore.Tensor(np.ones(shape=[2, 3]), mindspore.float32)
# c=mindspore.numpy.where(a>0,a,b)
# print(type(c))
# print(mindspore.ops.expand_dims(a,axis=1).shape)
# print(mindspore.ops.expand_dims(a,axis=2).shape)

# print(b.shape)
# c = mindspore.ops.MatMul(a,b)
# print(c.shape)
# a=torch.zeros(3, 3, 1, dtype=torch.bool)
# print(a.shape)
#
# b=mindspore.numpy.zeros((3,3,1),dtype=mindspore.bool_)
# print(b.shape)