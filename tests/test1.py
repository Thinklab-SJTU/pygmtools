import numpy as np
import torch
import mindspore.nn as nn
import mindspore as ms

x = ms.Tensor(np.ones([1, 2, 2, 3]).astype(np.float32))
pad_op = nn.Pad(paddings=((0, 0), (0, 0), (2, 2), (1, 2)))
output = pad_op(x)
print(output)
# Out:
# (1, 2, 6, 5)

# In Pytorch.
x = torch.ones(1, 2, 2, 3)
pad = (1, 2, 2, 2)
output = torch.nn.functional.pad(x, pad)
print(output)
# Out:
# torch.Size([1, 2, 6, 5])