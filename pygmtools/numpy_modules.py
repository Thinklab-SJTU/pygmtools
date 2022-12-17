# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
import math

############################################
#            Affinity Modules              #
############################################

class WeightedInnerProdAffinity():
    def __init__(self, d):
        self.d = d

        stdv = 1. / math.sqrt(self.d)
        self.A = np.random.uniform(-stdv,stdv,[self.d,self.d])
        self.A += np.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = np.matmul(X, self.A)
        M = np.matmul(M, Y.swapaxes(1, 2))
        return M

############################################
#         Graph Convolution Modules        #
############################################
def relu(X):
    X[X<0] = 0
    return X

def kaiming_uniform_(array: np.ndarray, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    """Numpy's kaiming_uniform_"""
    gain = math.sqrt(2/(a*a+1))
    fan_in = array.shape[1]
    fan_out = array.shape[0]
    if mode == 'fan_in':
        fan_mode = fan_in
    if mode == 'fan_out':
        fan_mode = fan_out
    bound = gain * math.sqrt(3/fan_mode)
    array = uniform_(array, -bound, bound)
    return array

def uniform_(array,a,b):
    array = np.random.uniform(a,b,array.shape)
    return array

def normalize_abs(array,axis):
    array_shape = array.shape
    k = abs(array).sum(axis)
    k = k.repeat(array_shape[axis],(axis-1+len(array_shape)) % len(array_shape)).reshape(array_shape)
    array = np.nan_to_num(array/k)
    return array

def expand_as(array,target_arary):
    ori_array_shape = array.shape
    array_axis = len(ori_array_shape)
    ori_target_arary_shape = target_arary.shape
    target_arary_axis = len(ori_target_arary_shape)
    if(array_axis != target_arary_axis):
        if(target_arary_axis > array_axis):
            for _ in np.arange(target_arary_axis-array_axis):
                array = np.expand_dims(array,axis=0)
        else:
            message = "The size of the input array exceeds the target array!"
            message += "\ninput array's shape:" + str(ori_array_shape)
            message += "\ntarget array's shape:" + str(ori_target_arary_shape)
            raise ValueError(message)
    array_shape = array.shape
    target_arary_shape = target_arary.shape
    l = target_arary_axis
    for i in np.arange(target_arary_axis):
        k = l-i-1
        m = array_shape[k]
        n = target_arary_shape[k]
        if(m == 1):
            array = array.repeat(n/m,axis=k)
        elif(m != n):
            message = "\nThe expanded size of the array (" + str(n) + ") must match the existing size (" + str(m)  
            message += ") at non-singleton dimension " + str(k)
            message += "\ninput array's shape:" + str(ori_array_shape)
            message += "\ntarget array's shape:" + str(ori_target_arary_shape)
            raise ValueError(message)
    return array

class Linear():
    """Numpy's Linear"""
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: np.ndarray

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.empty((out_features, in_features), dtype='f')
        if bias:
            self.bias = np.empty(out_features, dtype='f')
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = uniform_(self.bias, -bound, bound)

    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.matmul(input,self.weight.swapaxes(-1,-2)) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Sequential():

    def __init__(self, *args):
        self._modules = {}
        for idx, module in enumerate(args):
            self._modules[idx] = module

    def getitem(self, idx):
        return self._modules[idx]

    def setitem(self, idx, module):
        if (idx >= len(self._modules)):
            raise ValueError("Maximum value exceeded!")
        self._modules[idx] = module

    def delitem(self, idx):
        for i in range(idx, len(self._modules) - 1):
            self._modules[i] = self._modules[i + 1]
        del self._modules[len(self._modules) - 1]

    def len(self):
        return len(self._modules)

    def append(self, module):
        new_idx = int(list(self._modules.keys())[-1]) + 1
        self._modules[new_idx] = module

    def forward(self, inputs):
        for module in self._modules.values():
            inputs = module.forward(inputs)
        return inputs

class ReLU():

    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def forward(self, input: np.ndarray) -> np.ndarray:
        return relu(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Gconv():
    def __init__(self, in_features: int, out_features: int):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = Linear(self.num_inputs, self.num_outputs)
        self.u_fc = Linear(self.num_inputs, self.num_outputs)

    def forward(self, A: np.ndarray, x: np.ndarray, norm: bool=True) -> np.ndarray:
        r"""
        Forward computation of graph convolution network.

        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param x: :math:`(b\times n\times d)` input node embedding. :math:`d`: feature dimension
        :param norm: normalize connectivity matrix or not
        :return: :math:`(b\times n\times d^\prime)` new node embedding
        """
        if norm is True:
            A = normalize_abs(A,axis=-2)
        ax = self.a_fc.forward(x)
        ux = self.u_fc.forward(x)
        x = np.matmul(A,relu(ax)) + relu(ux) # has size (bs, N, num_outputs)
        return x

class ChannelIndependentConv():
    r"""
    Channel Independent Embedding Convolution.
    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    :param in_features: the dimension of input node features
    :param out_features: the dimension of output node features
    :param in_edges: the dimension of input edge features
    :param out_edges: (optional) the dimension of output edge features. It needs to be the same as ``out_features``
    """
    def __init__(self, in_features: int, out_features: int, in_edges: int, out_edges: int=None):
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        self.node_fc = Linear(in_features, out_features)
        self.node_sfc = Linear(in_features, out_features)
        self.edge_fc = Linear(in_edges, self.out_edges)

    def forward(self, A: np.ndarray, emb_node: np.ndarray, emb_edge: np.ndarray, mode: int=1):
        r"""
        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param emb_node: :math:`(b\times n\times d_n)` input node embedding. :math:`d_n`: node feature dimension
        :param emb_edge: :math:`(b\times n\times n\times d_e)` input edge embedding. :math:`d_e`: edge feature dimension
        :param mode: 1 or 2, refer to the paper for details
        :return: :math:`(b\times n\times d^\prime)` new node embedding,
         :math:`(b\times n\times n\times d^\prime)` new edge embedding
        """
        if mode == 1:
            node_x = self.node_fc.forward(emb_node)
            node_sx = self.node_sfc.forward(emb_node)
            edge_x = self.edge_fc.forward(emb_edge)
            
            A = np.expand_dims(A,axis=-1)
            A =  expand_as(A,edge_x) * edge_x

            node_x = np.matmul(A.swapaxes(2, 3).swapaxes(1, 2),
                                  np.expand_dims(node_x,axis=2).swapaxes(2, 3).swapaxes(1, 2))
            node_x = np.squeeze(node_x,axis=-1).swapaxes(1, 2)
            node_x = relu(node_x) + relu(node_sx)
            edge_x = relu(edge_x)

            return node_x, edge_x

        # The following code lines are not called in pygmtools
        # elif mode == 2:
        #     node_x = self.node_fc(emb_node)
        #     node_sx = self.node_sfc(emb_node)
        #     edge_x = self.edge_fc(emb_edge)
        #
        #     d_x = np.expand_dims(node_x,axis=-1) - np.expand_dims(node_x,axis=2)
        #     d_x = np.sum(d_x ** 2, axis=3, keepdim=False)
        #     d_x = np.exp(-d_x)
        #
        #     A = np.expand_dims(A,axis=-1)
        #     A = expand_as(A,edge_x) * edge_x
        #
        #     node_x = np.matmul(A.swapaxes(2, 3).swapaxes(1, 2),
        #                           np.expand_dims(node_x,axis=2).swapaxes(2, 3).swapaxes(1, 2))
        #     node_x = np.squeeze(node_x,axis=-1).swapaxes(1, 2)
        #     node_x = relu(node_x) + relu(node_sx)
        #     edge_x = relu(edge_x)
        #     return node_x, edge_x

        else:
            raise ValueError('Unknown mode {}. Possible options: 1 or 2'.format(mode))

class Siamese_Gconv():
    r"""
    Siamese Gconv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    """
    def __init__(self, in_features, num_features):
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, *args):
        # embx are tensors of size (bs, N, num_features)
        emb1 = self.gconv.forward(*g1)
        if len(args) == 0:
            return emb1
        else:
            returns = [emb1]
            for g in args:
                returns.append(self.gconv.forward(*g))
            return returns

class Siamese_ChannelIndependentConv():
    r"""
    Siamese Channel Independent Conv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    :param in_edges: the dimension of input edge features
    :param out_edges: (optional) the dimension of output edge features. It needs to be the same as ``num_features``
    """
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def forward(self, g1, *args):
        r"""
        Forward computation of Siamese Channel Independent Conv.

        :param g1: The first graph, which is a tuple of (:math:`(b\times n\times n)` {0,1} adjacency matrix,
         :math:`(b\times n\times d_n)` input node embedding, :math:`(b\times n\times n\times d_e)` input edge embedding,
         mode (``1`` or ``2``))
        :param args: Other graphs
        :return: A list of tensors composed of new node embeddings :math:`(b\times n\times d^\prime)`, appended with new
         edge embeddings :math:`(b\times n\times n\times d^\prime)`
        """
        emb1, emb_edge1 = self.gconv.forward(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv.forward(*g)
            embs.append(emb2), emb_edges.append(emb_edge2)
        return embs + emb_edges

class NGMConvLayer():
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=0):
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.classifier = Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.classifier = None

        self.n_func = Sequential(
            Linear(self.in_nfeat, self.out_nfeat),
            #nn.Linear(self.in_nfeat, self.out_nfeat // self.out_efeat),
            ReLU(),
            Linear(self.out_nfeat, self.out_nfeat),
            #nn.Linear(self.out_nfeat // self.out_efeat, self.out_nfeat // self.out_efeat),
            ReLU(),
        )

        self.n_self_func = Sequential(
            Linear(self.in_nfeat, self.out_nfeat),
            ReLU(),
            Linear(self.out_nfeat, self.out_nfeat),
            ReLU()
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True, sk_func=None):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        W_new = W

        if norm is True:
            A = normalize_abs(A,axis=2)
        
        x1 = self.n_func.forward(x)
        tmp1 = (np.expand_dims(A,axis=-1) * W_new).transpose((0, 3, 1, 2))
        tmp2 = np.expand_dims(x1,axis=2).transpose((0, 3, 1, 2))
        x2 = np.squeeze(np.matmul(tmp1,tmp2),axis=-1).swapaxes(1, 2)
        x2 += self.n_self_func.forward(x)
        
        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            assert sk_func is not None
            x3 = self.classifier.forward(x2)
            n1_rep = n1.repeat(self.sk_channel, axis=0)
            n2_rep = n2.repeat(self.sk_channel, axis=0)
            x4 = x3.transpose((0,2,1)).reshape(x.shape[0] * self.sk_channel, n2.max(), n1.max()).swapaxes(1, 2)
            x5 = np.ascontiguousarray(sk_func(x4, n1_rep, n2_rep, dummy_row=True).swapaxes(2, 1))
            
            x6 = x5.reshape(x.shape[0], self.sk_channel, n1.max() * n2.max()).transpose((0, 2, 1))
            x_new = np.concatenate((x2, x6), axis=-1)
        else:
            x_new = x2
        
        return W_new, x_new