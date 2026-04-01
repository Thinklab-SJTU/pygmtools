# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter


def _l1_normalize(input_tensor, axis=-1, eps=1e-12):
    denom = ops.ReduceSum(keep_dims=True)(ops.abs(input_tensor), axis)
    return input_tensor / ops.maximum(denom, mindspore.Tensor(eps, dtype=input_tensor.dtype))


class Identity(nn.Cell):
    def construct(self, x, *args, **kwargs):
        return x


class WeightedInnerProdAffinity(nn.Cell):
    """
    Weighted inner product affinity layer to compute the affinity matrix from feature space.
    """

    def __init__(self, d):
        super().__init__()
        self.d = d
        stdv = 1.0 / math.sqrt(self.d)
        init = ops.uniform((self.d, self.d), mindspore.Tensor(-stdv, mindspore.float32),
                           mindspore.Tensor(stdv, mindspore.float32)) + ops.eye(self.d, self.d, mindspore.float32)
        self.A = Parameter(init, name='A')

    def construct(self, X, Y):
        M = ops.matmul(X, self.A)
        M = ops.matmul(M, Y.swapaxes(1, 2))
        return M


class Gconv(nn.Cell):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Dense(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Dense(self.num_inputs, self.num_outputs)

    def construct(self, A, x, norm=True):
        if norm:
            A = _l1_normalize(A, axis=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        return ops.BatchMatMul()(A, ops.relu(ax)) + ops.relu(ux)


class ChannelIndependentConv(nn.Cell):
    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        super().__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        self.node_fc = nn.Dense(in_features, out_features)
        self.node_sfc = nn.Dense(in_features, out_features)
        self.edge_fc = nn.Dense(in_edges, self.out_edges)

    def construct(self, A, emb_node, emb_edge, mode=1):
        if mode != 1:
            raise ValueError(f'Unknown mode {mode}. Possible options: 1 or 2')

        node_x = self.node_fc(emb_node)
        node_sx = self.node_sfc(emb_node)
        edge_x = self.edge_fc(emb_edge)

        A = ops.expand_dims(A, -1)
        A = ops.mul(ops.broadcast_to(A, edge_x.shape), edge_x)

        batch_size, num_nodes = emb_node.shape[0], emb_node.shape[1]
        out_channels = self.out_edges
        lhs = ops.transpose(A, (0, 3, 1, 2)).reshape((batch_size * out_channels, num_nodes, num_nodes))
        rhs = ops.transpose(ops.expand_dims(node_x, 2), (0, 3, 1, 2)).reshape((batch_size * out_channels, num_nodes, 1))
        node_x = ops.BatchMatMul()(lhs, rhs)
        node_x = ops.reshape(node_x, (batch_size, out_channels, num_nodes))
        node_x = ops.transpose(node_x, (0, 2, 1))
        node_x = ops.relu(node_x) + ops.relu(node_sx)
        edge_x = ops.relu(edge_x)

        return node_x, edge_x


class Siamese_Gconv(nn.Cell):
    def __init__(self, in_features, num_features):
        super().__init__()
        self.gconv = Gconv(in_features, num_features)

    def construct(self, g1, *args):
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        returns = [emb1]
        for g in args:
            returns.append(self.gconv(*g))
        return returns


class Siamese_ChannelIndependentConv(nn.Cell):
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super().__init__()
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def construct(self, g1, *args):
        emb1, emb_edge1 = self.gconv(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv(*g)
            embs.append(emb2)
            emb_edges.append(emb_edge2)
        return embs + emb_edges


class NGMConvLayer(nn.Cell):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, sk_channel=0):
        super().__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.classifier = nn.Dense(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.classifier = None

        self.n_func = nn.SequentialCell([
            nn.Dense(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Dense(self.out_nfeat, self.out_nfeat),
            nn.ReLU(),
        ])
        self.n_self_func = nn.SequentialCell([
            nn.Dense(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Dense(self.out_nfeat, self.out_nfeat),
            nn.ReLU(),
        ])

    def construct(self, A, W, x, n1=None, n2=None, norm=True, sk_func=None):
        W_new = W

        if norm:
            A = _l1_normalize(A, axis=2)

        x1 = self.n_func(x)
        Aw = ops.expand_dims(A, -1) * W_new
        batch_size, num_nodes = x.shape[0], x.shape[1]
        out_channels = self.out_nfeat
        lhs = ops.transpose(Aw, (0, 3, 1, 2))
        if lhs.shape[1] != out_channels:
            lhs = ops.broadcast_to(lhs, (batch_size, out_channels, num_nodes, num_nodes))
        lhs = lhs.reshape((batch_size * out_channels, num_nodes, num_nodes))
        rhs = ops.transpose(ops.expand_dims(x1, 2), (0, 3, 1, 2)).reshape((batch_size * out_channels, num_nodes, 1))
        x2 = ops.BatchMatMul()(lhs, rhs)
        x2 = ops.reshape(x2, (batch_size, out_channels, num_nodes))
        x2 = ops.transpose(x2, (0, 2, 1))
        x2 = x2 + self.n_self_func(x)

        if self.classifier is not None:
            assert sk_func is not None
            n1max = int(n1.asnumpy().max())
            n2max = int(n2.asnumpy().max())
            x3 = self.classifier(x2)
            n1_rep = ops.repeat_interleave(n1, self.sk_channel, axis=0)
            n2_rep = ops.repeat_interleave(n2, self.sk_channel, axis=0)
            x4 = ops.transpose(x3, (0, 2, 1)).reshape((x.shape[0] * self.sk_channel, n2max, n1max))
            x4 = ops.transpose(x4, (0, 2, 1))
            x5 = ops.transpose(sk_func(x4, n1_rep, n2_rep, dummy_row=True), (0, 2, 1))
            x6 = ops.reshape(x5, (x.shape[0], self.sk_channel, n1max * n2max))
            x6 = ops.transpose(x6, (0, 2, 1))
            x_new = ops.concat((x2, x6), axis=-1)
        else:
            x_new = x2

        return W_new, x_new
