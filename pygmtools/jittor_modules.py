import jittor as jt
from jittor import Module, nn, Var
from jittor.nn import Sequential
from typing import Tuple, Optional, List, Union
import math


############################################
#            Affinity Modules              #
############################################


class WeightedInnerProdAffinity(Module):
    """
    Weighted inner product affinity layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(WeightedInnerProdAffinity, self).__init__()
        self.d = d
        self.A = jt.rand(self.d, self.d)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.uniform_(-stdv, stdv)
        self.A.data += jt.init.eye(self.d)

    def execute(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = jt.matmul(X, self.A)
        M = jt.matmul(M, Y.transpose(1, 2))
        return M


############################################
#         Graph Convolution Modules        #
############################################


class Gconv(Module):
    r"""
    Graph Convolutional Layer which is inspired and developed based on Graph Convolutional Network (GCN).
    Inspired by `Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
    <https://arxiv.org/abs/1609.02907>`_

    :param in_features: the dimension of input node features
    :param out_features: the dimension of output node features
    """
    def __init__(self, in_features: int, out_features: int):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def execute(self, A: Var, x: Var, norm: bool=True) -> Var:
        r"""
        Forward computation of graph convolution network.

        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param x: :math:`(b\times n\times d)` input node embedding. :math:`d`: feature dimension
        :param norm: normalize connectivity matrix or not
        :return: :math:`(b\times n\times d^\prime)` new node embedding
        """
        if norm is True:
            A = _l1_normalize(A, dim=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = jt.bmm(A, nn.relu(ax)) + nn.relu(ux) # has size (bs, N, num_outputs)
        return x


class ChannelIndependentConv(Module):
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
        super(ChannelIndependentConv, self).__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        # self.node_fc = nn.Linear(in_features, out_features // self.out_edges)
        self.node_fc = nn.Linear(in_features, out_features)
        self.node_sfc = nn.Linear(in_features, out_features)
        self.edge_fc = nn.Linear(in_edges, self.out_edges)

    def execute(self, A: Var, emb_node: Var, emb_edge: Var, mode: int=1) -> Tuple[Var, Var]:
        r"""
        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param emb_node: :math:`(b\times n\times d_n)` input node embedding. :math:`d_n`: node feature dimension
        :param emb_edge: :math:`(b\times n\times n\times d_e)` input edge embedding. :math:`d_e`: edge feature dimension
        :param mode: 1 or 2, refer to the paper for details
        :return: :math:`(b\times n\times d^\prime)` new node embedding,
         :math:`(b\times n\times n\times d^\prime)` new edge embedding
        """
        if mode == 1:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            A = A.unsqueeze(-1)
            A = jt.multiply(A.expand_as(edge_x), edge_x)

            node_x = jt.matmul(A.transpose(2, 3).transpose(1, 2),
                                  node_x.unsqueeze(2).transpose(2, 3).transpose(1, 2))
            node_x = node_x.squeeze(-1).transpose(1, 2)
            node_x = nn.relu(node_x) + nn.relu(node_sx)
            edge_x = nn.relu(edge_x)

            return node_x, edge_x

        elif mode == 2:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            d_x = node_x.unsqueeze(1) - node_x.unsqueeze(2)
            d_x = jt.sum(d_x ** 2, dim=3, keepdim=False)
            d_x = jt.exp(-d_x)

            A = A.unsqueeze(-1)
            A = jt.multiply(A.expand_as(edge_x), edge_x)

            node_x = jt.matmul(A.transpose(2, 3).transpose(1, 2),
                                  node_x.unsqueeze(2).transpose(2, 3).transpose(1, 2))
            node_x = node_x.squeeze(-1).transpose(1, 2)
            node_x = nn.relu(node_x) + nn.relu(node_sx)
            edge_x = nn.relu(edge_x)
            return node_x, edge_x

        else:
            raise ValueError('Unknown mode {}. Possible options: 1 or 2'.format(mode))


class Siamese_Gconv(Module):
    r"""
    Siamese Gconv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def execute(self, g1: Tuple[Var, Var, Var, int], *args) -> Union[Var, List[Var]]:
        r"""
        Forward computation of Siamese Gconv.

        :param g1: The first graph, which is a tuple of (:math:`(b\times n\times n)` {0,1} adjacency matrix,
         :math:`(b\times n\times d)` input node embedding, normalize connectivity matrix or not)
        :param args: Other graphs
        :return: A list of tensors composed of new node embeddings :math:`(b\times n\times d^\prime)`
        """
        # embx are tensors of size (bs, N, num_features)
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        else:
            returns = [emb1]
            for g in args:
                returns.append(self.gconv(*g))
            return returns


class Siamese_ChannelIndependentConv(Module):
    r"""
    Siamese Channel Independent Conv neural network for processing arbitrary number of graphs.

    :param in_features: the dimension of input node features
    :param num_features: the dimension of output node features
    :param in_edges: the dimension of input edge features
    :param out_edges: (optional) the dimension of output edge features. It needs to be the same as ``num_features``
    """
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super(Siamese_ChannelIndependentConv, self).__init__()
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def execute(self, g1: Tuple[Var, Var, Optional[bool]], *args) -> List[Var]:
        r"""
        Forward computation of Siamese Channel Independent Conv.

        :param g1: The first graph, which is a tuple of (:math:`(b\times n\times n)` {0,1} adjacency matrix,
         :math:`(b\times n\times d_n)` input node embedding, :math:`(b\times n\times n\times d_e)` input edge embedding,
         mode (``1`` or ``2``))
        :param args: Other graphs
        :return: A list of tensors composed of new node embeddings :math:`(b\times n\times d^\prime)`, appended with new
         edge embeddings :math:`(b\times n\times n\times d^\prime)`
        """
        emb1, emb_edge1 = self.gconv(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv(*g)
            embs.append(emb2), emb_edges.append(emb_edge2)
        return embs + emb_edges


class NGMConvLayer(Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=0, edge_emb=False):
        super(NGMConvLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.classifier = None

        if edge_emb:
            self.e_func = nn.Sequential(
                nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
                nn.ReLU(),
                nn.Linear(self.out_efeat, self.out_efeat),
                nn.ReLU()
            )
        else:
            self.e_func = None

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            #nn.Linear(self.in_nfeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            #nn.Linear(self.out_nfeat // self.out_efeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
        )

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )

    def execute(self, A, W, x, n1=None, n2=None, norm=True, sk_func=None):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        if self.e_func is not None:
            W1 = jt.multiply(A.unsqueeze(-1), x.unsqueeze(1))
            W2 = jt.concat((W, W1), dim=-1)
            W_new = self.e_func(W2)
        else:
            W_new = W

        if norm is True:
            A = jt.normalize(A, p=1, dim=2)
            A[jt.isnan(A)] = 0

        x1 = self.n_func(x)
        x2 = jt.matmul((A.unsqueeze(-1) * W_new).permute(0, 3, 1, 2), x1.unsqueeze(2).permute(0, 3, 1, 2)).squeeze(-1).transpose(1, 2)
        x2 += self.n_self_func(x)

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            assert sk_func is not None
            x3 = self.classifier(x2)
            n1_rep = jt.repeat_interleave(n1, self.sk_channel, dim=0)
            n2_rep = jt.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.permute(0,2,1).reshape((x.shape[0] * self.sk_channel, n2.max().item(), n1.max().item())).transpose(1, 2)
            x5 = sk_func(x4, n1_rep, n2_rep, dummy_row=True).transpose(2, 1) #.contiguous()

            x6 = x5.reshape((x.shape[0], self.sk_channel, n1.max().item() * n2.max().item())).permute(0, 2, 1)
            x_new = jt.concat((x2, x6), dim=-1)
        else:
            x_new = x2

        return W_new, x_new

def _l1_normalize(input: Var, dim=-1, eps=1e-12):
    return input / input.abs().sum(dim, keepdims=True).maximum(eps)
