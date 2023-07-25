import torch
import pygmtools.utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional, Tuple

VERY_LARGE_INT = 65536

###############################################################
#                      GENN-A*  Functions                     #
###############################################################


def default_parameter():
    params = dict()
    params['cuda'] = False
    params['pretrain'] = False
    params['channel'] = 36
    params['filters_1'] = 64
    params['filters_2'] = 32
    params['filters_3'] = 16
    params['tensor_neurons'] = 16
    params['dropout'] = 0
    params['astar_beam_width'] = 0
    params['astar_trust_fact'] = 1
    params['astar_no_pred'] = 0
    params['use_net'] = True
    return params


def check_layer_parameter(params):
    if params['pretrain'] == 'AIDS700nef':
        if params['channel'] != 36:
            return False
    elif params['pretrain'] == 'LINUX':
        if params['channel'] != 8:
            return False
    if params['filters_1'] != 64:
        return False
    if params['filters_2'] != 32:
        return False
    if params['filters_3'] != 16:
        return False
    if params['tensor_neurons'] != 16:
        return False
    return True


def node_metric(node1, node2):
    
    encoding = torch.sum(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)), dim=-1)
    non_zero = torch.nonzero(encoding)
    for i in range(non_zero.shape[0]):
        encoding[non_zero[i][0], non_zero[i][1], non_zero[i][2]] = 1
    return encoding


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def to_dense_batch(x: Tensor, batch: Optional[Tensor] = None,
                   fill_value: float = 0., max_num_nodes: Optional[int] = None,
                   batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    num_nodes = scatter_sum(batch.new_ones(x.size(0)), batch, dim=0,
                            dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


def to_dense_adj(edge_index: Tensor,batch=None,edge_attr=None,max_num_nodes: Optional[int] = None) -> Tensor:
    if batch is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = edge_index.new_zeros(num_nodes)

    batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_sum(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
          or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj = scatter_sum(edge_attr, idx, dim=0, dim_size=flattened_size)
    adj = adj.view(size)

    return adj


###############################################################
#                       GENN-A*  Modules                      #
###############################################################


class GraphPair:
    def __init__(self, x1: torch.Tensor, x2: torch.Tensor, adj1: torch.Tensor,
                 adj2: torch.Tensor, n1=None, n2=None):
        self.g1 = Graphs(x1, adj1, n1)
        self.g2 = Graphs(x2, adj2, n2)

    def __repr__(self):
        return f"{self.__class__.__name__}('g1' = {self.g1}, 'g2' = {self.g2})"

    def to_dict(self):
        data = dict()
        data['g1'] = self.g1
        data['g2'] = self.g2
        return data


class Graphs:
    def __init__(self, x: torch.Tensor, adj: torch.Tensor, nodes_num=None):
        assert len(x.shape) == len(adj.shape)
        if len(adj.shape) == 2:
            adj = adj.unsqueeze(dim=0)
            x = x.unsqueeze(dim=0)
        assert x.shape[0] == adj.shape[0]
        assert x.shape[1] == adj.shape[1]
        self.x = x
        self.adj = adj
        self.num_graphs = adj.shape[0]
        if nodes_num is not None:
            self.nodes_num = nodes_num
        else:
            self.nodes_num = torch.tensor([x.shape[1]]*x.shape[0])
        if self.x.shape[0] == 1:
            if self.x.shape[1] != nodes_num:
                self.x = self.x[:, :nodes_num, :]
                self.adj = self.adj[:, :nodes_num, :nodes_num]
        self.edge_index = None
        self.edge_weight = None
        self.batch = None
        self.graph_process()

    def graph_process(self):
        edge_index, edge_weight, _ = pygmtools.utils.dense_to_sparse(self.adj)
        self.edge_index = torch.cat([edge_index[:, :, 0].unsqueeze(dim=1),
                                     edge_index[:, :, 1].unsqueeze(dim=1)], dim=1)
        self.edge_weight = edge_weight.view(-1)
        if self.nodes_num.shape == torch.Size([]):
            batch = torch.tensor([0] * self.nodes_num)
        else:
            for i in range(len(self.nodes_num)):
                if i == 0:
                    batch = torch.tensor([i] * self.nodes_num[i])
                else:
                    cur_batch = torch.tensor([i] * self.nodes_num[i])
                    batch = torch.cat([batch, cur_batch])
        self.batch = batch

    def __repr__(self):
        message = "x = {}, adj = {}".format(list(self.x.shape), list(self.adj.shape))
        message += " edge_index = {}, edge_weight = {}".format(list(self.edge_index.shape), list(self.edge_weight.shape))
        message += " nodes_num = {}, num_graphs = {})".format(self.nodes_num.shape, self.num_graphs)
        self.message = message
        return f"{self.__class__.__name__}({self.message})"


class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args['filters_3'], self.args['filters_3']))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param batch: Batch vector, which assigns each node to a specific example
        :return representation: A graph level representation matrix. 
        """
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter_mean(x, batch, dim=0, dim_size=size)
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        coefs = torch.sigmoid((x * transformed_global[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return scatter_sum(weighted, batch, dim=0, dim_size=size)

    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self,args):
        """
        :param args: Arguments object.
        """
        super(TensorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args['filters_3'], self.args['filters_3'], self.args['tensor_neurons']))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args['tensor_neurons'], 2*self.args['filters_3']))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args['tensor_neurons'], 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = len(embedding_1)
        scoring = torch.matmul(embedding_1, self.weight_matrix.view(self.args['filters_3'],-1))
        scoring = scoring.view(batch_size, self.args['filters_3'], -1).permute([0, 2, 1])
        scoring = torch.matmul(scoring, embedding_2.view(batch_size, self.args['filters_3'], 1)).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(torch.mm(self.weight_matrix_block, torch.t(combined_representation)))
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores


class GCNConv(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(GCNConv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.weight = Parameter(torch.empty((in_features,out_features)))
        self.bias = Parameter(torch.empty(out_features))

    def forward(self, A: Tensor, x: Tensor, norm: bool=True) -> Tensor:
        r"""
        Forward computation of graph convolution network.

        :param A: :math:`(b\times n\times n)` {0,1} adjacency matrix. :math:`b`: batch size, :math:`n`: number of nodes
        :param x: :math:`(b\times n\times d)` input node embedding. :math:`d`: feature dimension
        :param norm: normalize connectivity matrix or not
        :return: :math:`(b\times n\times d^\prime)` new node embedding
        """
        x = torch.mm(x,self.weight) + self.bias
        D = torch.zeros_like(A)

        for i in range(A.shape[0]):
            A[i,i] = 1
            D[i,i] = torch.pow(torch.sum(A[i]),exponent=-0.5)
        A = torch.mm(torch.mm(D,A),D)
        return torch.mm(A,x)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.num_inputs}, out_features={self.num_outputs})"
