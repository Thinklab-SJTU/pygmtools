import torch
import pygmtools.utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
import functools

from .pytorch_backend import hungarian, _load_model
try:
    from pygmtools.c_astar import c_astar
except ImportError:
    raise ImportError(
        'Error when importing the shared library of c_astar. Please 1) try reinstalling '
        'pygmtools; or 2) try the solution here to compile the Cython code locally '
        'https://github.com/Thinklab-SJTU/pygmtools/issues/92#issuecomment-1850403638')


VERY_LARGE_INT = 65536


astar_pretrain_path = {
    'AIDS700nef': (['https://huggingface.co/heatingma/pygmtools/resolve/main/best_genn_AIDS700nef_gcn_astar.pt',
                    'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/pytorch_backend/best_genn_AIDS700nef_gcn_astar.pt'],
                    'b2516aea4c8d730704a48653a5ca94ba'),
    'LINUX': (['https://huggingface.co/heatingma/pygmtools/resolve/main/best_genn_LINUX_gcn_astar.pt',
               'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/pytorch_backend/best_genn_LINUX_gcn_astar.pt'],
               'fd3b2a8dfa3edb20607da2e2b96d2e96'),
}


###############################################################
#                  A*  Wrapper functions                      #
###############################################################

def classic_astar_kernel(K_padded, n1, n2, beam_width):
    """
    The true implementation of astar function
    """
    cache_dict = {}
    hun_func = functools.partial(heuristic_prediction_hun, cache_dict=cache_dict)

    x_pred, _ = c_astar(
        None,
        -K_padded, # maximize problem -> minimize problem
        n1, n2,
        None,
        hun_func,
        net_pred=False,
        beam_width=beam_width,
        trust_fact=1.,
        no_pred_size=0,
    )

    return x_pred


def genn_astar_kernel(feat1, feat2, A1, A2, n1, n2, channel, filters_1, filters_2, filters_3,
                 tensor_neurons, beam_width, trust_fact, no_pred_size, network, pretrain, use_net):
    """
    The true implementation of genn_astar function
    """
    if feat1 is None:
        forward_pass = False
        device = torch.device('cpu')
    else:
        if not feat1.shape[-1] == feat2.shape[-1]:
            raise ValueError('The feature dimensions of feat1 and feat2 must be consistent')
        forward_pass = True
        device = feat1.device

    if network is None:
        args = default_parameter()
        if device != torch.device('cpu'):
            args["cuda"] = True
        if forward_pass:
            if channel is None:
                args['channel'] = feat1.shape[-1]
            else:
                if not feat1.shape[-1] == channel:
                    raise ValueError(f'the channel {channel} must match the feature dimension of feat1')
                args['channel'] = channel
        else:
            if channel is None:
                args['channel'] = 8 if pretrain == "LINUX" else 36
            else:
                args['channel'] = channel

        args['filters_1'] = filters_1
        args['filters_2'] = filters_2
        args['filters_3'] = filters_3
        args['tensor_neurons'] = tensor_neurons
        args['astar_beam_width'] = beam_width
        args['astar_trust_fact'] = trust_fact
        args['astar_no_pred'] = no_pred_size
        args['pretrain'] = pretrain
        args['use_net'] = use_net

        network = GENN(args)

        network = network.to(device)
        if pretrain and args['use_net']:
            if pretrain in astar_pretrain_path:
                url, md5 = astar_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'best_genn_{pretrain}_gcn_astar.pt', url, md5)
                if check_layer_parameter(args):
                    _load_model(network, filename, device)
                else:
                    _load_model(network, filename, device, strict=False)
                    message = 'Warning: Pretrain {} does not support the parameters you entered, '.format(pretrain)
                    if args['pretrain'] == 'AIDS700nef':
                        message += "Supported parameters: ( channel:36, filters:(64,32,16) ), "
                    elif args['pretrain'] == 'LINUX':
                        message += "Supported parameters: ( channel:8, filters:(64,32,16) ), "
                    message += 'Input parameters: ( channel:{}, filters:({},{},{}) )'.format(args['channel'],
                                                                                             args['filters_1'],
                                                                                             args['filters_2'],
                                                                                             args['filters_3'])
                    print(message)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {astar_pretrain_path.keys()}')

    if forward_pass:
        if not A1.shape[0] == A2.shape[0]:
            raise ValueError('Batch dimension does not match')
        data = GraphPair(feat1, feat2, A1, A2, n1, n2)
        result = network(data)
    else:
        result = None
    return result, network


###############################################################
#                     Helper  Functions                       #
###############################################################


def hungarian_ged(node_cost_mat: torch.Tensor, n1, n2):
    if not node_cost_mat.shape[-2] == n1 + 1:
        raise RuntimeError(f'nost_cost_mat dimension mismatch in hungarian_ged. Got {node_cost_mat.shape[-2]} in dim '
                           f'-2 but {n1 + 1} is expected')
    if not node_cost_mat.shape[-1] == n2 + 1:
        raise RuntimeError(f'nost_cost_mat dimension mismatch in hungarian_ged. Got {node_cost_mat.shape[-1]} in dim '
                           f'-1 but {n2 + 1} is expected')
    device = node_cost_mat.device
    upper_left = node_cost_mat[:n1, :n2]
    upper_right = torch.full((n1, n1), float('inf'), device=device)
    torch.diagonal(upper_right)[:] = node_cost_mat[:-1, -1]
    lower_left = torch.full((n2, n2), float('inf'), device=device)
    torch.diagonal(lower_left)[:] = node_cost_mat[-1, :-1]
    lower_right = torch.zeros((n2, n1), device=device)
    large_cost_mat = torch.cat((torch.cat((upper_left, upper_right), dim=1),
                                torch.cat((lower_left, lower_right), dim=1)), dim=0)

    large_pred_x = hungarian(-large_cost_mat.unsqueeze(dim=0)).squeeze()
    pred_x = torch.zeros_like(node_cost_mat)
    pred_x[:n1, :n2] = large_pred_x[:n1, :n2]
    pred_x[:-1, -1] = torch.sum(large_pred_x[:n1, n2:], dim=1)
    pred_x[-1, :-1] = torch.sum(large_pred_x[n1:, :n2], dim=0)

    ged_lower_bound = torch.sum(pred_x * node_cost_mat)
    return pred_x, ged_lower_bound


def heuristic_prediction_hun(k: torch.Tensor, n1, n2, partial_pmat, cache_dict: dict=None):
    if cache_dict is not None and 'node_cost' in cache_dict:
        node_cost_mat = cache_dict['node_cost']
    else:
        k_prime = k.reshape(-1, n1 + 1, n2 + 1)
        node_costs = torch.empty(k_prime.shape[0])
        for i in range(k_prime.shape[0]):
            _, node_costs[i] = hungarian_ged(k_prime[i], n1, n2)
        node_cost_mat = node_costs.reshape(n1 + 1, n2 + 1)
        if cache_dict is not None:
            cache_dict['node_cost'] = node_cost_mat

    graph_1_mask = ~partial_pmat.sum(dim=-1).to(dtype=torch.bool)
    graph_2_mask = ~partial_pmat.sum(dim=-2).to(dtype=torch.bool)
    graph_1_mask[-1] = 1
    graph_2_mask[-1] = 1
    node_cost_mat = node_cost_mat[graph_1_mask, :]
    node_cost_mat = node_cost_mat[:, graph_2_mask]

    _, ged = hungarian_ged(node_cost_mat, torch.sum(graph_1_mask[:-1]), torch.sum(graph_2_mask[:-1]))

    return ged


def default_parameter():
    params = dict()
    params['cuda'] = False
    params['pretrain'] = False
    params['channel'] = 36
    params['filters_1'] = 64
    params['filters_2'] = 32
    params['filters_3'] = 16
    params['tensor_neurons'] = 16
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


class GENN(torch.nn.Module):
    def __init__(self, args: dict):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GENN, self).__init__()
        self.training = False
        self.args = args
        if self.args['use_net']:
            self.number_labels = self.args['channel']
            self.setup_layers()

        self.reset_cache()

    def reset_cache(self):
        self.gnn_1_cache = dict()
        self.gnn_2_cache = dict()
        self.heuristic_cache = dict()

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.feature_count = self.args['tensor_neurons']
        self.convolution_1 = GCNConv(self.number_labels, self.args['filters_1'])
        self.convolution_2 = GCNConv(self.args['filters_1'], self.args['filters_2'])
        self.convolution_3 = GCNConv(self.args['filters_2'], self.args['filters_3'])
        self.attention = AttentionModule(self.args)
        self.tensor_network = TensorNetworkModule(self.args)
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(self.feature_count, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def convolutional_pass(self, edge_index, x, edge_weight=None):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param x: Feature matrix.
        :param edge_weight: Edge weights.
        :return features: Abstract feature matrix.
        """

        features = self.convolution_1(edge_index, x, edge_weight)
        features = F.relu(features)
        features = self.convolution_2(edge_index, features, edge_weight)
        features = F.relu(features)
        features = self.convolution_3(edge_index, features, edge_weight)
        return features

    def forward(self, data: GraphPair):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        num = data.g1.num_graphs
        max_nodes_num_1 = torch.max(data.g1.nodes_num) + 1
        max_nodes_num_2 = torch.max(data.g2.nodes_num) + 1
        x_pred = torch.zeros(num, max_nodes_num_1, max_nodes_num_2)
        for i in range(num):
            x1 = data.g1.x[i]
            x2 = data.g2.x[i]
            adj1 = data.g1.adj[i]
            adj2 = data.g2.adj[i]
            n1 = data.g1.nodes_num[i]
            n2 = data.g2.nodes_num[i]
            exchange = True if x1.shape[0] > x2.shape[0] else False
            if not exchange:
                cur_data = GraphPair(x1, x2, adj1, adj2, n1, n2)
            else:
                cur_data = GraphPair(x2, x1, adj2, adj1, n2, n1)
            num_nodes_1 = data.g1.nodes_num[i] + 1
            num_nodes_2 = data.g2.nodes_num[i] + 1
            _x_pred = self._astar(cur_data)
            x_pred[i][:num_nodes_1, :num_nodes_2] = _x_pred.T if exchange else _x_pred
        return x_pred[:, :-1, :-1]

    def _astar(self, data: GraphPair):
        if self.args['cuda']:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        edge_index_1 = data.g1.edge_index.squeeze()
        edge_index_2 = data.g2.edge_index.squeeze()
        edge_attr_1 = data.g1.edge_weight
        edge_attr_2 = data.g2.edge_weight
        node_1 = data.g1.x.squeeze()
        node_2 = data.g2.x.squeeze()
        batch_1 = data.g1.batch.to(device)
        batch_2 = data.g2.batch.to(device)
        batch_num = data.g1.num_graphs

        ns_1 = torch.bincount(data.g1.batch)
        ns_2 = torch.bincount(data.g2.batch)

        if torch.any(ns_1 > ns_2):
            raise ValueError('Number of nodes in graph 1 should always <= number of nodes in graph 2.')

        adj_1 = to_dense_adj(edge_index_1, batch=batch_1, edge_attr=edge_attr_1)
        dummy_adj_1 = torch.zeros(adj_1.shape[0], adj_1.shape[1] + 1, adj_1.shape[2] + 1, device=device)
        dummy_adj_1[:, :-1, :-1] = adj_1
        adj_2 = to_dense_adj(edge_index_2, batch=batch_2, edge_attr=edge_attr_2)
        dummy_adj_2 = torch.zeros(adj_2.shape[0], adj_2.shape[1] + 1, adj_2.shape[2] + 1, device=device)
        dummy_adj_2[:, :-1, :-1] = adj_2

        node_1, _ = to_dense_batch(node_1, batch=batch_1)
        node_2, _ = to_dense_batch(node_2, batch=batch_2)

        dummy_node_1 = torch.zeros(adj_1.shape[0], node_1.shape[1] + 1, node_1.shape[-1], device=device)
        dummy_node_1[:, :-1, :] = node_1
        dummy_node_2 = torch.zeros(adj_2.shape[0], node_2.shape[1] + 1, node_2.shape[-1], device=device)
        dummy_node_2[:, :-1, :] = node_2
        k_diag = node_metric(dummy_node_1, dummy_node_2)

        mask_1 = torch.zeros_like(dummy_adj_1)
        mask_2 = torch.zeros_like(dummy_adj_2)
        for b in range(batch_num):
            mask_1[b, :ns_1[b] + 1, :ns_1[b] + 1] = 1
            mask_1[b, :ns_1[b], :ns_1[b]] -= torch.eye(ns_1[b], device=mask_1.device)
            mask_2[b, :ns_2[b] + 1, :ns_2[b] + 1] = 1
            mask_2[b, :ns_2[b], :ns_2[b]] -= torch.eye(ns_2[b], device=mask_2.device)

        a1 = dummy_adj_1.reshape(batch_num, -1, 1)
        a2 = dummy_adj_2.reshape(batch_num, 1, -1)
        m1 = mask_1.reshape(batch_num, -1, 1)
        m2 = mask_2.reshape(batch_num, 1, -1)
        k = torch.abs(a1 - a2) * torch.bmm(m1, m2)
        k[torch.logical_not(torch.bmm(m1, m2).to(dtype=torch.bool))] = VERY_LARGE_INT
        k = k.reshape(batch_num, dummy_adj_1.shape[1], dummy_adj_1.shape[2], dummy_adj_2.shape[1],
                      dummy_adj_2.shape[2])
        k = k.permute([0, 1, 3, 2, 4]) # shape: batch_num x n1+1 x n2+1 x n1+1 x n2+1

        x_pred = torch.zeros((batch_num, ns_1.max() + 1, ns_2.max() + 1), device=k.device)
        for b in range(batch_num):
            k_b = k[b, :ns_1[b] + 1, :ns_2[b] + 1, :ns_1[b] + 1, :ns_2[b] + 1]
            k_b = k_b.reshape((ns_1[b] + 1) * (ns_2[b] + 1), (ns_1[b] + 1) * (ns_2[b] + 1))
            k_b = k_b / 2

            k_diag_view = torch.diagonal(k_b)
            k_diag_view[:] = k_diag[b, :ns_1[b] + 1, :ns_2[b] + 1].reshape(-1)

            self.reset_cache()

            heuristic_func = functools.partial(heuristic_prediction_hun, cache_dict=self.heuristic_cache)

            x_pred_b, _ = c_astar(
                data, k_b, ns_1[b].item(), ns_2[b].item(),
                self.net_prediction_cache,
                heuristic_func,
                net_pred=self.args['use_net'],
                beam_width=self.args['astar_beam_width'],
                trust_fact=self.args['astar_trust_fact'],
                no_pred_size=self.args['astar_no_pred'],
            )
            x_pred[b, :ns_1[b] + 1, :ns_2[b] + 1] = x_pred_b

        return x_pred

    def net_prediction_cache(self, data: GraphPair, partial_pmat=None, return_ged_norm=False):
        """
        Forward pass with graphs.
        :param data: Data class.
        :param partial_pmat: Matched matrix.
        :param return_ged_norm: Whether to return to Normal Graph Edit Distance.
        :return score: Similarity score.
        """
        features_1 = data.g1.x.squeeze()
        features_2 = data.g2.x.squeeze()
        batch_1 = data.g1.batch.to(features_1.device)
        batch_2 = data.g2.batch.to(features_2.device)
        adj1 = data.g1.adj.squeeze()
        adj2 = data.g2.adj.squeeze()

        if 'gnn_feat' not in self.gnn_1_cache:
            abstract_features_1 = self.convolutional_pass(adj1, features_1)
            self.gnn_1_cache['gnn_feat'] = abstract_features_1
        else:
            abstract_features_1 = self.gnn_1_cache['gnn_feat']
        if 'gnn_feat' not in self.gnn_2_cache:
            abstract_features_2 = self.convolutional_pass(adj2, features_2)
            self.gnn_2_cache['gnn_feat'] = abstract_features_2
        else:
            abstract_features_2 = self.gnn_2_cache['gnn_feat']

        graph_1_mask = torch.ones_like(batch_1)
        graph_2_mask = torch.ones_like(batch_2)
        graph_1_matched = partial_pmat.sum(dim=-1).to(dtype=torch.bool)[:graph_1_mask.shape[0]]
        graph_2_matched = partial_pmat.sum(dim=-2).to(dtype=torch.bool)[:graph_2_mask.shape[0]]
        graph_1_mask = torch.logical_not(graph_1_matched)
        graph_2_mask = torch.logical_not(graph_2_matched)
        abstract_features_1 = abstract_features_1[graph_1_mask]
        abstract_features_2 = abstract_features_2[graph_2_mask]
        batch_1 = batch_1[graph_1_mask]
        batch_2 = batch_2[graph_2_mask]
        pooled_features_1 = self.attention(abstract_features_1, batch_1)
        pooled_features_2 = self.attention(abstract_features_2, batch_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        score = self.scoring_layer(scores).view(-1)

        if return_ged_norm:
            return score
        else:
            ged = - torch.log(score) * (batch_1.shape[0] + batch_2.shape[0]) / 2
            return ged


