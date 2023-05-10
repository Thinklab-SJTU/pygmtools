import torch
import pygmtools.utils
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from torch_scatter import scatter
from pytorch_backend import hungarian_ged,to_dense_adj,to_dense_batch
VERY_LARGE_INT = 65536

if not os.path.exists("a_star.cpp"):
    command = "python a_star_setup.py build_ext --inplace"
    os.system(command)
    
from a_star import a_star

############################################
#              GENN-A*  Modules            #
############################################


class Graph_pair():
    def __init__(self,x1,x2,Adj1,Adj2,n1=None,n2=None):        
        self.g1 = Graphs(x1,Adj1,n1)
        self.g2 = Graphs(x2,Adj2,n2)        
    def __repr__(self):
        return f"{self.__class__.__name__}('g1' = {self.g1}, 'g2' = {self.g2})"
    def to_dict(self):
        data = {}
        data['g1'] = self.g1
        data['g2'] = self.g2
        return data
    
class Graphs():
    def __init__(self,x,Adj,nodes_num=None):
        assert len(x.shape) == len(Adj.shape)
        if(len(Adj.shape) == 2):
            Adj = Adj.unsqueeze(dim=0)
            x = x.unsqueeze(dim=0)
        assert x.shape[0] == Adj.shape[0]
        assert x.shape[1] == Adj.shape[1]
        self.x = x
        self.Adj = Adj
        self.num_graphs = Adj.shape[0]
        if nodes_num is not None:
            self.nodes_num = nodes_num  
        else:
            self.nodes_num = torch.tensor([x.shape[1]]*x.shape[0])
        self.graph_process()
        
    def graph_process(self):
        edge_index,edge_weight,_ = pygmtools.utils.dense_to_sparse(self.Adj)
        self.edge_index = torch.cat([edge_index[:,:,0].unsqueeze(dim=1),edge_index[:,:,1].unsqueeze(dim=1)],dim=1)
        self.edge_weight = edge_weight.view(-1)
        if(self.nodes_num.shape == torch.Size([])):
            batch = torch.tensor([0] * self.nodes_num)
        else:
            for i in range(len(self.nodes_num)):
                if(i == 0):
                    batch = torch.tensor([i] * self.nodes_num[i])
                else:
                    cur_batch = torch.tensor([i] * self.nodes_num[i])
                    batch = torch.cat([batch,cur_batch])
        self.batch = batch
    def __repr__(self):
        messgae = "x = {}, Adj = {}".format(list(self.x.shape),list(self.Adj.shape))
        messgae += " edge_index = {}, edge_weight = {}".format(list(self.edge_index.shape),list(self.edge_weight.shape))
        messgae += " nodes_num = {}, num_graphs = {})".format(self.nodes_num.shape,self.num_graphs)
        self.message = messgae
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
        mean = scatter(x, batch, dim=0, dim_size=size, reduce='mean')
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))
        
        coefs = torch.sigmoid((x * transformed_global[batch] * 10).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x
        
        return scatter(weighted, batch, dim=0, dim_size=size, reduce='add')
        
    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))

class DenseAttentionModule(torch.nn.Module):
    """
    SimGNN Dense Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(DenseAttentionModule, self).__init__()
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

    def forward(self, x, mask=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param mask: Mask matrix indicating the valid nodes for each graph. 
        :return representation: A graph level representation matrix. 
        """
        B, N, _ = x.size()
        
        if mask is not None:
            num_nodes = mask.view(B, N).sum(dim=1).unsqueeze(-1)
            mean = x.sum(dim=1)/num_nodes.to(x.dtype)
        else:
            mean = x.mean(dim=1)
        
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))
        
        koefs = torch.sigmoid(torch.matmul(x, transformed_global.unsqueeze(-1)))
        weighted = koefs * x
        
        if mask is not None:
            weighted = weighted * mask.view(B, N, 1).to(x.dtype)
        
        return weighted.sum(dim=1)

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
    def __init__(self, args):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GENN, self).__init__()
        self.args = args
        if self.args['use_net']:
            self.number_labels = self.args['feature_num']
            self.setup_layers()

        self.reset_cache()

    def reset_cache(self):
        self.gnn_1_cache = dict()
        self.gnn_2_cache = dict()
        self.heuristic_cache = dict()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args['histogram']:
            self.feature_count = self.args['tensor_neurons'] + self.args['bins']
        else:
            self.feature_count = self.args['tensor_neurons']

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
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

    def calculate_histogram(self, abstract_features_1, abstract_features_2, batch_1, batch_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for target graphs.
        :param abstract_features_2: Feature matrix for source graphs.
        :param batch_1: Batch vector for source graphs, which assigns each node to a specific example
        :param batch_1: Batch vector for target graphs, which assigns each node to a specific example
        :return hist: Histogram of similarity scores.
        """
        abstract_features_1, mask_1 = to_dense_batch(abstract_features_1, batch_1)
        abstract_features_2, mask_2 = to_dense_batch(abstract_features_2, batch_2)

        B1, N1, _ = abstract_features_1.size()
        B2, N2, _ = abstract_features_2.size()

        mask_1 = mask_1.view(B1, N1)
        mask_2 = mask_2.view(B2, N2)
        num_nodes = torch.max(mask_1.sum(dim=1), mask_2.sum(dim=1))

        scores = torch.matmul(abstract_features_1, abstract_features_2.permute([0, 2, 1])).detach()

        hist_list = []
        for i, mat in enumerate(scores):
            mat = torch.sigmoid(mat[:num_nodes[i], :num_nodes[i]]).view(-1)
            hist = torch.histc(mat, bins=self.args['bins'])
            hist = hist / torch.sum(hist)
            hist = hist.view(1, -1)
            hist_list.append(hist)

        return torch.stack(hist_list).view(-1, self.args['bins'])

    def convolutional_pass(self, A, x, edge_weight=None):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """

        features = self.convolution_1(A, x, edge_weight)
        features = F.relu(features)
        features = F.dropout(features, p=self.args['dropout'], training=self.training)
        features = self.convolution_2(A, features, edge_weight)
        features = F.relu(features)
        features = F.dropout(features, p=self.args['dropout'], training=self.training)
        features = self.convolution_3(A, features, edge_weight)
        return features

    def diffpool(self, abstract_features, edge_index, batch):
        """
        Making differentiable pooling.
        :param abstract_features: Node feature matrix.
        :param batch: Batch vector, which assigns each node to a specific example
        :return pooled_features: Graph feature matrix.
        """
        x, mask = to_dense_batch(abstract_features, batch)
        adj = to_dense_adj(edge_index, batch)
        return self.attention(x, adj, mask)

    def forward(self,data:Graph_pair):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        num = data.g1.num_graphs
        for i in range(num):
            cur_data = Graph_pair(data.g1.x[i],data.g2.x[i],data.g1.Adj[i],data.g2.Adj[i],
                                data.g1.nodes_num[i],data.g2.nodes_num[i])   
            if(i == 0):
                x_pred = self.A_star(cur_data)
            else:
                x_pred = torch.cat([x_pred,self.A_star(cur_data)],dim=0)
        return x_pred[:,:-1,:-1]

    def A_star(self,data:Graph_pair):
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
        batch_1 = data.g1.batch
        batch_2 = data.g2.batch
        batch_num = data.g1.num_graphs

        ns_1 = torch.bincount(data.g1.batch)
        ns_2 = torch.bincount(data.g2.batch)
        
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
        k_diag = self.node_metric(dummy_node_1, dummy_node_2)
        
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
        k = k.reshape(batch_num, dummy_adj_1.shape[1], dummy_adj_1.shape[2], dummy_adj_2.shape[1], dummy_adj_2.shape[2])
        k = k.permute([0, 1, 3, 2, 4])
        k = k.reshape(batch_num, dummy_adj_1.shape[1] * dummy_adj_2.shape[1], dummy_adj_1.shape[2] * dummy_adj_2.shape[2])
        k = k / 2
        
        for b in range(batch_num):
            k_diag_view = torch.diagonal(k[b])
            k_diag_view[:] = k_diag[b].reshape(-1)

        self.reset_cache()
        
        x_pred, _ = a_star(
            data, k, ns_1.cpu().numpy(), ns_2.cpu().numpy(),
            self.net_prediction_cache,
            self.heuristic_prediction_hun,
            net_pred=self.args['use_net'],
            beam_width=self.args['astar_beamwidth'],
            trust_fact=self.args['astar_trustfact'],
            no_pred_size=self.args['astar_nopred'],
        )
        
        return x_pred
    
    def node_metric(self,node1,node2):
        
        encoding = torch.sum(torch.abs(node1.unsqueeze(2) - node2.unsqueeze(1)), dim=-1)
        non_zero = torch.nonzero(encoding)
        for i in range(non_zero.shape[0]):
            encoding[non_zero[i][0],non_zero[i][1],non_zero[i][2]] = 1
        return encoding
  
    def net_prediction_cache(self, data:Graph_pair, partial_pmat=None, return_ged_norm=False):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data.g1.edge_index.squeeze()
        edge_index_2 = data.g2.edge_index.squeeze()
        features_1 = data.g1.x.squeeze()
        features_2 = data.g2.x.squeeze()
        batch_1 = data.g1.batch
        batch_2 = data.g2.batch
        Adj1 = data.g1.Adj.squeeze()
        Adj2 = data.g2.Adj.squeeze()
        if 'gnn_feat' not in self.gnn_1_cache:
            abstract_features_1 = self.convolutional_pass(Adj1,features_1)
            self.gnn_1_cache['gnn_feat'] = abstract_features_1
        else:
            abstract_features_1 = self.gnn_1_cache['gnn_feat']
        if 'gnn_feat' not in self.gnn_2_cache:
            abstract_features_2 = self.convolutional_pass(Adj2,features_2)
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
        
        if self.args['histogram']:
            hist = self.calculate_histogram(abstract_features_1, abstract_features_2, batch_1, batch_2)

        if self.args['diffpool']:
            pooled_features_1 = self.diffpool(abstract_features_1, edge_index_1, batch_1)
            pooled_features_2 = self.diffpool(abstract_features_2, edge_index_2, batch_2)
        else:
            pooled_features_1 = self.attention(abstract_features_1, batch_1)
            pooled_features_2 = self.attention(abstract_features_2, batch_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        
        if self.args['histogram']:
            scores = torch.cat((scores, hist), dim=1)

        score = self.scoring_layer(scores).view(-1)
        
        if return_ged_norm:
            return score
        else:
            ged = - torch.log(score) * (batch_1.shape[0] + batch_2.shape[0]) / 2
            return ged

    def heuristic_prediction_hun(self, k, n1, n2, partial_pmat):
        k_prime = k.reshape(-1, n1+1, n2+1)
        node_costs = torch.empty(k_prime.shape[0])
        for i in range(k_prime.shape[0]):
            _, node_costs[i] = hungarian_ged(k_prime[i], n1, n2)
        node_cost_mat = node_costs.reshape(n1+1, n2+1)
        self.heuristic_cache['node_cost'] = node_cost_mat

        graph_1_mask = ~partial_pmat.sum(dim=-1).to(dtype=torch.bool)
        graph_2_mask = ~partial_pmat.sum(dim=-2).to(dtype=torch.bool)
        graph_1_mask[-1] = 1
        graph_2_mask[-1] = 1
        node_cost_mat = node_cost_mat[graph_1_mask, :]
        node_cost_mat = node_cost_mat[:, graph_2_mask]

        _, ged = hungarian_ged(node_cost_mat, torch.sum(graph_1_mask[:-1]), torch.sum(graph_2_mask[:-1]))
    
        return ged

