import torch
from multiprocessing import Pool
import scipy.optimize as opt
import numpy as np
from typing import Optional, Tuple
from torch import Tensor
from torch_scatter import scatter_add,scatter

def hungarian_ged(node_cost_mat, n1, n2):
    assert node_cost_mat.shape[-2] == n1+1
    assert node_cost_mat.shape[-1] == n2+1
    device = node_cost_mat.device
    upper_left = node_cost_mat[:n1, :n2]
    upper_right = torch.full((n1, n1), float('inf'), device=device)
    torch.diagonal(upper_right)[:] = node_cost_mat[:-1, -1]
    lower_left = torch.full((n2, n2), float('inf'), device=device)
    torch.diagonal(lower_left)[:] = node_cost_mat[-1, :-1]
    lower_right = torch.zeros((n2, n1), device=device)

    large_cost_mat = torch.cat((torch.cat((upper_left, upper_right), dim=1),
                                torch.cat((lower_left, lower_right), dim=1)), dim=0)

    large_pred_x = hungarian(-large_cost_mat)
    pred_x = torch.zeros_like(node_cost_mat)
    pred_x[:n1, :n2] = large_pred_x[:n1, :n2]
    pred_x[:-1, -1] = torch.sum(large_pred_x[:n1, n2:], dim=1)
    pred_x[-1, :-1] = torch.sum(large_pred_x[n1:, :n2], dim=0)

    ged_lower_bound = torch.sum(pred_x * node_cost_mat)

    return pred_x, ged_lower_bound

def hungarian(s: torch.Tensor, n1=None, n2=None, mask=None, nproc=1):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :param nproc: number of parallel processes (default =1 for no parallel)
    :return: optimal permutation matrix
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood.')

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num
    if mask is None:
        mask = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([hung_kernel(perm_mat[b], n1[b], n2[b], mask[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    if matrix_input:
        perm_mat.squeeze_(0)

    return perm_mat

def hung_kernel(s: torch.Tensor, n1=None, n2=None, mask=None):
    if mask is None:
        if n1 is None:
            n1 = s.shape[0]
        if n2 is None:
            n2 = s.shape[1]
        row, col = opt.linear_sum_assignment(s[:n1, :n2])
    else:
        mask = mask.cpu()
        s_mask = s[mask]
        if s_mask.size > 0:
            dim0 = torch.sum(mask, dim=0).max()
            dim1 = torch.sum(mask, dim=1).max()
            row, col = opt.linear_sum_assignment(s_mask.reshape(dim0, dim1))
            row = torch.nonzero(torch.sum(mask, dim=1), as_tuple=True)[0][row]
            col = torch.nonzero(torch.sum(mask, dim=0), as_tuple=True)[0][col]
        else:
            row, col = [], []
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat

def default_parameter():
    params = {}
    params['cuda'] = False
    params['pretrain'] = False
    params['feature_num'] = 36
    params['filters_1'] = 64
    params['filters_2'] = 32
    params['filters_3'] = 16
    params['tensor_neurons'] = 16
    params['bottle_neck_neurons'] = 16
    params['bins'] = 16
    params['dropout'] = 0
    params['astar_beamwidth'] = 0
    params['astar_trustfact'] = 1
    params['astar_nopred'] = 0
    params['use_net'] = True
    params['histogram'] = False
    params['diffpool'] = False 
    return params

def check_layer_parameter(params):
    if(params['pretrain'] == 'AIDS700nef'):
        if params['feature_num'] != 36:
            return False
    elif(params['pretrain'] == 'LINUX'):
        if params['feature_num'] != 8:
            return False
    if params['filters_1'] != 64:
        return False
    if params['filters_2'] != 32:
        return False
    if params['filters_3'] != 16:
        return False
    if params['tensor_neurons'] != 16:
        return False
    if params['bottle_neck_neurons'] != 16:
        return False
    return True

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

    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0,
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
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='add')
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
    scatter(edge_attr, idx, dim=0, out=adj, reduce='add')
    adj = adj.view(size)

    return adj

