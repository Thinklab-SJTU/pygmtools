import torch
import numpy as np
from multiprocessing import Pool
from torch import Tensor
from pygmtools.numpy_backend import _hung_kernel


#############################################
#     Linear Assignment Problem Solvers     #
#############################################


def hungarian(s: Tensor, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    """
    Pytorch implementation of Hungarian algorithm
    """
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

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat


def sinkhorn(s: Tensor, nrows: Tensor=None, ncols: Tensor=None,
             dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False) -> Tensor:
    """
    Pytorch implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.transpose(1, 2)
        nrows, ncols = ncols, nrows
        transposed = True

    if nrows is None:
        nrows = [s.shape[1] for _ in range(batch_size)]
    if ncols is None:
        ncols = [s.shape[2] for _ in range(batch_size)]

    # operations are performed on log_s
    s = s / tau

    if dummy_row:
        assert s.shape[2] >= s.shape[1]
        dummy_shape = list(s.shape)
        dummy_shape[1] = s.shape[2] - s.shape[1]
        ori_nrows = nrows
        nrows = ncols
        s = torch.cat((s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
        for b in range(batch_size):
            s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
            s[b, nrows[b]:, :] = -float('inf')
            s[b, :, ncols[b]:] = -float('inf')

    if batched_operation:
        log_s = s

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                log_s = log_s - log_sum
                log_s[torch.isnan(log_s)] = -float('inf')
            else:
                log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - log_sum
                log_s[torch.isnan(log_s)] = -float('inf')

        if dummy_row and dummy_shape[1] > 0:
            log_s = log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        return torch.exp(log_s)
    else:
        ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s = s[b, row_slice, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                else:
                    log_sum = torch.logsumexp(log_s, 0, keepdim=True)
                    log_s = log_s - log_sum

            ret_log_s[b, row_slice, col_slice] = log_s

        if dummy_row:
            if dummy_shape[1] > 0:
                ret_log_s = ret_log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        if transposed:
            ret_log_s = ret_log_s.transpose(1, 2)

        return torch.exp(ret_log_s)


#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################


def rrwm(K: Tensor, n1: Tensor, n2: Tensor, n1max, n2max, x0: Tensor,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> Tensor:
    """
    Pytorch implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
    d = K.sum(dim=2, keepdim=True)
    dmax = d.max(dim=1, keepdim=True).values
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = torch.bmm(K, v)
        last_v = v
        n = torch.norm(v, p=1, dim=1, keepdim=True)
        v = v / n

        # reweighted jump
        s = v.view(batch_num, n2max, n1max).transpose(1, 2)
        s = beta * s / s.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter).transpose(1, 2).reshape(batch_num, n1n2, 1) + \
            (1 - alpha) * v
        n = torch.norm(v, p=1, dim=1, keepdim=True)
        v = torch.matmul(v, 1 / n)

        if torch.norm(v - last_v) < 1e-5:
            break

    return v.view(batch_num, n2max, n1max).transpose(1, 2)


def sm(K: Tensor, n1: Tensor, n2: Tensor, n1max, n2max, x0: Tensor,
       max_iter: int) -> Tensor:
    """
    Pytorch implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for i in range(max_iter):
        v = torch.bmm(K, v)
        n = torch.norm(v, p=2, dim=1)
        v = torch.matmul(v, (1 / n).view(batch_num, 1, 1))
        if torch.norm(v - vlast) < 1e-5:
            break
        vlast = v

    x = v.view(batch_num, n2max, n1max).transpose(1, 2)
    return x

def ipfp(K: Tensor, n1: Tensor, n2: Tensor, n1max, n2max, x0: Tensor,
         max_iter) -> Tensor:
    """
    Pytorch implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v

    def comp_obj_score(v1, K, v2):
        return torch.bmm(torch.bmm(v1.view(batch_num, 1, -1), K), v2)

    for i in range(max_iter):
        cost = torch.bmm(K, v).reshape(batch_num, n2max, n1max).transpose(1, 2)
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.transpose(1, 2).view(batch_num, -1, 1)
        alpha = comp_obj_score(v, K, binary_v - v)  # + torch.mm(k_diag.view(1, -1), (binary_sol - v).view(-1, 1))
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = alpha / beta
        v = torch.where(torch.logical_or(beta <= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_sol = comp_obj_score(last_v, K, last_v)
        if torch.max(torch.abs(
                last_v_sol - torch.bmm(cost.reshape(batch_num, 1, -1), binary_sol.reshape(batch_num, -1, 1))
        ) / last_v_sol) < 1e-3:
            break
        last_v = v

    pred_x = binary_sol
    return pred_x


def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = torch.full(batch_num, n1max, device=K.device)
    if n2 is None:
        n2 = torch.full(batch_num, n2max, device=K.device)
    if n1max is None:
        n1max = torch.max(n1)
    if n2max is None:
        n2max = torch.max(n2)

    assert n1max * n2max == n1n2, 'the input size of K does not match with n1max * n2max!'

    # initialize x0 (also v0)
    if x0 is None:
        x0 = torch.zeros(batch_num, n1max, n2max, dtype=K.dtype, device=K.device)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = torch.tensor(1.) / (n1[b] * n2[b])
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


#############################################
#              Utils Functions              #
#############################################


def inner_prod_aff_fn(feat1, feat2):
    """
    Pytorch implementation of inner product affinity function
    """
    return torch.matmul(feat1, feat2.transpose(1, 2))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    Pytorch implementation of Gaussian affinity function
    """
    feat1 = feat1.unsqueeze(2)
    feat2 = feat2.unsqueeze(1)
    return torch.exp(-((feat1 - feat2) ** 2).sum(dim=-1) / sigma)


def build_batch(input):
    """
    Pytorch implementation of building a batched tensor
    """
    assert type(input[0]) == torch.Tensor
    device = input[0].device
    it = iter(input)
    t = next(it)
    max_shape = list(t.shape)
    ori_shape = [[_] for _ in max_shape]
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
                ori_shape[i].append(t.shape[i])
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in input:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_pattern = tuple(pad_pattern.tolist())
        padded_ts.append(torch.nn.functional.pad(t, pad_pattern, 'constant', 0))

    return torch.stack(padded_ts, dim=0), *[torch.tensor(_, dtype=torch.int64, device=device) for _ in ori_shape]


def dense_to_sparse(dense_adj):
    """
    Pytorch implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn = build_batch([torch.nonzero(a, as_tuple=False) for a in dense_adj])
    edge_weight = build_batch([dense_adj[b][(conn[b, :, 0], conn[b, :, 1])] for b in range(batch_size)]).unsqueeze(-1)
    return conn, edge_weight


def to_numpy(input):
    """
    Pytorch function to_numpy
    """
    return input.detach().cpu().numpy()


def from_numpy(input):
    """
    Pytorch function from_numpy
    """
    return torch.from_numpy(input)


def _aff_mat_from_node_edge_aff(node_aff: Tensor, edge_aff: Tensor, connectivity1: Tensor, connectivity2: Tensor,
                                n1, n2, ne1, ne2):
    """
    Pytorch implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        device = edge_aff.device
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = torch.max(torch.max(connectivity1, dim=-1).values, dim=-1).values + 1
        if n2 is None:
            n2 = torch.max(torch.max(connectivity2, dim=-1).values, dim=-1).values + 1
        if ne1 is None:
            ne1 = [edge_aff.shape[1]] * batch_size
        if ne2 is None:
            ne2 = [edge_aff.shape[1]] * batch_size
    else:
        device = node_aff.device
        dtype = node_aff.dtype
        batch_size = node_aff.shape[0]
        if n1 is None:
            n1 = [node_aff.shape[1]] * batch_size
        if n2 is None:
            n2 = [node_aff.shape[2]] * batch_size

    n1max = max(n1)
    n2max = max(n2)
    ks = []
    for b in range(batch_size):
        k = torch.zeros(n2max, n1max, n2max, n1max, dtype=dtype, device=device)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:ne1[b]]
            conn2 = connectivity2[b][:ne2[b]]
            edge_indices = torch.cat([conn1.repeat_interleave(ne2[b], dim=0), conn2.repeat(ne1[b], 1)], dim=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b].reshape(-1)
        k = k.reshape(n2max * n1max, n2max * n1max)
        # node-wise affinity
        if node_aff is not None:
            k_diag = torch.diagonal(k)
            k_diag[:] = node_aff[b].transpose(0, 1).reshape(-1)
        ks.append(k)

    return torch.stack(ks, dim=0)


def _check_data_type(input: Tensor):
    """
    Pytorch implementation of _check_data_type
    """
    if type(input) is not Tensor:
        raise ValueError(f'Expected Pytorch Tensor, but got {type(input)}. Perhaps the wrong backend?')


def _check_shape(input, dim_num):
    """
    Pytorch implementation of _check_shape
    """
    return len(input.shape) == dim_num


def _get_shape(input):
    """
    Pytorch implementation of _get_shape
    """
    return input.shape


def _squeeze(input, dim):
    """
    Pytorch implementation of _squeeze
    """
    return input.squeeze(dim)


def _unsqueeze(input, dim):
    """
    Pytorch implementation of _unsqueeze
    """
    return input.unsqueeze(dim)
