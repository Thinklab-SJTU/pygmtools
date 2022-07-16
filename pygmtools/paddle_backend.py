import paddle
import numpy as np
from multiprocessing import Pool

import pygmtools.utils
from pygmtools.numpy_backend import _hung_kernel


#############################################
#     Linear Assignment Problem Solvers     #
#############################################

def hungarian(s: paddle.Tensor, n1: paddle.Tensor=None, n2: paddle.Tensor=None, nproc: int=1) -> paddle.Tensor:
    """
    Paddle implementation of Hungarian algorithm
    """
    device = s.place
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

    perm_mat = paddle.to_tensor(perm_mat, place=device)

    return perm_mat


def sinkhorn(s: paddle.Tensor, nrows: paddle.Tensor=None, ncols: paddle.Tensor=None,
             dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False) -> paddle.Tensor:
    """
    Paddle implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.transpose((0, 2, 1))
        nrows, ncols = ncols, nrows
        transposed = True

    if nrows is None:
        nrows = paddle.to_tensor([s.shape[1] for _ in range(batch_size)], place=s.place, dtype=paddle.int)
    if ncols is None:
        ncols = paddle.to_tensor([s.shape[2] for _ in range(batch_size)], place=s.place, dtype=paddle.int)

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if paddle.any(transposed_batch):
        s_t = s.transpose((0, 2, 1))
        s_t = paddle.concat((
            s_t[:, :s.shape[1], :],
            paddle.to_tensor(paddle.full((batch_size, s.shape[1], s.shape[2]-s.shape[1]), -float('inf')), place=s.place)), axis=2)
        s = paddle.where(transposed_batch.reshape((batch_size, 1, 1)), s_t, s)

        new_nrows = paddle.where(transposed_batch, ncols, nrows)
        new_ncols = paddle.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

    # operations are performed on log_s
    s = s / tau

    if dummy_row:
        assert s.shape[2] >= s.shape[1]
        dummy_shape = list(s.shape)
        dummy_shape[1] = s.shape[2] - s.shape[1]
        ori_nrows = nrows
        nrows = ncols
        s = paddle.concat((s, paddle.to_tensor(paddle.full(dummy_shape, -float('inf')), place=s.place)), axis=1)
        for b in range(batch_size):
            s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
            s[b, nrows[b]:, :] = -float('inf')
            s[b, :, ncols[b]:] = -float('inf')

    if batched_operation:
        log_s = s

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = paddle.logsumexp(log_s, 2, keepdim=True)
                log_s = log_s - log_sum
                nan_indices = paddle.nonzero(paddle.isnan(log_s), True)
                if nan_indices[0].size > 0:
                    log_s[nan_indices] = -float('inf')
            else:
                log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - log_sum
                nan_indices = paddle.nonzero(paddle.isnan(log_s), True)
                if nan_indices[0].size > 0:
                    log_s[nan_indices] = -float('inf')

        ret_log_s = log_s
    else:
        ret_log_s = paddle.to_tensor(paddle.full((batch_size, s.shape[1], s.shape[2]), -float('inf')), place=s.place, dtype=s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s = s[b, row_slice, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum
                else:
                    log_sum = paddle.logsumexp(log_s, 0, keepdim=True)
                    log_s = log_s - log_sum

            ret_log_s[b, row_slice, col_slice] = log_s

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

    if paddle.any(transposed_batch):
        s_t = ret_log_s.transpose((0, 2, 1))
        s_t = paddle.concat((
            s_t[:, :ret_log_s.shape[1], :],
            paddle.to_tensor(paddle.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2]-ret_log_s.shape[1]), -float('inf')), place=s.place)), axis=2)
        ret_log_s = paddle.where(transposed_batch.reshape((batch_size, 1, 1)), s_t, ret_log_s)

    if transposed:
        ret_log_s = ret_log_s.transpose((0, 2, 1))

    return paddle.exp(ret_log_s)


#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################


def rrwm(K: paddle.Tensor, n1: paddle.Tensor, n2: paddle.Tensor, n1max, n2max, x0: paddle.Tensor,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> paddle.Tensor:
    """
    Paddle implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
    d = paddle.sum(K, axis=2, keepdim=True)
    dmax = paddle.max(d, axis=1, keepdim=True)
    K = K / (dmax + paddle.min(d) * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = paddle.bmm(K, v)
        last_v = v
        n = paddle.norm(v, p=1, axis=1, keepdim=True)
        v = v / n

        # reweighted jump
        s = paddle.reshape(v, (batch_num, n2max, n1max)).transpose((0, 2, 1))
        s = beta * s / s.max(axis=1, keepdim=True).max(axis=2, keepdim=True)
        v = alpha * paddle.reshape(sinkhorn(s, n1, n2, max_iter=sk_iter).transpose((0, 2, 1)),(batch_num, n1n2, 1)) + \
            (1 - alpha) * v
        n = paddle.norm(v, p=1, axis=1, keepdim=True)
        v = paddle.matmul(v, 1 / n)

        if paddle.norm(v - last_v) < 1e-5:
            break

    return v.reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))


def sm(K: paddle.Tensor, n1: paddle.Tensor, n2: paddle.Tensor, n1max, n2max, x0: paddle.Tensor,
       max_iter: int) -> paddle.Tensor:
    """
    Paddle implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for _ in range(max_iter):
        v = paddle.bmm(K, v)
        n = paddle.norm(v, p=2, axis=1)
        v = paddle.matmul(v, paddle.reshape(1 / n, (batch_num, 1, 1)))
        if paddle.norm(v - vlast) < 1e-5:
            break
        vlast = v

    x = paddle.reshape(v, (batch_num, n2max, n1max)).transpose((0, 2, 1))
    return x


def ipfp(K: paddle.Tensor, n1: paddle.Tensor, n2: paddle.Tensor, n1max, n2max, x0: paddle.Tensor,
         max_iter) -> paddle.Tensor:
    """
    Paddle implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v

    def comp_obj_score(v1, K, v2):
        return paddle.bmm(paddle.bmm(paddle.reshape(v1, (batch_num, 1, -1)), K), v2)

    for i in range(max_iter):
        cost = paddle.reshape(paddle.bmm(K, v),(batch_num, n2max, n1max)).transpose((0, 2, 1))
        binary_sol = hungarian(cost, n1, n2)
        binary_v = paddle.reshape(binary_sol.transpose((0, 2, 1)),(batch_num, -1, 1))
        alpha = comp_obj_score(v, K, binary_v - v)  
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = alpha / beta
        v = paddle.where(paddle.logical_or(beta <= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_sol = comp_obj_score(last_v, K, last_v)
        if paddle.max(paddle.abs(
                last_v_sol - paddle.bmm(paddle.reshape(cost,(batch_num, 1, -1)), paddle.reshape(binary_sol, (batch_num, -1, 1)))
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
        n1 = paddle.to_tensor(paddle.full((batch_num,), n1max, dtype=paddle.int32), place=K.place)
    if n2 is None:
        n2 = paddle.to_tensor(paddle.full((batch_num,), n2max, dtype=paddle.int32), place=K.place)
    if n1max is None:
        n1max = paddle.max(n1)
    if n2max is None:
        n2max = paddle.max(n2)

    assert n1max * n2max == n1n2, 'the input size of K does not match with n1max * n2max!'

    # initialize x0 (also v0)
    if x0 is None:
        x0 = paddle.to_tensor(paddle.zeros((batch_num, n1max, n2max), dtype=K.dtype), place=K.place)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = paddle.to_tensor(1.) / (n1[b] * n2[b])

    v0 = paddle.reshape(paddle.transpose(x0, perm=(0, 2, 1)), (batch_num, n1n2, 1))

    return batch_num, n1, n2, n1max, n2max, n1n2, v0



#############################################
#              Utils Functions              #
#############################################


def inner_prod_aff_fn(feat1, feat2):
    """
    Paddle implementation of inner product affinity function
    """
    return paddle.matmul(feat1, feat2.transpose((0, 2, 1)))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    Paddle implementation of Gaussian affinity function
    """
    feat1 = feat1.unsqueeze(2)
    feat2 = feat2.unsqueeze(1)
    return paddle.exp(-((feat1 - feat2) ** 2).sum(axis=-1) / sigma)


def build_batch(input, return_ori_dim=False):
    """
    Paddle implementation of building a batched tensor
    """
    assert type(input[0]) == paddle.Tensor
    device = input[0].place
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
        pad_pattern = np.zeros((len(max_shape), 2), dtype=np.int64)
        pad_pattern[:, 1] = max_shape - np.array(t.shape)
        padded_ts.append(np.pad(t, pad_pattern, 'constant', constant_values=0))

    if return_ori_dim:
        return paddle.to_tensor(np.stack(padded_ts, axis=0)), tuple([paddle.to_tensor(_, dtype=paddle.int64, place=device) for _ in ori_shape])
    else:
        return paddle.to_tensor(np.stack(padded_ts, axis=0))


def dense_to_sparse(dense_adj):
    """
    Paddle implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn, ori_shape = build_batch([paddle.nonzero(a, as_tuple=False) for a in dense_adj], return_ori_dim=True)
    nedges = ori_shape[0]
    edge_weight = build_batch([dense_adj[b][(conn[b, :, 0], conn[b, :, 1])] for b in range(batch_size)])
    return conn, paddle.unsqueeze(edge_weight, axis=-1), nedges


def compute_affinity_score(X, K):
    """
    Paddle implementation of computing affinity score
    """
    b, n, _ = X.shape
    vx = paddle.reshape(X.transpose((0, 2, 1)),(b, -1, 1)) # (b, n*n, 1)
    vxt = vx.transpose((0, 2, 1))  # (b, 1, n*n)
    affinity = paddle.bmm(paddle.bmm(vxt, K), vx)
    return affinity


def to_numpy(input):
    """
    Paddle function to_numpy
    """
    return input.detach().cpu().numpy()


def from_numpy(input, device):
    """
    Paddle function from_numpy
    """
    if device is None:
        return paddle.to_tensor(input)
    else:
        return paddle.to_tensor(input, place=device)


def generate_isomorphic_graphs(node_num, graph_num, node_feat_dim):
    """
    Paddle implementation of generate_isomorphic_graphs
    """
    X_gt = paddle.zeros((graph_num, node_num, node_num))
    X_gt[0, paddle.arange(0, node_num, dtype=paddle.int64), paddle.arange(0, node_num, dtype=paddle.int64)] = 1
    for i in range(graph_num):
        if i > 0:
            X_gt[i, paddle.arange(0, node_num, dtype=paddle.int64), paddle.randperm(node_num)] = 1
    joint_X = paddle.reshape(X_gt, (graph_num * node_num, node_num))
    X_gt = paddle.mm(joint_X, paddle.t(joint_X))
    X_gt = paddle.transpose(paddle.reshape(X_gt, (graph_num, node_num, graph_num, node_num)), perm=(0, 2, 1, 3))
    A0 = paddle.rand((node_num, node_num))
    paddle.diagonal(A0)[:] = 0
    As = [A0]
    for i in range(graph_num):
        if i > 0:
            As.append(paddle.mm(paddle.mm(X_gt[i, 0], A0), X_gt[0, i]))
    if node_feat_dim > 0:
        F0 = paddle.rand((node_num, node_feat_dim))
        Fs = [F0]
        for i in range(graph_num):
            if i > 0:
                Fs.append(paddle.mm(X_gt[i, 0], F0))
        return paddle.stack(As, axis=0), X_gt, paddle.stack(Fs, axis=0)
    else:
        return paddle.stack(As, axis=0), X_gt


def _aff_mat_from_node_edge_aff(node_aff: paddle.Tensor, edge_aff: paddle.Tensor, connectivity1: paddle.Tensor, connectivity2: paddle.Tensor,
                                n1, n2, ne1, ne2):
    """
    Paddle implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        device = edge_aff.place
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = paddle.to_tensor(np.amax(connectivity1.numpy(), axis=(1, 2)).copy() + 1)
        if n2 is None:
            n2 = paddle.to_tensor(np.amax(connectivity2.numpy(), axis=(1, 2)).copy() + 1)
        if ne1 is None:
            ne1 = [edge_aff.shape[1]] * batch_size
        if ne2 is None:
            ne2 = [edge_aff.shape[1]] * batch_size
    else:
        device = node_aff.place
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
        k = paddle.to_tensor(paddle.zeros((n2max, n1max, n2max, n1max), dtype=dtype), place=device)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:ne1[b]].numpy()
            conn2 = connectivity2[b][:ne2[b]].numpy()
            edge_indices = np.concatenate([conn1.repeat(ne2[b], axis=0), np.tile(conn2, (ne1[b], 1))], axis=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b, :ne1[b], :ne2[b]].reshape([-1])
        k = k.reshape((n2max * n1max, n2max * n1max))
        # node-wise affinity
        if node_aff is not None:
            k[np.arange(n2max * n1max), np.arange(n2max * n1max)] = node_aff[b].transpose((1, 0)).reshape([-1])
        ks.append(k)
    return paddle.stack(ks, axis=0)


def _check_data_type(input: paddle.Tensor):
    """
    Paddle implementation of _check_data_type
    """
    if type(input) is not paddle.Tensor:
        raise ValueError(f'Expected Paddle Tensor, but got {type(input)}. Perhaps the wrong backend?')


def _check_shape(input, dim_num):
    """
    Paddle implementation of _check_shape
    """
    return len(input.shape) == dim_num


def _get_shape(input):
    """
    Paddle implementation of _get_shape
    """
    return input.shape


def _squeeze(input, dim):
    """
    Paddle implementation of _squeeze
    """
    return paddle.squeeze(input, axis=dim)


def _unsqueeze(input, dim):
    """
    Paddle implementation of _unsqueeze
    """
    return paddle.unsqueeze(input, axis=dim)


def _transpose(input, dim1, dim2):
    """
    Paddle implementaiton of _transpose
    """
    return paddle.transpose(input, (dim2, dim1))


def _mm(input1, input2):
    """
    Paddle implementation of _mm
    """
    return paddle.mm(input1, input2)
