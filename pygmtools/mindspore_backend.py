from multiprocessing import Pool
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import stop_gradient
import math

import pygmtools
import inspect
import functools
_max_signature = inspect.signature(mindspore.ops.max)
if 'keep_dims' in _max_signature.parameters:
    def _ms_max(*args, keep_dims=False, **kwargs):
        return mindspore.ops.max(*args, keep_dims=keep_dims, **kwargs)
elif 'keepdims' in _max_signature.parameters:
    def _ms_max(*args, keep_dims=False, **kwargs):
        max, indices = mindspore.ops.max(*args, keepdims=keep_dims, **kwargs)
        return indices, max
else:
    raise ValueError('Mindspore function mindspore.ops.max has unsupported signature. It is likely you are working with '
                     'a new Mindspore version which breaks backward compatibility. Please report your Mindspore version '
                     'to GitHub issues.')

_logsumexp_signature = inspect.signature(mindspore.ops.logsumexp)
if 'keep_dims' in _logsumexp_signature.parameters:
    _ms_logsumexp_keepdim = functools.partial(mindspore.ops.logsumexp, keep_dims=True)
elif 'keepdim' in _logsumexp_signature.parameters:
    _ms_logsumexp_keepdim = functools.partial(mindspore.ops.logsumexp, keepdim=True)
else:
    raise ValueError('Mindspore function mindspore.ops.logsumexp has unsupported signature. It is likely you are '
                     'working with a new Mindspore version which breaks backward compatibility. Please report your '
                     'Mindspore version to GitHub issues.')

_norm_signature = inspect.signature(mindspore.ops.norm)
if 'axis' in _norm_signature.parameters and 'p' in _norm_signature.parameters and 'keep_dims' in _norm_signature.parameters:
    def _ms_norm(*args, p=None, axis=None, keep_dims=False, **kwargs):
        return mindspore.ops.norm(*args, p=p, axis=axis, keep_dims=keep_dims, **kwargs)
elif 'dim' in _norm_signature.parameters and 'ord' in _norm_signature.parameters and 'keepdim' in _norm_signature.parameters:
    def _ms_norm(*args, p=None, axis=None, keep_dims=False, **kwargs):
        return mindspore.ops.norm(*args, ord=p, dim=axis, keepdim=keep_dims, **kwargs)
else:
    raise ValueError('Mindspore function mindspore.ops.norm has unsupported signature. It is likely you are '
                     'working with a new Mindspore version which breaks backward compatibility. Please report your '
                     'Mindspore version to GitHub issues.')


#############################################
#     Linear Assignment Problem Solvers     #
#############################################

from pygmtools.numpy_backend import _hung_kernel


def hungarian(s: mindspore.Tensor, n1: mindspore.Tensor = None, n2: mindspore.Tensor = None,
              unmatch1: mindspore.Tensor = None, unmatch2: mindspore.Tensor = None,
              nproc: int = 1) -> mindspore.Tensor:
    """
    mindspore implementation of Hungarian algorithm
    """
    # device = s.device
    batch_num = s.shape[0]

    perm_mat = stop_gradient(s).asnumpy() * -1
    if n1 is not None:
        n1 = n1.asnumpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.asnumpy()
    else:
        n2 = [None] * batch_num
    if unmatch1 is not None:
        unmatch1 = -unmatch1.asnumpy()
    else:
        unmatch1 = [None] * batch_num
    if unmatch2 is not None:
        unmatch2 = -unmatch2.asnumpy()
    else:
        unmatch2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2, unmatch1, unmatch2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack(
            [_hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) for b in range(batch_num)])

    perm_mat = mindspore.Tensor(perm_mat)

    return perm_mat


def sinkhorn(s: mindspore.Tensor, nrows: mindspore.Tensor = None, ncols: mindspore.Tensor = None,
             unmatchrows: mindspore.Tensor = None, unmatchcols: mindspore.Tensor = None,
             dummy_row: bool = False, max_iter: int = 10, tau: float = 1.,
             batched_operation: bool = False) -> mindspore.Tensor:
    """
    mindspore implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.swapaxes(1, 2)
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True

    if nrows is None:
        nrows = mindspore.Tensor([s.shape[1] for _ in range(batch_size)])
    if ncols is None:
        ncols = mindspore.Tensor([s.shape[2] for _ in range(batch_size)])

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if transposed_batch.any():
        s_t = s.swapaxes(1, 2)
        s_t = mindspore.ops.concat((
            s_t[:, :s.shape[1], :],
            mindspore.numpy.full((batch_size, s.shape[1], s.shape[2] - s.shape[1]), -float('inf'))),
            axis=2)
        s = mindspore.numpy.where(transposed_batch.view(batch_size, 1, 1), s_t, s)

        new_nrows = mindspore.numpy.where(transposed_batch, ncols, nrows)
        new_ncols = mindspore.numpy.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = mindspore.ops.concat((
                unmatchrows,
                mindspore.numpy.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]),
                                     -float('inf'))),
                axis=1)
            new_unmatchrows = mindspore.numpy.where(transposed_batch.view(batch_size, 1), unmatchcols, unmatchrows_pad)[
                              :,
                              :unmatchrows.shape[1]]
            new_unmatchcols = mindspore.numpy.where(transposed_batch.view(batch_size, 1), unmatchrows_pad, unmatchcols)
            unmatchrows = new_unmatchrows
            unmatchcols = new_unmatchcols

    # operations are performed on log_s
    log_s = s / tau
    if unmatchrows is not None and unmatchcols is not None:
        unmatchrows = unmatchrows / tau
        unmatchcols = unmatchcols / tau

    if dummy_row:
        if not log_s.shape[2] >= log_s.shape[1]:
            raise RuntimeError('Error in Sinkhorn with dummy row')
        dummy_shape = list(log_s.shape)
        dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
        ori_nrows = nrows
        nrows = ncols.copy()
        log_s = mindspore.ops.concat((log_s, mindspore.numpy.full(dummy_shape, -float('inf'), dtype=log_s.dtype)),
                                     axis=1)
        if unmatchrows is not None:
            unmatchrows = mindspore.ops.concat((unmatchrows,
                                                mindspore.numpy.full((dummy_shape[0], dummy_shape[1]),
                                                                     -float('inf'), dtype=log_s.dtype
                                                                     )), axis=1)
        for b in range(batch_size):
            if int(nrows[b]) > int(ori_nrows[b]):
                log_s[b, int(ori_nrows[b]):int(nrows[b]), :int(ncols[b])] = -100

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = mindspore.numpy.full((log_s.shape[0], log_s.shape[1] + 1, log_s.shape[2] + 1),
                                         -float('inf'), dtype=log_s.dtype
                                         )
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        for b in range(batch_size):
            r, c = int(nrows[b]), int(ncols[b])
            log_s[b, 0:r, c] = unmatchrows[b, 0:r]
            log_s[b, r, 0:c] = unmatchcols[b, 0:c]
    row_mask = mindspore.numpy.zeros((batch_size, log_s.shape[1], 1), dtype=mindspore.bool_)
    col_mask = mindspore.numpy.zeros((batch_size, 1, log_s.shape[2]), dtype=mindspore.bool_)
    for b in range(batch_size):
        r, c = int(nrows[b]), int(ncols[b])
        row_mask[b, 0:r, 0] = 1
        col_mask[b, 0, 0:c] = 1
    if unmatchrows is not None and unmatchcols is not None:
        ncols += 1
        nrows += 1

    if batched_operation:
        for b in range(batch_size):
            log_s[b, int(nrows[b]):, :] = -float('inf')
            log_s[b, :, int(ncols[b]):] = -float('inf')

        for i in range(max_iter):
            if i % 2 == 0:
                index, m = _ms_max(log_s, axis=2, keep_dims=True)
                log_sum = _ms_logsumexp_keepdim(log_s - m, 2) + m
                log_s = log_s - mindspore.numpy.where(row_mask, log_sum, mindspore.numpy.zeros_like(log_sum))
                if mindspore.ops.isnan(log_s).any():
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
            else:
                index, m = _ms_max(log_s, axis=1, keep_dims=True)
                log_sum = _ms_logsumexp_keepdim(log_s - m, 1) + m
                log_s = log_s - mindspore.numpy.where(col_mask, log_sum, mindspore.numpy.zeros_like(log_sum))
                if mindspore.ops.isnan(log_s).any():
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

        ret_log_s = log_s
    else:
        ret_log_s = mindspore.numpy.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), dtype=log_s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]))
            col_slice = slice(0, int(ncols[b]))
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    index, m = _ms_max(log_s_b, axis=1, keep_dims=True)
                    log_sum = _ms_logsumexp_keepdim(log_s_b - m, 1) + m
                    log_s_b = log_s_b - mindspore.numpy.where(row_mask_b, log_sum, mindspore.numpy.zeros_like(log_sum))
                else:
                    index, m = _ms_max(log_s_b, axis=0, keep_dims=True)
                    log_sum = _ms_logsumexp_keepdim(log_s_b - m, 0) + m
                    log_s_b = log_s_b - mindspore.numpy.where(col_mask_b, log_sum, mindspore.numpy.zeros_like(log_sum))

            ret_log_s[b, row_slice, col_slice] = log_s_b

    if unmatchrows is not None and unmatchcols is not None:
        ncols -= 1
        nrows -= 1
        for b in range(batch_size):
            ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
            ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
        ret_log_s = ret_log_s[:, :-1, :-1]

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            if int(nrows[b]) > int(ori_nrows[b]):
                ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

    if transposed_batch.any():
        s_t = ret_log_s.swapaxes(1, 2)
        s_t = mindspore.ops.concat((
            s_t[:, :ret_log_s.shape[1], :],
            mindspore.numpy.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2] - ret_log_s.shape[1]),
                                 -float('inf'), )), axis=2)
        ret_log_s = mindspore.numpy.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)

    if transposed:
        ret_log_s = ret_log_s.swapaxes(1, 2)

    return mindspore.ops.exp(ret_log_s)


#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################


def rrwm(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> mindspore.Tensor:
    """
    mindspore implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
    d = K.sum(axis=2, keepdims=True)
    dmax = d.max(axis=1, keepdims=True)
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = mindspore.ops.BatchMatMul()(K, v)
        last_v = v
        n = _ms_norm(v, axis=1, p=1, keep_dims=True)
        v = v / n

        # reweighted jump
        s = v.view(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
        s = beta * s / s.max(axis=1, keepdims=True).max(axis=2, keepdims=True)
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True).swapaxes(1, 2).reshape(batch_num, n1n2, 1) + \
            (1 - alpha) * v
        n = _ms_norm(v, axis=1, p=1, keep_dims=True)
        v = mindspore.ops.matmul(v, 1 / n)

        if (v - last_v).sum().sqrt() < 1e-5:
            break

    return v.view(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)


def sm(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
       max_iter: int) -> mindspore.Tensor:
    """
    mindspore implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for i in range(max_iter):
        v = mindspore.ops.BatchMatMul()(K, v)
        n = _ms_norm(v, axis=1, p=2)
        v = mindspore.ops.matmul(v, (1 / n).view(batch_num, 1, 1))
        if (v - vlast).sum().sqrt() < 1e-5:
            break
        vlast = v

    x = v.view(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
    return x


def ipfp(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
         max_iter) -> mindspore.Tensor:
    """
    mindspore implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v
    best_v = v
    best_obj = -1

    def comp_obj_score(v1, K, v2):
        return mindspore.ops.BatchMatMul()(mindspore.ops.BatchMatMul()(v1.view(batch_num, 1, -1), K), v2)

    for i in range(max_iter):
        cost = mindspore.ops.BatchMatMul()(K, v).reshape(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.swapaxes(1, 2).view(batch_num, -1, 1)
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = - alpha / beta
        v = mindspore.numpy.where(mindspore.ops.logical_or(beta >= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_obj = comp_obj_score(last_v, K, last_v)

        current_obj = comp_obj_score(binary_v, K, binary_v)
        best_v = mindspore.numpy.where(current_obj > best_obj, binary_v, best_v)
        best_obj = mindspore.numpy.where(current_obj > best_obj, current_obj, best_obj)

        if (_ms_max(mindspore.ops.abs(last_v_obj - current_obj) / last_v_obj)[1] < 1e-3).any():
            break
        last_v = v

    pred_x = best_v.reshape(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
    return pred_x


def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = mindspore.numpy.full((batch_num,), n1max, dtype=mindspore.numpy.int_)
    if n2 is None:
        n2 = mindspore.numpy.full((batch_num,), n2max, dtype=mindspore.numpy.int_)
    if n1max is None:
        n1max = _ms_max(n1)[1]
    if n2max is None:
        n2max = _ms_max(n2)[1]

    if not n1max * n2max == n1n2:
        raise ValueError('the input size of K does not match with n1max * n2max!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = mindspore.numpy.zeros((batch_num, int(n1max), int(n2max)), dtype=K.dtype)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = mindspore.Tensor(1.) / (n1[b] * n2[b])
    v0 = x0.swapaxes(1, 2).reshape((batch_num, n1n2, 1))

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


#############################################
#              Utils Functions              #
#############################################

def inner_prod_aff_fn(feat1, feat2):
    """
    mindspore implementation of inner product affinity function
    """
    return mindspore.ops.matmul(feat1, feat2.swapaxes(1, 2))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    mindspore implementation of Gaussian affinity function
    """
    feat1 = mindspore.ops.expand_dims(feat1, axis=2)
    feat2 = mindspore.ops.expand_dims(feat2, axis=1)
    return mindspore.ops.exp(-((feat1 - feat2) ** 2).sum(axis=-1) / sigma)


def build_batch(input, return_ori_dim=False):
    """
    mindspore implementation of building a batched tensor
    """
    _check_data_type(input[0], 'input', True)
    # device = input[0].device
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
        pad_list = list((pad_pattern[2 * i], pad_pattern[2 * i + 1]) for i in range(int(len(pad_pattern) / 2)))
        while len(pad_list) < t.ndim:
            pad_list.append((0, 0))
        pad_list.reverse()
        pad_pattern = tuple(pad_list)
        mindspore_pad = nn.Pad(pad_pattern, mode="CONSTANT")
        padded_ts.append(mindspore_pad(t))

    if return_ori_dim:
        return mindspore.ops.stack(padded_ts, axis=0), tuple(
            [mindspore.Tensor(_, dtype=mindspore.int64) for _ in ori_shape])
    else:
        return mindspore.ops.stack(padded_ts, axis=0)


def dense_to_sparse(dense_adj):
    """
    mindspore implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn, ori_shape = build_batch([mindspore.ops.nonzero(a) for a in dense_adj], return_ori_dim=True)
    nedges = ori_shape[0]
    edge_weight = build_batch([dense_adj[b][(conn[b, :, 0], conn[b, :, 1])] for b in range(batch_size)])
    return conn, mindspore.ops.expand_dims(edge_weight, axis=-1), nedges


def to_numpy(input):
    """
    mindspore function to_numpy
    """
    return stop_gradient(input).asnumpy()


def from_numpy(input, device=None):
    """
    mindspore function from_numpy
    """
    return mindspore.Tensor(input)


def permutation_loss(pred_dsmat: mindspore.Tensor, gt_perm: mindspore.Tensor, n1: mindspore.Tensor,
                     n2: mindspore.Tensor) -> mindspore.Tensor:
    """
    Pytorch implementation of permutation_loss
    """
    batch_num = pred_dsmat.shape[0]

    pred_dsmat = mindspore.Tensor(pred_dsmat, dtype=mindspore.float32)

    if not mindspore.ops.logical_and(pred_dsmat >= 0, pred_dsmat <= 1).all:
        raise ValueError("pred_dsmat contains invalid numerical entries.")
    if not mindspore.ops.logical_and(gt_perm >= 0, gt_perm <= 1).all:
        raise ValueError("gt_perm contains invalid numerical entries.")

    if n1 is None:
        n1 = mindspore.Tensor([pred_dsmat.shape[1] for _ in range(batch_num)])
    if n2 is None:
        n2 = mindspore.Tensor([pred_dsmat.shape[2] for _ in range(batch_num)])

    loss = mindspore.Tensor(0.)
    n_sum = mindspore.ops.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(n1[b]), slice(n2[b])]
        weight = mindspore.ops.ones_like(pred_dsmat[batch_slice])
        loss += mindspore.ops.BinaryCrossEntropy(reduction='sum')(
            pred_dsmat[batch_slice],
            gt_perm[batch_slice], weight)
        n1_b = mindspore.Tensor(n1[b], dtype=n_sum.dtype)
        n_sum += n1_b

    return loss / n_sum


def _aff_mat_from_node_edge_aff(node_aff: mindspore.Tensor, edge_aff: mindspore.Tensor, connectivity1: mindspore.Tensor,
                                connectivity2: mindspore.Tensor,
                                n1, n2, ne1, ne2):
    """
    mindspore implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        # device = edge_aff.device
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = mindspore.Tensor([math.sqrt(connectivity1.shape[1])] * batch_size)
        if n2 is None:
            n2 = mindspore.Tensor([math.sqrt(connectivity2.shape[1])] * batch_size)
        if ne1 is None:
            ne1 = [edge_aff.shape[1]] * batch_size
        if ne2 is None:
            ne2 = [edge_aff.shape[2]] * batch_size
    else:
        # device = node_aff.device
        dtype = node_aff.dtype
        batch_size = node_aff.shape[0]
        if n1 is None:
            n1 = [node_aff.shape[1]] * batch_size
        if n2 is None:
            n2 = [node_aff.shape[2]] * batch_size

    n1max = int(max(n1))
    n2max = int(max(n2))
    ks = []
    for b in range(batch_size):
        k = mindspore.numpy.zeros((n2max, n1max, n2max, n1max), dtype=dtype)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:int(ne1[b])]
            conn2 = connectivity2[b][:int(ne2[b])]
            edge_indices = mindspore.ops.concat(
                [mindspore.ops.repeat_elements(conn1, int(ne2[b]), axis=0),
                 mindspore.numpy.tile(conn2, (int(ne1[b]), 1))],
                axis=1)  # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3],
                            edge_indices[:, 1])  # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b, :int(ne1[b]), :int(ne2[b])].reshape(-1)
        k = k.reshape((n2max * n1max, n2max * n1max))
        # node-wise affinity
        if node_aff is not None:
            k[mindspore.numpy.arange(n2max * n1max), mindspore.numpy.arange(n2max * n1max)] = node_aff[b].transpose(1,
                                                                                                                    0).reshape(
                -1)
            # k_diag = mindspore.numpy.diagonal(k)
            # k_diag[:] = node_aff[b].transpose(0, 1).reshape(-1)
        ks.append(k)

    return mindspore.ops.stack(ks, axis=0)


def _check_data_type(input: mindspore.Tensor, var_name, raise_err):
    """
    mindspore implementation of _check_data_type
    """
    ms_types = [mindspore.Tensor]
    if hasattr(mindspore.common, '_stub_tensor'):  # MS tensor may be automatically transformed to StubTensor
        ms_types += [mindspore.common._stub_tensor.StubTensor]
    is_tensor = any([type(input) is t for t in ms_types])

    if raise_err and not is_tensor:
        raise ValueError(f'Expected MindSpore Tensor{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}.')
    return is_tensor


def _check_shape(input, dim_num):
    """
    mindspore implementation of _check_shape
    """
    return len(input.shape) == dim_num


def _get_shape(input):
    """
    mindspore implementation of _get_shape
    """
    return input.shape


def _squeeze(input, dim):
    """
    mindspore implementation of _squeeze
    """
    return mindspore.ops.squeeze(input, axis=dim)


def _unsqueeze(input, dim):
    """
    mindspore implementation of _unsqueeze
    """
    return mindspore.ops.expand_dims(input, axis=dim)


def _transpose(input, dim1, dim2):
    """
    mindspore implementaiton of _transpose
    """
    return input.swapaxes(dim1, dim2)


def _mm(input1, input2):
    """
    mindspore implementation of _mm
    """
    return mindspore.ops.matmul(input1, input2)


############################################
#          Neural Network Solvers          #
############################################

from pygmtools.mindspore_modules import *


class PCA_GM_Net(nn.Cell):
    """
    MindSpore implementation of PCA-GM and IPCA-GM network.
    """

    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, cross_iter_num=-1):
        super().__init__()
        self.gnn_layer = num_layers
        self.gnn_layer_list = nn.CellList()
        self.affinity_list = nn.CellList()
        self.cross_graph_list = nn.CellList()
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(in_channel, hidden_channel)
            elif 0 < i < self.gnn_layer - 1:
                gnn_layer = Siamese_Gconv(hidden_channel, hidden_channel)
            else:
                gnn_layer = Siamese_Gconv(hidden_channel, out_channel)
            self.gnn_layer_list.append(gnn_layer)

            if i == self.gnn_layer - 1:
                self.affinity_list.append(WeightedInnerProdAffinity(out_channel))
            elif i == self.gnn_layer - 2 and cross_iter_num <= 0:
                self.affinity_list.append(WeightedInnerProdAffinity(hidden_channel))
            else:
                self.affinity_list.append(Identity())

            if i == self.gnn_layer - 2:
                self.cross_graph_list.append(nn.Dense(hidden_channel * 2, hidden_channel))
            else:
                self.cross_graph_list.append(Identity())

    def construct(self, feat1, feat2, A1, A2, n1, n2, cross_iter_num, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(
            sinkhorn, dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False
        )
        emb1, emb2 = feat1, feat2
        if cross_iter_num <= 0:
            for i in range(self.gnn_layer):
                emb1, emb2 = self.gnn_layer_list[i]([A1, emb1], [A2, emb2])
                if i == self.gnn_layer - 2:
                    s = self.affinity_list[i](emb1, emb2)
                    s = _sinkhorn_func(s, n1, n2)
                    cross_graph = self.cross_graph_list[i]
                    new_emb1 = cross_graph(mindspore.ops.concat((emb1, mindspore.ops.BatchMatMul()(s, emb2)), axis=-1))
                    new_emb2 = cross_graph(
                        mindspore.ops.concat((emb2, mindspore.ops.BatchMatMul()(s.swapaxes(1, 2), emb1)), axis=-1)
                    )
                    emb1, emb2 = new_emb1, new_emb2

            s = self.affinity_list[self.gnn_layer - 1](emb1, emb2)
            s = _sinkhorn_func(s, n1, n2)
        else:
            for i in range(self.gnn_layer - 1):
                emb1, emb2 = self.gnn_layer_list[i]([A1, emb1], [A2, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = mindspore.ops.zeros((emb1.shape[0], emb1.shape[1], emb2.shape[1]), emb1.dtype)
            for _ in range(cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = self.cross_graph_list[i]
                emb1 = cross_graph(mindspore.ops.concat((emb1_0, mindspore.ops.BatchMatMul()(s, emb2_0)), axis=-1))
                emb2 = cross_graph(
                    mindspore.ops.concat((emb2_0, mindspore.ops.BatchMatMul()(s.swapaxes(1, 2), emb1_0)), axis=-1)
                )
                i = self.gnn_layer - 1
                emb1, emb2 = self.gnn_layer_list[i]([A1, emb1], [A2, emb2])
                s = self.affinity_list[i](emb1, emb2)
                s = _sinkhorn_func(s, n1, n2)
        return s


class CIE_Net(nn.Cell):
    """
    MindSpore implementation of CIE graph matching network.
    """

    def __init__(self, in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers):
        super().__init__()
        self.gnn_layer = num_layers
        self.gnn_layer_list = nn.CellList()
        self.affinity_list = nn.CellList()
        self.cross_graph_list = nn.CellList()
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_ChannelIndependentConv(in_node_channel, hidden_channel, in_edge_channel)
            elif 0 < i < self.gnn_layer - 1:
                gnn_layer = Siamese_ChannelIndependentConv(hidden_channel, hidden_channel, hidden_channel)
            else:
                gnn_layer = Siamese_ChannelIndependentConv(hidden_channel, out_channel, hidden_channel)
            self.gnn_layer_list.append(gnn_layer)

            if i == self.gnn_layer - 1:
                self.affinity_list.append(WeightedInnerProdAffinity(out_channel))
            elif i == self.gnn_layer - 2:
                self.affinity_list.append(WeightedInnerProdAffinity(hidden_channel))
            else:
                self.affinity_list.append(Identity())

            if i == self.gnn_layer - 2:
                self.cross_graph_list.append(nn.Dense(hidden_channel * 2, hidden_channel))
            else:
                self.cross_graph_list.append(Identity())

    def construct(self, feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(
            sinkhorn, dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False
        )
        emb1, emb2 = feat_node1, feat_node2
        emb_edge1, emb_edge2 = feat_edge1, feat_edge2
        for i in range(self.gnn_layer):
            emb1, emb2, emb_edge1, emb_edge2 = self.gnn_layer_list[i]([A1, emb1, emb_edge1], [A2, emb2, emb_edge2])
            if i == self.gnn_layer - 2:
                s = self.affinity_list[i](emb1, emb2)
                s = _sinkhorn_func(s, n1, n2)
                cross_graph = self.cross_graph_list[i]
                new_emb1 = cross_graph(mindspore.ops.concat((emb1, mindspore.ops.BatchMatMul()(s, emb2)), axis=-1))
                new_emb2 = cross_graph(
                    mindspore.ops.concat((emb2, mindspore.ops.BatchMatMul()(s.swapaxes(1, 2), emb1)), axis=-1)
                )
                emb1, emb2 = new_emb1, new_emb2

        s = self.affinity_list[self.gnn_layer - 1](emb1, emb2)
        s = _sinkhorn_func(s, n1, n2)
        return s


class NGM_Net(nn.Cell):
    """
    MindSpore implementation of NGM network.
    """

    def __init__(self, gnn_channels, sk_emb):
        super().__init__()
        self.gnn_layer = len(gnn_channels)
        self.gnn_layer_list = nn.CellList()
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = NGMConvLayer(1, 1, gnn_channels[i] + sk_emb, gnn_channels[i], sk_channel=sk_emb)
            else:
                gnn_layer = NGMConvLayer(
                    gnn_channels[i - 1] + sk_emb, gnn_channels[i - 1],
                    gnn_channels[i] + sk_emb, gnn_channels[i], sk_channel=sk_emb
                )
            self.gnn_layer_list.append(gnn_layer)
        self.classifier = nn.Dense(gnn_channels[-1] + sk_emb, 1)

    def construct(self, K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(
            sinkhorn, dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False
        )
        emb = v0
        A = (K != 0).astype(K.dtype)
        emb_K = mindspore.ops.expand_dims(K, axis=-1)
        for i in range(self.gnn_layer):
            emb_K, emb = self.gnn_layer_list[i](A, emb_K, emb, n1, n2, sk_func=_sinkhorn_func)
        v = self.classifier(emb)
        s = v.reshape((v.shape[0], int(n2max), -1)).swapaxes(1, 2)
        return _sinkhorn_func(s, n1, n2, dummy_row=True)


pca_gm_pretrain_path = {
    'voc': ('pca_gm_voc_mindspore.ckpt',
            ['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc_mindspore.ckpt'],
            'e49379424ffea6759526bd9d436a0dcf'),
    'willow': ('pca_gm_willow_mindspore.ckpt',
               ['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_willow_mindspore.ckpt'],
               '3ed733a9b04e2a83142b08db2b9952cf'),
    'voc-all': ('pca_gm_voc-all_mindspore.ckpt',
                ['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc-all_mindspore.ckpt'],
                '28d40ccdc8bc743d2eca459758d887a2'),
}

ipca_gm_pretrain_path = {
    'voc': ('ipca_gm_voc_mindspore.ckpt',
            ['https://huggingface.co/heatingma/pygmtools/resolve/main/ipca_gm_voc_mindspore.ckpt'],
            'c9f888eefbc22684317f5deedf175da7'),
    'willow': ('ipca_gm_willow_mindspore.ckpt',
               ['https://huggingface.co/heatingma/pygmtools/resolve/main/ipca_gm_willow_mindspore.ckpt'],
               '8d25c7bc7d7350467e07e53d1d004d63'),
}

cie_pretrain_path = {
    'voc': ('cie_voc_mindspore.ckpt',
            ['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_voc_mindspore.ckpt'],
            'be80f98e26af89a68421286f60d544f7'),
    'willow': ('cie_willow_mindspore.ckpt',
               ['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_willow_mindspore.ckpt'],
               'b295e0bfa7367e9a2830b0ec25a99220'),
}

ngm_pretrain_path = {
    'voc': ('ngm_voc_mindspore.ckpt',
            ['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_voc_mindspore.ckpt'],
            'afa3d94ac9685dba82629e9ef79b19cf'),
    'willow': ('ngm_willow_mindspore.ckpt',
               ['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_willow_mindspore.ckpt'],
               '877aad75a62ad6cddbd45a0f4ece3790'),
}


def _save_model(model, path):
    mindspore.save_checkpoint(model, path)


def _load_model(model, path, strict=True):
    param_dict = mindspore.load_checkpoint(path)
    try:
        param_not_load, ckpt_not_load = mindspore.load_param_into_net(model, param_dict, strict_load=strict)
    except TypeError:
        param_not_load, ckpt_not_load = mindspore.load_param_into_net(model, param_dict)
    if len(ckpt_not_load) > 0:
        print('Warning: Unexpected key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in ckpt_not_load)))
    if len(param_not_load) > 0:
        print('Warning: Missing key(s) in state_dict: {}. '.format(
            ', '.join('"{}"'.format(k) for k in param_not_load)))


def _get_pretrain_file(pretrain, pretrain_path):
    if pretrain in pretrain_path:
        filename, url, md5 = pretrain_path[pretrain]
        return pygmtools.utils.download(filename, url, md5)
    raise ValueError(f'Unknown pretrain tag. Available tags: {pretrain_path.keys()}')


def pca_gm(feat1, feat2, A1, A2, n1, n2,
           in_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
           network, pretrain):
    """
    MindSpore implementation of PCA-GM.
    """
    forward_pass = feat1 is not None
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers)
        if pretrain:
            _load_model(network, _get_pretrain_file(pretrain, pca_gm_pretrain_path))
    if forward_pass:
        batch_size = feat1.shape[0]
        if n1 is None:
            n1 = mindspore.Tensor([feat1.shape[1]] * batch_size, dtype=mindspore.int32)
        if n2 is None:
            n2 = mindspore.Tensor([feat2.shape[1]] * batch_size, dtype=mindspore.int32)
        result = network(feat1, feat2, A1, A2, n1, n2, -1, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


def ipca_gm(feat1, feat2, A1, A2, n1, n2,
            in_channel, hidden_channel, out_channel, num_layers, cross_iter, sk_max_iter, sk_tau,
            network, pretrain):
    """
    MindSpore implementation of IPCA-GM.
    """
    forward_pass = feat1 is not None
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers, cross_iter)
        if pretrain:
            _load_model(network, _get_pretrain_file(pretrain, ipca_gm_pretrain_path))
    if forward_pass:
        batch_size = feat1.shape[0]
        if n1 is None:
            n1 = mindspore.Tensor([feat1.shape[1]] * batch_size, dtype=mindspore.int32)
        if n2 is None:
            n2 = mindspore.Tensor([feat2.shape[1]] * batch_size, dtype=mindspore.int32)
        result = network(feat1, feat2, A1, A2, n1, n2, cross_iter, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


def cie(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2,
        in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
        network, pretrain):
    """
    MindSpore implementation of CIE.
    """
    forward_pass = feat_node1 is not None
    if network is None:
        network = CIE_Net(in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers)
        if pretrain:
            _load_model(network, _get_pretrain_file(pretrain, cie_pretrain_path))
    if forward_pass:
        batch_size = feat_node1.shape[0]
        if n1 is None:
            n1 = mindspore.Tensor([feat_node1.shape[1]] * batch_size, dtype=mindspore.int32)
        if n2 is None:
            n2 = mindspore.Tensor([feat_node1.shape[1]] * batch_size, dtype=mindspore.int32)
        result = network(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


def ngm(K, n1, n2, n1max, n2max, x0, gnn_channels, sk_emb, sk_max_iter, sk_tau, network, return_network, pretrain):
    """
    MindSpore implementation of NGM.
    """
    forward_pass = K is not None
    if network is None:
        network = NGM_Net(gnn_channels, sk_emb)
        if pretrain:
            _load_model(network, _get_pretrain_file(pretrain, ngm_pretrain_path))
    if forward_pass:
        batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
        v0 = v0 / mindspore.ops.mean(v0)
        result = network(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network
