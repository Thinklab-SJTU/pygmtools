from multiprocessing import Pool
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import stop_gradient

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
        assert log_s.shape[2] >= log_s.shape[1]
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
                index, m = mindspore.ops.max(log_s, axis=2, keep_dims=True)
                log_sum = mindspore.ops.logsumexp(log_s - m, 2, keep_dims=True) + m
                log_s = log_s - mindspore.numpy.where(row_mask, log_sum, mindspore.numpy.zeros_like(log_sum))
                assert not mindspore.ops.isnan(log_s).any()
            else:
                index, m = mindspore.ops.max(log_s, axis=1, keep_dims=True)
                log_sum = mindspore.ops.logsumexp(log_s - m, 1, keep_dims=True) + m
                log_s = log_s - mindspore.numpy.where(col_mask, log_sum, mindspore.numpy.zeros_like(log_sum))
                assert not mindspore.ops.isnan(log_s).any()

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
                    index, m = mindspore.ops.max(log_s_b, axis=1, keep_dims=True)
                    log_sum = mindspore.ops.logsumexp(log_s_b - m, 1, keep_dims=True) + m
                    log_s_b = log_s_b - mindspore.numpy.where(row_mask_b, log_sum, mindspore.numpy.zeros_like(log_sum))
                else:
                    index, m = mindspore.ops.max(log_s_b, axis=0, keep_dims=True)
                    log_sum = mindspore.ops.logsumexp(log_s_b - m, 0, keep_dims=True) + m
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
        v = mindspore.ops.BatchMatMul(K, v)
        last_v = v
        n = mindspore.ops.norm(v, axis=1, p=1, keep_dims=True)
        v = v / n

        # reweighted jump
        s = v.view(batch_num, n2max, n1max).transpose(1, 2)
        s = beta * s / s.max(dim=1, keepdim=True).values.max(dim=2, keepdim=True).values
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter).transpose(1, 2).reshape(batch_num, n1n2, 1) + \
            (1 - alpha) * v
        n = mindspore.ops.norm(v, axis=1, p=1, keep_dims=True)
        v = mindspore.ops.matmul(v, 1 / n)

        if mindspore.ops.norm(v - last_v, axis=None) < 1e-5:
            break

    return v.view(batch_num, n2max, n1max).transpose(1, 2)


def sm(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
       max_iter: int) -> mindspore.Tensor:
    """
    mindspore implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for i in range(max_iter):
        v = mindspore.ops.BatchMatMul(K, v)
        n = mindspore.ops.norm(v, axis=1, p=2)
        v = mindspore.ops.matmul(v, (1 / n).view(batch_num, 1, 1))
        if mindspore.ops.norm(v - vlast, axis=None) < 1e-5:
            break
        vlast = v

    x = v.view(batch_num, n2max, n1max).swapaxes(1, 2)
    return x


def ipfp(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
         max_iter) -> mindspore.Tensor:
    """
    mindspore implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v

    def comp_obj_score(v1, K, v2):
        return mindspore.ops.BatchMatMul(mindspore.ops.BatchMatMul(v1.view(batch_num, 1, -1), K), v2)

    for i in range(max_iter):
        cost = mindspore.ops.BatchMatMul(K, v).reshape(batch_num, n2max, n1max).transpose(1, 2)
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.swapaxes(1, 2).view(batch_num, -1, 1)
        alpha = comp_obj_score(v, K, binary_v - v)  # + torch.mm(k_diag.view(1, -1), (binary_sol - v).view(-1, 1))
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = alpha / beta
        v = mindspore.numpy.where(mindspore.ops.logical_or(beta <= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_sol = comp_obj_score(last_v, K, last_v)
        if mindspore.ops.max(mindspore.ops.abs(
                last_v_sol - mindspore.ops.BatchMatMul(cost.reshape(batch_num, 1, -1),
                                                       binary_sol.reshape(batch_num, -1, 1))
        ) / last_v_sol)[1] < 1e-3:
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
        n1 = mindspore.numpy.full((batch_num,), n1max, dtype=mindspore.int_)
    if n2 is None:
        n2 = mindspore.numpy.full((batch_num,), n2max, dtype=mindspore.int_)
    if n1max is None:
        n1max = mindspore.ops.max(n1)[1]
    if n2max is None:
        n2max = mindspore.ops.max(n2)[1]

    assert n1max * n2max == n1n2, 'the input size of K does not match with n1max * n2max!'

    # initialize x0 (also v0)
    if x0 is None:
        x0 = mindspore.numpy.zeros((batch_num, n1max, n2max), dtype=K.dtype)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = mindspore.Tensor(1.) / (n1[b] * n2[b])
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


#############################################
#              Utils Functions              #
#############################################

def build_batch(input, return_ori_dim=False):
    print('mindspore:')
    print(input)
    """
    mindspore implementation of building a batched tensor
    """
    assert type(input[0]) == mindspore.Tensor
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
        pad_pattern = tuple(pad_pattern.tolist())
        pad_pattern = tuple(
            list((pad_pattern[2 * i], pad_pattern[2 * i + 1]) for i in range(int(len(pad_pattern) / 2))))
        # print(type(pad_pattern))
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
    return conn, edge_weight.unsqueeze(-1), nedges


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
    # if device is None:
    #     return torch.from_numpy(input)
    # else:
    #     return torch.from_numpy(input).to(device)


def _check_data_type(input: mindspore.Tensor, var_name=None):
    """
    mindspore implementation of _check_data_type
    """
    if type(input) is not mindspore.Tensor:
        raise ValueError(f'Expected Pytorch Tensor{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}. Perhaps the wrong backend?')


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
