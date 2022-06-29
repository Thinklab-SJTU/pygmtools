import itertools
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
        s = paddle.transpose(s, perm=(0, 2, 1))
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
        s = paddle.cat((s, paddle.to_tensor(paddle.full(dummy_shape, -float('inf')), place = s.place)), dim=1)
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
                log_s[paddle.isnan(log_s)] = -float('inf')
            else:
                log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - log_sum
                log_s[paddle.isnan(log_s)] = -float('inf')

        if dummy_row and dummy_shape[1] > 0:
            log_s = log_s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

        if transposed:
            log_s = paddle.transpose(log_s, perm=(0, 2, 1))
            log_s = log_s.transpose(1, 2)

        return paddle.exp(log_s)
    else:
        ret_log_s = paddle.to_tensor(paddle.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), dtype=s.dtype), place=s.place)

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

        if transposed:
            ret_log_s = paddle.transpose(ret_log_s, perm=(0, 2, 1))

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
        s = paddle.reshape(v, (batch_num, n2max, n1max)).transpose([0, 2, 1])
        s = beta * s / s.max(axis=1, keepdim=True).max(axis=2, keepdim=True)
        v = paddle.reshape(alpha * sinkhorn(s, n1, n2, max_iter=sk_iter).transpose([0, 2, 1]),(batch_num, n1n2, 1)) + \
            (1 - alpha) * v
        n = paddle.norm(v, p=1, axis=1, keepdim=True)
        v = paddle.matmul(v, 1 / n)

        if paddle.norm(v - last_v) < 1e-5:
            break

    return v.view(batch_num, n2max, n1max).transpose(1, 2)


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
        v = paddle.reshape(paddle.matmul(v, (1 / n)), (batch_num, 1, 1))
        if paddle.norm(v - vlast) < 1e-5:
            break
        vlast = v

    x = paddle.reshape(v, (batch_num, n2max, n1max)).transpose((0, 2, 1))
    return x


def ipfp(K: paddle.Tensor, n1: paddle.Tensor, n2: paddle.Tensor, n1max, n2max, x0: paddle.Tensor,
         max_iter) -> paddle.Tensor:
    """
    Pytorch implementation of IPFP algorithm
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
        alpha = comp_obj_score(v, K, binary_v - v)  # + torch.mm(k_diag.view(1, -1), (binary_sol - v).view(-1, 1))
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