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
            ret_log_s = ret_log_s.transpose(1, 2)

        return paddle.exp(ret_log_s)

