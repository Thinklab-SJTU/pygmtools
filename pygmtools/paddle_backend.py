# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import itertools
import functools
import paddle
import numpy as np
from multiprocessing import Pool
import os

import pygmtools.utils
from pygmtools.numpy_backend import _hung_kernel


#############################################
#     Linear Assignment Problem Solvers     #
#############################################

def hungarian(s: paddle.Tensor, n1: paddle.Tensor=None, n2: paddle.Tensor=None,
              unmatch1: paddle.Tensor=None, unmatch2: paddle.Tensor=None,
              nproc: int=1) -> paddle.Tensor:
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
    if unmatch1 is not None:
        unmatch1 = -unmatch1.cpu().numpy()
    else:
        unmatch1 = [None] * batch_num
    if unmatch2 is not None:
        unmatch2 = -unmatch2.cpu().numpy()
    else:
        unmatch2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2, unmatch1, unmatch2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) for b in range(batch_num)])

    perm_mat = paddle.to_tensor(perm_mat, place=device)

    return perm_mat


def sinkhorn(s: paddle.Tensor, nrows: paddle.Tensor=None, ncols: paddle.Tensor=None,
             unmatchrows: paddle.Tensor=None, unmatchcols: paddle.Tensor=None,
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
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True

    if nrows is None:
        nrows = paddle.to_tensor([s.shape[1] for _ in range(batch_size)], place=s.place, dtype=paddle.int32)
    if ncols is None:
        ncols = paddle.to_tensor([s.shape[2] for _ in range(batch_size)], place=s.place, dtype=paddle.int32)


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

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = paddle.concat((
                unmatchrows,
                paddle.to_tensor(paddle.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]), -float('inf'), dtype=unmatchrows.dtype), place=unmatchrows.place)),
            axis=1)
            new_unmatchrows = paddle.where(transposed_batch.reshape((batch_size, 1)), unmatchcols, unmatchrows_pad)[:, :unmatchrows.shape[1]]
            new_unmatchcols = paddle.where(transposed_batch.reshape((batch_size, 1)), unmatchrows_pad, unmatchcols)
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
        nrows = ncols.clone()
        log_s = paddle.concat((log_s, paddle.to_tensor(paddle.full(dummy_shape, -float('inf'), dtype=log_s.dtype), place=log_s.place)), axis=1)
        if unmatchrows is not None:
            unmatchrows = paddle.concat((unmatchrows, paddle.to_tensor(paddle.full((dummy_shape[0], dummy_shape[1]), -float('inf'), dtype=log_s.dtype), place=log_s.place)), axis=1)
        for b in range(batch_size):
            log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = paddle.to_tensor(paddle.full((log_s.shape[0], log_s.shape[1]+1, log_s.shape[2]+1), -float('inf'), dtype=log_s.dtype), place=log_s.place)
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        for b in range(batch_size):
            log_s[b, :nrows[b], ncols[b]] = unmatchrows[b, :nrows[b]]
            log_s[b, nrows[b], :ncols[b]] = unmatchcols[b, :ncols[b]]
    row_mask = paddle.zeros((batch_size, log_s.shape[1], 1), dtype=paddle.bool)
    col_mask = paddle.zeros((batch_size, 1, log_s.shape[2]), dtype=paddle.bool)
    for b in range(batch_size):
        row_mask[b, :nrows[b], 0] = 1
        col_mask[b, 0, :ncols[b]] = 1
    if unmatchrows is not None and unmatchcols is not None:
        ncols += 1
        nrows += 1

    if batched_operation:
        for b in range(batch_size):
            log_s[b, nrows[b]:, :] = -float('inf')
            log_s[b, :, ncols[b]:] = -float('inf')

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = paddle.logsumexp(log_s, 2, keepdim=True)
                log_s = log_s - paddle.where(row_mask, log_sum, paddle.zeros_like(log_sum))
                nan_indices = paddle.nonzero(paddle.isnan(log_s), True)
                if not nan_indices[0].size == 0:
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
            else:
                log_sum = paddle.logsumexp(log_s, 1, keepdim=True)
                log_s = log_s - paddle.where(col_mask, log_sum, paddle.zeros_like(log_sum))
                nan_indices = paddle.nonzero(paddle.isnan(log_s), True)
                if not nan_indices[0].size == 0:
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

        ret_log_s = log_s
    else:
        ret_log_s = paddle.to_tensor(paddle.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf')), place=log_s.place, dtype=log_s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = paddle.logsumexp(log_s_b, 1, keepdim=True)
                    log_s_b = log_s_b - paddle.where(row_mask_b, log_sum, paddle.zeros_like(log_sum))
                else:
                    log_sum = paddle.logsumexp(log_s_b, 0, keepdim=True)
                    log_s_b = log_s_b - paddle.where(col_mask_b, log_sum, paddle.zeros_like(log_sum))

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

    if paddle.any(transposed_batch):
        s_t = ret_log_s.transpose((0, 2, 1))
        s_t = paddle.concat((
            s_t[:, :ret_log_s.shape[1], :],
            paddle.to_tensor(paddle.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2]-ret_log_s.shape[1]), -float('inf')), place=log_s.place)), axis=2)
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
        v = alpha * paddle.reshape(sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True).transpose((0, 2, 1)),(batch_num, n1n2, 1)) + \
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
    best_v = v
    best_obj = paddle.to_tensor(paddle.full((batch_num, 1, 1), -1.), place=K.place)

    def comp_obj_score(v1, K, v2):
        return paddle.bmm(paddle.bmm(paddle.reshape(v1, (batch_num, 1, -1)), K), v2)

    for i in range(max_iter):
        cost = paddle.reshape(paddle.bmm(K, v),(batch_num, n2max, n1max)).transpose((0, 2, 1))
        binary_sol = hungarian(cost, n1, n2)
        binary_v = paddle.reshape(binary_sol.transpose((0, 2, 1)),(batch_num, -1, 1))
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = - alpha / beta
        v = paddle.where(paddle.logical_or(beta >= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_obj = comp_obj_score(last_v, K, last_v)

        current_obj = comp_obj_score(binary_v, K, binary_v)
        best_v = paddle.where(current_obj > best_obj, binary_v, best_v)
        best_obj = paddle.where(current_obj > best_obj, current_obj, best_obj)

        if paddle.max(paddle.abs(last_v_obj - current_obj) / last_v_obj) < 1e-3:
            break
        last_v = v

    pred_x = paddle.reshape(best_v, (batch_num, n2max, n1max)).transpose((0, 2, 1))
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

    if not n1max * n2max == n1n2:
        raise ValueError('the input size of K does not match with n1max * n2max!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = paddle.to_tensor(paddle.zeros((batch_num, n1max, n2max), dtype=K.dtype), place=K.place)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = paddle.to_tensor(1.) / (n1[b] * n2[b])

    v0 = paddle.reshape(paddle.transpose(x0, perm=(0, 2, 1)), (batch_num, n1n2, 1))

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


############################################
#      Multi-Graph Matching Solvers        #
############################################


def cao_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    r"""
    Paddle implementation of CAO solver (mode="c")

    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X: initial matching, (m, m, n, n)
    :param num_graph: number of graphs, int
    :param num_node: number of nodes, int
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    param_lambda = lambda_init
    device = K.place

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='paddle').unsqueeze(-1).unsqueeze(-1)

    for iter in range(max_iter):
        if iter >= iter_boost:
            param_lambda = np.min([param_lambda * lambda_step, lambda_max])
        # pair_con = get_batch_pc_opt(X)
        pair_aff = _comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))).reshape((m, m))
        pair_aff = pair_aff - paddle.to_tensor(paddle.eye(m) , place=device) * pair_aff
        norm = paddle.max(pair_aff)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                con_ori = _get_single_pc_opt(X, i, j)
                if iter < iter_boost:
                    score_ori = aff_ori
                else:
                    score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
                X_upt = X[i, j]
                for k in range(m):
                    X_combo = paddle.matmul(X[i, k], X[k, j])
                    aff_combo = _comp_aff_score(X_combo, K[i, j]) / norm
                    con_combo = _get_single_pc_opt(X, i, j, X_combo)
                    if iter < iter_boost:
                        score_combo = aff_combo
                    else:
                        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda
                    if score_combo > score_ori:
                        X_upt = X_combo
                X[i, j] = X_upt
                X[j, i] = X_upt.transpose((1, 0))
    return X


def cao_fast_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    r"""
    Paddle implementation of CAO solver in fast config (mode="pc")

    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X: initial matching, (m, m, n, n)
    :param num_graph: number of graphs, int
    :param num_node: number of nodes, int
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    param_lambda = lambda_init

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='paddle').unsqueeze(-1).unsqueeze(-1)

    device = K.place
    mask1 = paddle.to_tensor(paddle.arange(m).reshape((m, 1)).tile((1, m)), place = device)
    mask2 = paddle.to_tensor(paddle.arange(m).reshape((1, m)).tile((m, 1)), place = device)
    mask = paddle.to_tensor((mask1 < mask2), dtype = 'float32')
    X_mask = mask.reshape((m, m, 1, 1))

    for iter in range(max_iter):
        if iter >= iter_boost:
            param_lambda = np.min([param_lambda * lambda_step, lambda_max])

        pair_aff = _comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))).reshape((m, m))
        pair_aff = pair_aff - paddle.to_tensor(paddle.eye(m), place=device) * pair_aff
        norm = paddle.max(pair_aff)

        X1 = X.reshape((m, 1, m, n, n)).tile((1, m, 1, 1, 1)).reshape((-1, n, n))  # X1[i,j,k] = X[i,k]
        X2 = X.reshape((1, m, m, n, n)).tile((m, 1, 1, 1, 1)).transpose((0, 2, 1, 3, 4)).reshape((-1, n, n))  # X2[i,j,k] = X[k,j]
        X_combo = paddle.bmm(X1, X2).reshape((m, m, m, n, n)) # X_combo[i,j,k] = X[i, k] * X[k, j]

        aff_ori = (_comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))) / norm).reshape((m, m))
        pair_con = _get_batch_pc_opt(X)
        con_ori = paddle.sqrt(pair_con)

        K_repeat = K.reshape((m, m, 1, n * n, n * n)).tile((1, 1, m, 1, 1)).reshape((-1, n * n, n * n))
        aff_combo = (_comp_aff_score(X_combo.reshape((-1, n, n)), K_repeat) / norm).reshape((m, m, m))
        con1 = pair_con.reshape((m, 1, m)).tile((1, m, 1))  # con1[i,j,k] = pair_con[i,k]
        con2 = pair_con.reshape((1, m, m)).tile((m, 1, 1)).transpose((0, 2, 1))  # con2[i,j,k] = pair_con[j,k]
        con_combo = paddle.sqrt(con1 * con2)

        if iter < iter_boost:
            score_ori = aff_ori
            score_combo = aff_combo
        else:
            score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
            score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

        idx = paddle.argmax(score_combo, axis=-1)
        score_combo = paddle.max(score_combo, axis=-1)

        if not paddle.all(score_combo + 1e-4 >= score_ori):
            raise RuntimeError('CAO-fast internal error', paddle.min(score_combo - score_ori))
        X_upt = X_combo[mask1, mask2, idx]
        X = X_upt * X_mask + X_upt.transpose((1, 0, 3, 2))* X_mask.transpose((1, 0, 2, 3)) + X * (1 - X_mask - X_mask.transpose((1, 0, 2, 3)))
        if not paddle.all(X.transpose((1, 0, 3, 2)) == X):
            raise RuntimeError('CAO-fast internal error')
    return X


def mgm_floyd_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node
    device = K.place

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='paddle').unsqueeze(-1).unsqueeze(-1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))).reshape((m, m))
        pair_aff = pair_aff - paddle.to_tensor(paddle.eye(m), place=device) * pair_aff
        norm = paddle.max(pair_aff)

        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                score_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                X_combo = paddle.matmul(X[i, k], X[k, j])
                score_combo = _comp_aff_score(X_combo, K[i, j]) / norm

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose((1, 0))

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))).reshape((m, m))
        pair_aff = pair_aff - paddle.to_tensor(paddle.eye(m), place=device) * pair_aff
        norm = paddle.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                con_ori = _get_single_pc_opt(X, i, j)
                score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda

                X_combo = paddle.matmul(X[i, k], X[k, j])
                aff_combo = _comp_aff_score(X_combo, K[i, j]) / norm
                con_combo = _get_single_pc_opt(X, i, j, X_combo)
                score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose((1, 0))
    return X


def mgm_floyd_fast_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node
    device = K.place

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='paddle').unsqueeze(-1).unsqueeze(-1)

    mask1 = paddle.arange(m).reshape((m, 1)).tile((1, m))
    mask2 = paddle.arange(m).reshape((1, m)).tile((m, 1))
    mask = paddle.to_tensor(paddle.to_tensor((mask1 < mask2), dtype = 'float32'), place = device)
    X_mask = mask.reshape((m, m, 1, 1))

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))).reshape((m, m))
        pair_aff = pair_aff - paddle.to_tensor(paddle.eye(m), place=device) * pair_aff
        norm = paddle.max(pair_aff)

        X1 = X[:, k].reshape((m, 1, n, n)).tile((1, m, 1, 1)).reshape((-1, n, n))  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape((1, m, n, n)).tile((m, 1, 1, 1)).reshape((-1, n, n))  # X[i, j] = X[j, k]
        X_combo = paddle.bmm(X1, X2).reshape((m, m, n, n))

        aff_ori = (_comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))) / norm).reshape((m, m))
        aff_combo = (_comp_aff_score(X_combo.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))) / norm).reshape((m, m))

        score_ori = aff_ori
        score_combo = aff_combo

        upt = paddle.to_tensor((score_ori < score_combo), dtype = 'float32')
        upt = (upt * mask).reshape((m, m, 1, 1))
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.transpose((1, 0, 2, 3)).transpose((0, 1, 3, 2)) * (1 - X_mask)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))).reshape((m, m))
        pair_aff = pair_aff - paddle.to_tensor(paddle.eye(m), place=device) * pair_aff
        norm = paddle.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)

        X1 = X[:, k].reshape((m, 1, n, n)).tile((1, m, 1, 1)).reshape((-1, n, n))  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape((1, m, n, n)).tile((m, 1, 1, 1)).reshape((-1, n, n))  # X[i, j] = X[j, k]
        X_combo = paddle.bmm(X1, X2).reshape((m, m, n, n))

        aff_ori = (_comp_aff_score(X.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))) / norm).reshape((m, m))
        aff_combo = (_comp_aff_score(X_combo.reshape((-1, n, n)), K.reshape((-1, n * n, n * n))) / norm).reshape((m, m))

        con_ori = paddle.sqrt(pair_con)
        con1 = pair_con[:, k].reshape((m, 1)).tile((1, m))
        con2 = pair_con[k, :].reshape((1, m)).tile((m, 1))
        con_combo = paddle.sqrt(con1 * con2)

        score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

        upt = paddle.to_tensor((score_ori < score_combo), dtype = 'float32')
        upt = (upt * mask).reshape((m, m, 1, 1))
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.transpose((1, 0, 2, 3)).transpose((0, 1, 3, 2)) * (1 - X_mask)
    return X


def _get_single_pc_opt(X, i, j, Xij=None):
    """
    CAO/Floyd helper function (compute consistency)
    :param X: (m, m, n, n) all the matching results
    :param i: index
    :param j: index
    :return: the consistency of X_ij
    """
    m, _, n, _ = X.shape
    if Xij is None:
        Xij = X[i, j]
    X1 = X[i, :].reshape((-1, n, n))
    X2 = X[:, j].reshape((-1, n, n))
    X_combo = paddle.bmm(X1, X2)
    pair_con = 1 - paddle.sum(paddle.abs(Xij - X_combo)) / (2 * n * m)
    return pair_con


def _get_batch_pc_opt(X):
    """
    CAO/Floyd-fast helper function (compute consistency in batch)
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    m, _, n, _ = X.shape
    X1 = X.reshape((m, 1, m, n, n)).tile((1, m, 1, 1, 1)).reshape((-1, n, n))  # X1[i, j, k] = X[i, k]
    X2 = X.reshape((1, m, m, n, n)).tile((m, 1, 1, 1, 1)).transpose((0, 2, 1, 3, 4))
    X2 = paddle.reshape(X2, (-1, n, n))  # X2[i, j, k] = X[k, j]
    X_combo = paddle.bmm(X1, X2).reshape((m, m, m, n, n))
    X_ori = X.reshape((m, m, 1, n, n)).tile((1, 1, m, 1, 1))
    pair_con = 1 - paddle.sum(paddle.abs(X_combo - X_ori), axis=(2, 3, 4)) / (2 * n * m)
    return pair_con


def gamgm(
        A, W, ns, n_univ, U0,
        init_tau, min_tau, sk_gamma,
        sk_iter, max_iter, quad_weight,
        converge_thresh, outlier_thresh, bb_smooth,
        verbose,
        cluster_M=None, projector='sinkhorn', hung_iter=True # these arguments are reserved for clustering
):
    """
    Paddle implementation of Graduated Assignment for Multi-Graph Matching (with compatibility for 2GM and clustering)
    """
    num_graphs = A.shape[0]
    if ns is None:
        ns = paddle.to_tensor(paddle.full((num_graphs,), A.shape[1]), dtype=paddle.int32, place=A.place)
    n_indices = paddle.cumsum(ns, axis=0)

    # build a super adjacency matrix A
    supA = paddle.to_tensor(paddle.zeros((n_indices[-1], n_indices[-1])), place=A.place)
    for i in range(num_graphs):
        start_n = n_indices[i] - ns[i]
        end_n = n_indices[i]
        supA[start_n:end_n, start_n:end_n] = A[i, :ns[i], :ns[i]]

    # handle the type of n_univ
    if type(n_univ) is paddle.Tensor:
        n_univ = n_univ.item()

    # randomly init U
    if U0 is None:
        U0 = paddle.to_tensor(paddle.full((n_indices[-1], n_univ), 1 / n_univ), place=A.place)
        U0 += paddle.randn(U0.shape) / 1000

    # init cluster_M if not given
    if cluster_M is None:
        cluster_M = paddle.to_tensor(paddle.ones((num_graphs, num_graphs)), place=A.place)

    # reshape W into supW
    supW = paddle.to_tensor(paddle.zeros((n_indices[-1], n_indices[-1])), place=A.place)
    for i, j in itertools.product(range(num_graphs), repeat=2):
        start_x = n_indices[i] - ns[i]
        end_x = n_indices[i]
        start_y = n_indices[j] - ns[j]
        end_y = n_indices[j]
        supW[start_x:end_x, start_y:end_y] = W[i, j, :ns[i], :ns[j]]

    U = gamgm_real(
        supA, supW, ns, n_indices, n_univ, num_graphs, U0,
        init_tau, min_tau, sk_gamma,
        sk_iter, max_iter, quad_weight,
        converge_thresh, outlier_thresh,
        verbose,
        cluster_M, projector, hung_iter
    )

    # build MultiMatchingResult
    result = pygmtools.utils.MultiMatchingResult(True, 'paddle')

    for i in range(num_graphs):
        start_n = n_indices[i] - ns[i]
        end_n = n_indices[i]
        result[i] = U[start_n:end_n]

    return result


def gamgm_real(
        supA, supW, ns, n_indices, n_univ, num_graphs, U0,
        init_tau, min_tau, sk_gamma,
        sk_iter, max_iter, quad_weight,
        converge_thresh, outlier_thresh,
        verbose,
        cluster_M, projector, hung_iter # these arguments are reserved for clustering
        ):
    """
    The real forward function of GAMGM
    """
    U = U0
    sinkhorn_tau = init_tau
    iter_flag = True

    while iter_flag:
        for i in range(max_iter):
            # compact matrix form update of V
            UUt = paddle.mm(U, U.t())
            lastUUt = UUt
            cluster_weight = paddle.repeat_interleave(cluster_M, paddle.to_tensor(ns, dtype=paddle.int64), axis=0)
            cluster_weight = paddle.repeat_interleave(cluster_weight, paddle.to_tensor(ns, dtype=paddle.int64), axis=1)
            quad = paddle.matmul(paddle.matmul(paddle.matmul(supA, UUt * cluster_weight), supA), U) * quad_weight * 2

            unary = paddle.mm(supW * cluster_weight, U)
            if verbose:
                if projector == 'sinkhorn':
                    print_str = f'tau={sinkhorn_tau:.3e}'
                else:
                    print_str = 'hungarian'
                print(print_str + f' #iter={i}/{max_iter} '
                      f'quad score: {(quad * U).sum().numpy().squeeze():.3e}, '
                      f'unary score: {(unary * U).sum().numpy().squeeze():.3e}')
            V = (quad + unary) / num_graphs

            U_list = []
            if projector == 'hungarian':
                n_start = 0
                for n_end in n_indices:
                    U_list.append(pygmtools.hungarian(V[n_start:n_end, :n_univ], backend='paddle'))
                    n_start = n_end
            elif projector == 'sinkhorn':
                if paddle.all(ns == ns[0]):
                    if ns[0] <= n_univ:
                        U_list.append(
                            sinkhorn(
                                V.reshape((num_graphs, -1, n_univ)),
                                max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True
                            ).reshape((-1, n_univ)))
                    else:
                        U_list.append(
                            sinkhorn(
                                V.reshape((num_graphs, -1, n_univ)).transpose((0, 2, 1)),
                                max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True
                            ).transpose((0, 2, 1)).reshape((-1, n_univ)))
                else:
                    V_list = []
                    n1 = []
                    n_start = 0
                    for n_end in n_indices:
                        V_list.append(V[n_start:n_end, :n_univ])
                        n1.append(n_end - n_start)
                        n_start = n_end
                    V_batch = build_batch(V_list)
                    n1 = paddle.to_tensor(n1, place=V_batch.place)
                    U = sinkhorn(V_batch, n1,
                                 max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True)
                    n_start = 0
                    for idx, n_end in enumerate(n_indices):
                        U_list.append(U[idx, :n_end - n_start, :])
                        n_start = n_end
            else:
                raise NameError('Unknown projecter name: {}'.format(projector))

            U = paddle.concat(U_list, axis=0)
            if num_graphs == 2:
                U[:ns[0], :] = paddle.to_tensor(paddle.eye(ns[0], n_univ), place=U.place)

            # calculate gap to discrete
            if projector == 'sinkhorn' and verbose:
                U_list_hung = []
                n_start = 0
                for n_end in n_indices:
                    U_list_hung.append(pygmtools.hungarian(V[n_start:n_end, :n_univ], backend='paddle'))
                    n_start = n_end
                U_hung = paddle.concat(U_list_hung, axis=0)
                diff = paddle.linalg.norm(paddle.mm(U, U.t()) - lastUUt)
                print(f'tau={sinkhorn_tau:.3e} #iter={i}/{max_iter} '
                      f'gap to discrete: {paddle.mean(paddle.abs(U - U_hung)).numpy().squeeze():.3e}, '
                      f'iter diff: {diff.numpy().squeeze():.3e}')

            if projector == 'hungarian' and outlier_thresh > 0:
                U_hung = U
                UUt = paddle.mm(U_hung, U_hung.t())
                cluster_weight = paddle.repeat_interleave(cluster_M, paddle.to_tensor(ns, dtype=paddle.int64), axis=0)
                cluster_weight = paddle.repeat_interleave(cluster_weight, paddle.to_tensor(ns, dtype=paddle.int64), axis=1)
                quad = paddle.matmul(paddle.matmul(paddle.matmul(supA, UUt * cluster_weight), supA), U_hung) * quad_weight * 2
                unary = paddle.mm(supW * cluster_weight, U_hung)
                max_vals = (unary + quad).max(axis=1)
                U = U * (unary + quad > outlier_thresh)
                if verbose:
                    print(f'hungarian #iter={i}/{max_iter} '
                          f'unary+quad score thresh={outlier_thresh:.3f}, '
                          f'#>thresh={paddle.sum(max_vals > outlier_thresh).numpy().squeeze()}/{max_vals.shape[0]} '
                          f'min:{max_vals.min().numpy().squeeze():.4f}, '
                          f'mean:{max_vals.mean().numpy().squeeze():.4f}, '
                          f'median:{max_vals.median().numpy().squeeze():.4f}, '
                          f'max:{max_vals.max().numpy().squeeze():.4f}')

            if paddle.linalg.norm(paddle.mm(U, U.t()) - lastUUt) < converge_thresh:
                break

        if verbose: print('-' * 20)

        if i == max_iter - 1: # not converged
            if hung_iter:
                pass
            else:
                U_list = [pygmtools.hungarian(_, backend='paddle') for _ in U_list]
                U = paddle.concat(U_list, axis=0)
                break

        # projection control
        if projector == 'hungarian':
            break
        elif sinkhorn_tau > min_tau:
            sinkhorn_tau *= sk_gamma
        else:
            if hung_iter:
                projector = 'hungarian'
            else:
                U_list = [pygmtools.hungarian(_, backend='paddle') for _ in U_list]
                U = paddle.concat(U_list, axis=0)
                break

    return U


############################################
#          Neural Network Solvers          #
############################################

from pygmtools.paddle_modules import *

class PCA_GM_Net(paddle.nn.Layer):
    """
    Paddle implementation of PCA-GM and IPCA-GM network
    """
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, cross_iter_num=-1):
        super(PCA_GM_Net, self).__init__()
        self.gnn_layer = num_layers
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(in_channel, hidden_channel)
            elif 0 < i < self.gnn_layer - 1:
                gnn_layer = Siamese_Gconv(hidden_channel, hidden_channel)
            else:
                gnn_layer = Siamese_Gconv(hidden_channel, out_channel)
                self.add_sublayer('affinity_{}'.format(i), WeightedInnerProdAffinity(out_channel))
            self.add_sublayer('gnn_layer_{}'.format(i), gnn_layer)
            if i == self.gnn_layer - 2:  # only the second last layer will have cross-graph module
                self.add_sublayer('cross_graph_{}'.format(i), paddle.nn.Linear(hidden_channel * 2, hidden_channel, weight_attr=weight_init))
                if cross_iter_num <= 0:
                 self.add_sublayer('affinity_{}'.format(i), WeightedInnerProdAffinity(hidden_channel))


    def forward(self, feat1, feat2, A1, A2, n1, n2, cross_iter_num, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(sinkhorn,
                                           dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb1, emb2 = feat1, feat2
        if cross_iter_num <= 0:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A1, emb1], [A2, emb2])

                if i == self.gnn_layer - 2:
                    affinity = getattr(self, 'affinity_{}'.format(i))
                    s = affinity(emb1, emb2)
                    s = _sinkhorn_func(s, n1, n2)
                    
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    new_emb1 = cross_graph(paddle.concat((emb1, paddle.bmm(s, emb2)), axis=-1))
                    new_emb2 = cross_graph(paddle.concat((emb2, paddle.bmm(s.transpose([0, 2, 1]), emb1)), axis=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2

            affinity = getattr(self, 'affinity_{}'.format(self.gnn_layer - 1))
            s = affinity(emb1, emb2)
            s = _sinkhorn_func(s, n1, n2)

        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A1, emb1], [A2, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = paddle.zeros((emb1.shape[0], emb1.shape[1], emb2.shape[1]))

            for x in range(cross_iter_num):
                # cross-graph convolution in second last layer
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(paddle.concat((emb1_0, paddle.bmm(s, emb2_0)), axis=-1))
                emb2 = cross_graph(paddle.concat((emb2_0, paddle.bmm(s.transpose([0,2, 1]), emb1_0)), axis=-1))

                # last layer
                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A1, emb1], [A2, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = _sinkhorn_func(s, n1, n2)

        return s

pca_gm_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc_paddle.pdparams',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1PoeWfa4v3n4Bk_9VlSUSd7akZf1rL8ct',
             'https://www.dropbox.com/scl/fi/i2ofx7g81g27vwojqiwso/pca_gm_voc_paddle.pdparams?rlkey=v7lgfjhcql9y4u8kitgdbz2zy&dl=1'],
             '03b1dedeed7195aa98431b3c561d5de3'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_willow_paddle.pdparams',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1hgCpvcvt5eoz1xbMbuDyBuYH6SCue6UD',
                'https://www.dropbox.com/scl/fi/767iyhkdzcmf1oj3wfb9r/pca_gm_willow_paddle.pdparams?rlkey=m353n3lpz0ypvrb305gsh1386&dl=1'],
                'ebf2dae8593a3640012832858bec3499'),
    'voc-all': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc-all_paddle.pdparams',
                 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=116_v4rC31T-hq3kE2d_thMKmJ65swrCD',
                 'https://www.dropbox.com/scl/fi/pbo80d0z2rc2u7nzcevbe/pca_gm_voc-all_paddle.pdparams?rlkey=odrwmzbopevv28oa4yyjpxulf&dl=1'],
                 '677c03d7180fefaaee9fcc85a03c8d53')
}

def pca_gm(feat1, feat2, A1, A2, n1, n2,
           in_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
           network, pretrain):
    """
    Paddle implementation of PCA-GM
    """
    if feat1 is None:
        forward_pass = False
        device = 'cpu'
    else:
        forward_pass = True
        device = paddle.device.get_device()
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers)
        network = network.to(device)
        if pretrain:
            if pretrain in pca_gm_pretrain_path:
                url, md5 = pca_gm_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'pca_gm_{pretrain}_paddle.pdparams', url, md5)
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {pca_gm_pretrain_path.keys()}')
    if forward_pass:
        batch_size = feat1.shape[0]
        if n1 is None:
            n1 = paddle.to_tensor([feat1.shape[1]] * batch_size)
        if n2 is None:
            n2 = paddle.to_tensor([feat2.shape[1]] * batch_size)  
        result = network(feat1, feat2, A1, A2, n1, n2, -1, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


ipca_gm_pretrain_path = { 
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ipca_gm_voc_paddle.pdparams',
            'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1bVGl9lhhzkLeWKsjiWYHgzFV-evCo1sh',
            'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/paddle_backend/ipca_gm_voc_paddle.pdparams'],
            '2fb842d4fbdeed60ac2846201a24f771'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_willow_paddle.pdparams',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=17RdDzNp2SjYahRd9aep5dEbidWJNR4uu',
                'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/paddle_backend/ipca_gm_willow_paddle.pdparams'],
                '763ea7b3c0518e27ca16c5ae987eccaa'),
}

def ipca_gm(feat1, feat2, A1, A2, n1, n2,
           in_channel, hidden_channel, out_channel, num_layers, cross_iter, sk_max_iter, sk_tau,
           network, pretrain):
    """
    Paddle implementation of IPCA-GM
    """
    if feat1 is None:
        forward_pass = False
        device = 'cpu'
    else:
        forward_pass = True
        device = paddle.device.get_device()
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers, cross_iter)
        network = network.to(device)
        if pretrain:
            if pretrain in ipca_gm_pretrain_path:
                url, md5 = ipca_gm_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'ipca_gm_{pretrain}_paddle.pdparams', url, md5)
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {ipca_gm_pretrain_path.keys()}')
    if forward_pass:
        batch_size = feat1.shape[0]
        if n1 is None:
            n1 = paddle.to_tensor([feat1.shape[1]] * batch_size)
        if n2 is None:
            n2 = paddle.to_tensor([feat2.shape[1]] * batch_size)
        result = network(feat1, feat2, A1, A2, n1, n2, cross_iter, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network

class CIE_Net(paddle.nn.Layer):
    """
    Paddle implementation of CIE graph matching network
    """
    def __init__(self, in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers):
        super(CIE_Net, self).__init__()
        self.gnn_layer = num_layers
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_ChannelIndependentConv(in_node_channel, hidden_channel, in_edge_channel)
            elif 0 < i < self.gnn_layer - 1:
                gnn_layer = Siamese_ChannelIndependentConv(hidden_channel, hidden_channel, hidden_channel)
            else:
                gnn_layer = Siamese_ChannelIndependentConv(hidden_channel, out_channel, hidden_channel)
                self.add_sublayer('affinity_{}'.format(i), WeightedInnerProdAffinity(out_channel))
            self.add_sublayer('gnn_layer_{}'.format(i), gnn_layer)
            if i == self.gnn_layer - 2:  # only the second last layer will have cross-graph module
                self.add_sublayer('cross_graph_{}'.format(i), paddle.nn.Linear(hidden_channel * 2, hidden_channel, weight_attr=weight_init))
                self.add_sublayer('affinity_{}'.format(i), WeightedInnerProdAffinity(hidden_channel))

    def forward(self, feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(sinkhorn,
                                           dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb1, emb2 = feat_node1, feat_node2
        emb_edge1, emb_edge2 = feat_edge1, feat_edge2
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            # during forward process, the network structure will not change
            emb1, emb2, emb_edge1, emb_edge2 = gnn_layer([A1, emb1, emb_edge1], [A2, emb2, emb_edge2])

            if i == self.gnn_layer - 2:
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = _sinkhorn_func(s, n1, n2)

                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                new_emb1 = cross_graph(paddle.concat((emb1, paddle.bmm(s, emb2)), axis=-1))
                new_emb2 = cross_graph(paddle.concat((emb2, paddle.bmm(s.transpose([0,2, 1]), emb1)), axis=-1))
                emb1 = new_emb1
                emb2 = new_emb2

        affinity = getattr(self, 'affinity_{}'.format(self.gnn_layer - 1))
        s = affinity(emb1, emb2)
        s = _sinkhorn_func(s, n1, n2)
        return s

cie_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_voc_paddle.pdparams',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=13dtxDfySvfqQcYvPiTd8Pb2xgcXIesAz',
             'https://www.dropbox.com/scl/fi/ivewf6zi2zlu759qfoqkh/cie_voc_paddle.pdparams?rlkey=t8i9k1cvjaacqiu0u70o25h2d&dl=1'],
             '2c52c70e4a8919d24fde261756d1d6c4'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_willow_paddle.pdparams',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1_-G00V3yhJ3IL_Xp6cMcVV9N2vWgEDwk',
                'https://www.dropbox.com/scl/fi/dusrgilik8wwuqxv85g9i/cie_willow_paddle.pdparams?rlkey=3j6w7iqjyoxasiubg1todly92&dl=1'],
                '2619383120ca67d68c40eebd9dde9d95'),
}

def cie(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2,
        in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
        network, pretrain):
    """
    Paddle implementation of CIE
    """
    if feat_node1 is None:
        forward_pass = False
        device = 'cpu'
    else:
        forward_pass = True
        device = paddle.device.get_device()
    if network is None:
        network = CIE_Net(in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers)
        network = network.to(device)
        if pretrain:
            if pretrain in cie_pretrain_path:
                url, md5 = cie_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'cie_{pretrain}_paddle.pdparams', url, md5)
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {cie_pretrain_path.keys()}')

    if forward_pass:
        batch_size = feat_node1.shape[0]
        if n1 is None:
            n1 = paddle.to_tensor([feat_node1.shape[1]] * batch_size)
        if n2 is None:
            n2 = paddle.to_tensor([feat_node1.shape[1]] * batch_size)
        result = network(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


class NGM_Net(paddle.nn.Layer):
    """
    Paddle implementation of NGM network
    """

    def __init__(self, gnn_channels, sk_emb):
        super(NGM_Net, self).__init__()
        self.gnn_layer = len(gnn_channels)
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = NGMConvLayer(1, 1,
                                         gnn_channels[i] + sk_emb, gnn_channels[i],
                                         sk_channel=sk_emb)
            else:
                gnn_layer = NGMConvLayer(gnn_channels[i - 1] + sk_emb, gnn_channels[i - 1],
                                         gnn_channels[i] + sk_emb, gnn_channels[i],
                                         sk_channel=sk_emb)
            self.add_sublayer('gnn_layer_{}'.format(i), gnn_layer)
        self.classifier = nn.Linear(gnn_channels[-1] + sk_emb, 1)

    def forward(self, K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(sinkhorn,
                                           dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb = v0
        A = paddle.cast((K != 0), K.dtype)
        emb_K = K.unsqueeze(-1)

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, f'gnn_layer_{i}')
            emb_K, emb = gnn_layer(A, emb_K, emb, n1, n2, sk_func=_sinkhorn_func)

        v = self.classifier(emb)
        s = v.reshape([v.shape[0], n2max, -1]).transpose((0, 2, 1))

        return _sinkhorn_func(s, n1, n2, dummy_row=True)

ngm_pretrain_path = { 
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_voc_paddle.pdparams',
             'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/paddle_backend/ngm_voc_paddle.pdparams',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1cd5AvddQtpWmENPLEWjJ76cKG7Df43nh'],
             'bf1808dd16304e03ff33133c7ea9de90'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_willow_paddle.pdparams',
                'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/paddle_backend/ngm_willow_paddle.pdparams',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1j1kXWsassE3bAVWjPy2g0jUGG2IeODc8'],
                'a5daf06cdaf6cc370928b5f8fc01c585'),
}

def ngm(K, n1, n2, n1max, n2max, x0, gnn_channels, sk_emb, sk_max_iter, sk_tau, network, return_network, pretrain):
    """
    Paddle implementation of NGM
    """
    if K is None:
        forward_pass = False
        device = 'cpu'
    else:
        forward_pass = True
        device = paddle.device.get_device()
    if network is None:
        network = NGM_Net(gnn_channels, sk_emb)
        network = network.to(device)
        if pretrain:
            if pretrain in ngm_pretrain_path:
                url, md5 = ngm_pretrain_path[pretrain]
                try:
                    filename = pygmtools.utils.download(f'ngm_{pretrain}_paddle.pdparams', url, md5)
                except:
                    filename = os.path.dirname(__file__) + f'/temp/ngm_{pretrain}_paddle.pdparams'
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {ngm_pretrain_path.keys()}')

    if forward_pass:
        batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
        v0 = v0 / paddle.mean(v0)
        result = network(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


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
    _check_data_type(input[0], 'input', True)
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


def permutation_loss(pred_dsmat, gt_perm, n1, n2):
    """
    Paddle implementation of permutation_loss
    """
    batch_num = pred_dsmat.shape[0]

    pred_dsmat = pred_dsmat.astype("float32")

    if not ((pred_dsmat >= 0) * (pred_dsmat <= 1)).all():
        raise ValueError("pred_dsmat contains invalid numerical entries.")
    if not ((gt_perm >= 0) * (gt_perm <= 1)).all():
        raise ValueError("gt_perm contains invalid numerical entries.")

    if n1 is None:
        n1 = paddle.to_tensor([pred_dsmat.shape[1] for _ in range(batch_num)], place=pred_dsmat.place)
    if n2 is None:
        n2 = paddle.to_tensor([pred_dsmat.shape[2] for _ in range(batch_num)], place=pred_dsmat.place)

    loss = paddle.to_tensor(0., place=pred_dsmat.place)
    n_sum = paddle.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(n1[b]), slice(n2[b])]
        loss += paddle.nn.functional.binary_cross_entropy(
            pred_dsmat[batch_slice],
            gt_perm[batch_slice],
            reduction='sum')
        n_sum += paddle.to_tensor(n1[b].astype(n_sum.dtype), place=pred_dsmat.place)

    return loss / n_sum


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
            ne2 = [edge_aff.shape[2]] * batch_size
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


def _check_data_type(input: paddle.Tensor, var_name, raise_err):
    """
    Paddle implementation of _check_data_type
    """
    if raise_err and type(input) is not paddle.Tensor:
        raise ValueError(f'Expected Paddle Tensor{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}.')
    return type(input) is paddle.Tensor


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
    Paddle implementation of _transpose
    """
    return paddle.transpose(input, (dim2, dim1))


def _mm(input1, input2):
    """
    Paddle implementation of _mm
    """
    return paddle.mm(input1, input2)

def _load_model(model, path):
    """
    Load Paddle model from a given path. Unmatched keys shall be shown in paddle warning.
    """
    module = model
    module.set_dict(paddle.load(path))