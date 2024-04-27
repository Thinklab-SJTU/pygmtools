# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
import jittor as jt
from jittor import Var
import functools
import itertools
import pygmtools.utils
from multiprocessing import Pool
import os
from pygmtools.numpy_backend import _hung_kernel

#############################################
#     Linear Assignment Problem Solvers     #
#############################################

def hungarian(s: Var, n1: Var=None, n2: Var=None,
              unmatch1: Var=None, unmatch2: Var=None,
              nproc: int=1) -> Var:
    """
    Jittor implementation of Hungarian algorithm
    """
    batch_num = s.shape[0]

    perm_mat = s.detach().numpy() * -1
    if n1 is not None:
        n1 = n1.numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.numpy()
    else:
        n2 = [None] * batch_num
    if unmatch1 is not None:
        unmatch1 = -unmatch1.numpy()
    else:
        unmatch1 = [None] * batch_num
    if unmatch2 is not None:
        unmatch2 = -unmatch2.numpy()
    else:
        unmatch2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2, unmatch1, unmatch2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) for b in range(batch_num)])

    perm_mat = jt.Var(perm_mat)

    return perm_mat

def sinkhorn(s: Var, nrows: Var=None, ncols: Var=None,
             unmatchrows: Var=None, unmatchcols: Var=None,
             dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False) -> Var:
    """
    Jittor implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.transpose(1, 2)
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True

    if nrows is None:
        nrows = jt.Var([s.shape[1] for _ in range(batch_size)])
    if ncols is None:
        ncols = jt.Var([s.shape[2] for _ in range(batch_size)])

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if jt.any(transposed_batch):
        s_t = s.transpose(1, 2)
        s_t = jt.concat((
            s_t[:, :s.shape[1], :],
            jt.full((batch_size, s.shape[1], s.shape[2]-s.shape[1]), -float('inf'))), dim=2)
        cond = transposed_batch.view(batch_size, 1, 1)
        if cond.shape != s.shape:
            cond = cond.expand(s.shape)
        s = jt.where(cond, s_t, s)

        new_nrows = jt.where(transposed_batch, ncols, nrows)
        new_ncols = jt.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = jt.concat((
                unmatchrows,
                jt.full((batch_size, unmatchcols.shape[1]-unmatchrows.shape[1]), -float('inf'))), dim=1)
            new_unmatchrows = jt.where(transposed_batch.view(batch_size, 1).expand(unmatchcols.shape), unmatchcols, unmatchrows_pad)[:, :unmatchrows.shape[1]]
            new_unmatchcols = jt.where(transposed_batch.view(batch_size, 1).expand(unmatchcols.shape), unmatchrows_pad, unmatchcols)
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
        log_s = jt.concat((log_s, jt.full(dummy_shape, -float('inf'))), dim=1)
        if unmatchrows is not None:
            unmatchrows = jt.concat((unmatchrows, jt.full((dummy_shape[0], dummy_shape[1]), -float('inf'))), dim=1)
        for b in range(batch_size):
            log_s[b, int(ori_nrows[b]):int(nrows[b]), :int(ncols[b])] = -100

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = jt.full((log_s.shape[0], log_s.shape[1]+1, log_s.shape[2]+1), -float('inf'))
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        for b in range(batch_size):
            r, c = int(nrows[b]), int(ncols[b])
            log_s[b, 0:r, c] = unmatchrows[b, 0:r]
            log_s[b, r, 0:c] = unmatchcols[b, 0:c]
    row_mask = jt.zeros((batch_size, log_s.shape[1], 1), dtype=jt.bool)
    col_mask = jt.zeros((batch_size, 1, log_s.shape[2]), dtype=jt.bool)
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
                m = log_s.max(2, keepdims=True)  #optimized logsumexp
                log_sum = jt.nn.logsumexp(log_s - m, 2, keepdim=True) + m
                log_s = log_s - jt.where(row_mask, log_sum, jt.zeros_like(log_sum))
                if jt.flags.use_cuda == 0:
                    if jt.any(jt.isnan(log_s)):
                        raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
            else:
                m = log_s.max(1, keepdims=True)
                log_sum = jt.nn.logsumexp(log_s - m, 1, keepdim=True) + m                
                log_s = log_s - jt.where(col_mask, log_sum, jt.zeros_like(log_sum))
                if jt.flags.use_cuda == 0:
                    if jt.any(jt.isnan(log_s)):
                        raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

        ret_log_s = log_s
    else:
        ret_log_s = jt.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), dtype=log_s.dtype)
        for b in range(batch_size):
            r,c = nrows[b],ncols[b]
            if not isinstance(nrows[b],int):
                r = int(nrows[b].item())
            if not isinstance(ncols[b],int):
                c = int(ncols[b].item())
            log_s_b = log_s[b, 0:r, 0:c]
            row_mask_b = row_mask[b, 0:r, :]
            col_mask_b = col_mask[b, :, 0:c]
            for i in range(max_iter):
                if i % 2 == 0:
                    m = log_s_b.max(1, keepdims=True)
                    log_sum = jt.nn.logsumexp(log_s_b - m, 1, keepdim=True) + m
                    log_s_b = log_s_b - jt.where(row_mask_b, log_sum, jt.zeros_like(log_sum))
                else:
                    m = log_s_b.max(0, keepdims=True)
                    log_sum = jt.nn.logsumexp(log_s_b - m, 0, keepdim=True) + m
                    log_s_b = log_s_b - jt.where(col_mask_b, log_sum, jt.zeros_like(log_sum))
            ret_log_s[b, 0:r, 0:c] = log_s_b

    if unmatchrows is not None and unmatchcols is not None:
        ncols -= 1
        nrows -= 1
        for b in range(batch_size):
            r, c = int(nrows[b]), int(ncols[b])
            ret_log_s[b, 0:r+1, c] = -float('inf')
            ret_log_s[b, r, 0:c] = -float('inf')
        ret_log_s = ret_log_s[:, :-1, :-1]

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            ret_log_s[int(b), int(ori_nrows[b]):int(nrows[b]), :int(ncols[b])] = -float('inf')

    if jt.any(transposed_batch):
        s_t = ret_log_s.transpose(1, 2)
        s_t = jt.concat((
            s_t[:, :ret_log_s.shape[1], :],
            jt.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2]-ret_log_s.shape[1]), -float('inf'))), dim=2)
        cond = transposed_batch.view(batch_size, 1, 1)
        if cond.shape != s_t.shape:
            cond = cond.expand(s_t.shape)
        ret_log_s = jt.where(cond, s_t, ret_log_s)    

    if transposed:
        ret_log_s = ret_log_s.transpose(1, 2)

    return jt.exp(ret_log_s)

#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################

def rrwm(K: Var, n1: Var, n2: Var, n1max, n2max, x0: Var,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> Var:
    """
    Jittor implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
    d = K.sum(dim=2, keepdims=True)
    dmax = d.max(dim=1, keepdims=True) 
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    for i in range(max_iter):
        # try fixing memory error caused by growing scale of
        # computation graph when testing multi-graph solvers
        if jt.number_of_lived_ops() > 100000:
            jt.clean_graph() 
            
        # random walk
        v = jt.bmm(K, v)
        last_v = v
        n = jt.norm(v, p=1, dim=1, keepdim=True)
        v = v / n

        # reweighted jump
        s = v.view((batch_num, int(n2max), int(n1max))).transpose(1, 2)
        s = beta * s / s.max(dim=1, keepdims=True).max(dim=2, keepdims=True)
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True).transpose(1, 2).reshape(batch_num, n1n2, 1) + \
            (1 - alpha) * v
        n = jt.norm(v, p=1, dim=1, keepdim=True)
        v = jt.matmul(v, 1 / n)
        if (v - last_v).sum().sqrt() < 1e-5:
            break

    return v.view((batch_num, int(n2max), int(n1max))).transpose(1, 2)

def sm(K: Var, n1: Var, n2: Var, n1max, n2max, x0: Var,
        max_iter: int) -> Var:
    """
    Jittor implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for i in range(max_iter):
        v = jt.bmm(K, v)
        n = jt.norm(v, p=2, dim=1)
        v = jt.matmul(v, (1 / n).reshape(batch_num, 1, 1))
        if (v - vlast).sum().sqrt() < 1e-5:
            break        
        vlast = v

    x = v.reshape(batch_num, n2max, n1max).transpose(1,2)
    return x

def ipfp(K: Var, n1: Var, n2: Var, n1max, n2max, x0: Var,
         max_iter) -> Var:
    """
    Jittor implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v
    best_v = v
    best_obj = jt.full((batch_num, 1, 1), -1)

    def comp_obj_score(v1, K, v2):
        return jt.bmm(jt.bmm(v1.view(batch_num, 1, -1), K), v2)

    for i in range(max_iter):
        cost = jt.bmm(K, v).reshape((batch_num, int(n2max), int(n1max))).transpose(1, 2)
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.transpose(1, 2).view(batch_num, -1, 1)
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = - alpha / beta
        cond = jt.logical_or(beta >= 0, t0 >= 1)
        if cond.shape != binary_v.shape:
            cond = cond.expand(binary_v.shape)
        v = jt.where(cond, binary_v, v + t0 * (binary_v - v))
        last_v_obj = comp_obj_score(last_v, K, last_v)

        current_obj = comp_obj_score(binary_v, K, binary_v)
        cond = current_obj > best_obj
        if cond.shape != binary_v.shape:
            cond = cond.expand(binary_v.shape)
        best_v = jt.where(cond, binary_v, best_v)  # current_obj > best_obj
        best_obj = jt.where(current_obj > best_obj, current_obj, best_obj)

        if jt.max(jt.abs(last_v_obj - current_obj) / last_v_obj) < 1e-3:
            break
        last_v = v

    pred_x = best_v.reshape((batch_num, int(n2max), int(n1max))).transpose(1, 2)
    return pred_x


############################################
#      Multi-Graph Matching Solvers        #
############################################


def cao_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    r"""
    Jittor implementation of CAO solver (mode="c")

    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X: initial matching, (m, m, n, n)
    :param num_graph: number of graphs, int
    :param num_node: number of nodes, int
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    param_lambda = lambda_init

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='jittor').unsqueeze(-1).unsqueeze(-1)

    for iter in range(max_iter):
        if iter >= iter_boost:
            param_lambda = np.min([param_lambda * lambda_step, lambda_max])
        # pair_con = get_batch_pc_opt(X)
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - jt.init.eye(m) * pair_aff
        norm = jt.max(pair_aff)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                con_ori = _get_single_pc_opt(X, i, j)
                # con_ori = jt.sqrt(pair_con[i, j])
                if iter < iter_boost:
                    score_ori = aff_ori
                else:
                    score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
                X_upt = X[i, j]
                for k in range(m):
                    X_combo = jt.matmul(X[i, k], X[k, j])
                    aff_combo = _comp_aff_score(X_combo, K[i, j]) / norm
                    con_combo = _get_single_pc_opt(X, i, j, X_combo)
                    # con_combo = jt.sqrt(pair_con[i, k] * pair_con[k, j])
                    if iter < iter_boost:
                        score_combo = aff_combo
                    else:
                        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda
                    if score_combo > score_ori:
                        X_upt = X_combo
                X[i, j] = X_upt
                X[j, i] = X_upt.transpose(0, 1)
    return X


def cao_fast_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    r"""
    Jittor implementation of CAO solver in fast config (mode="pc")

    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X: initial matching, (m, m, n, n)
    :param num_graph: number of graphs, int
    :param num_node: number of nodes, int
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    param_lambda = lambda_init

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='jittor').unsqueeze(-1).unsqueeze(-1)

    mask1 = jt.arange(m).reshape(m, 1).repeat(1, m)
    mask2 = jt.arange(m).reshape(1, m).repeat(m, 1)
    mask = (mask1 < mask2).float()
    X_mask = mask.reshape(m, m, 1, 1)

    for iter in range(max_iter):
        if iter >= iter_boost:
            param_lambda = np.min([param_lambda * lambda_step, lambda_max])

        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - jt.init.eye(m) * pair_aff
        norm = jt.max(pair_aff)

        X1 = X.reshape(m, 1, m, n, n).repeat(1, m, 1, 1, 1).reshape(-1, n, n)  # X1[i,j,k] = X[i,k]
        X2 = X.reshape(1, m, m, n, n).repeat(m, 1, 1, 1, 1).transpose(1, 2).reshape(-1, n, n)  # X2[i,j,k] = X[k,j]
        X_combo = jt.bmm(X1, X2).reshape(m, m, m, n, n) # X_combo[i,j,k] = X[i, k] * X[k, j]

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        pair_con = _get_batch_pc_opt(X)
        con_ori = jt.sqrt(pair_con)

        K_repeat = K.reshape(m, m, 1, n * n, n * n).repeat(1, 1, m, 1, 1).reshape(-1, n * n, n * n)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K_repeat) / norm).reshape(m, m, m)
        con1 = pair_con.reshape(m, 1, m).repeat(1, m, 1)  # con1[i,j,k] = pair_con[i,k]
        con2 = pair_con.reshape(1, m, m).repeat(m, 1, 1).transpose(1, 2)  # con2[i,j,k] = pair_con[j,k]
        con_combo = jt.sqrt(con1 * con2)

        if iter < iter_boost:
            score_ori = aff_ori
            score_combo = aff_combo
        else:
            score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
            score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

        score_combo_max = jt.max(score_combo, dim=-1)
        idx = []
        for i in range(score_combo.shape[0]):
            idx.append([])
            for j in range(score_combo.shape[1]):
                ix = jt.where(score_combo[i][j]==score_combo_max[i][j])[0]
                idx[i].append(ix[0].item() if ix.shape[0]>1 else ix.item())
        idx = jt.Var(idx)

        if not jt.all(score_combo_max + 1e-4 >= score_ori):
            raise RuntimeError('CAO-fast internal error', jt.min(score_combo_max - score_ori))
        X_upt = X_combo[mask1, mask2, idx, :, :]
        X = X_upt * X_mask + X_upt.transpose(0, 1).transpose(2, 3) * X_mask.transpose(0, 1) + X * (1 - X_mask - X_mask.transpose(0, 1))
        if not jt.all(X.transpose(0, 1).transpose(2, 3) == X):
            raise RuntimeError('CAO-fast internal error')
    return X


def mgm_floyd_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='jittor').unsqueeze(-1).unsqueeze(-1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - jt.init.eye(m) * pair_aff
        norm = jt.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, jt.mean(pair_aff).item(), jt.mean(get_batch_pc_opt(X)).item()
        # ))

        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                score_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                X_combo = jt.matmul(X[i, k], X[k, j])
                score_combo = _comp_aff_score(X_combo, K[i, j]) / norm

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose(0, 1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - jt.init.eye(m) * pair_aff
        norm = jt.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                con_ori = _get_single_pc_opt(X, i, j)
                # con_ori = jt.sqrt(pair_con[i, j])
                score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda

                X_combo = jt.matmul(X[i, k], X[k, j])
                aff_combo = _comp_aff_score(X_combo, K[i, j]) / norm
                con_combo = _get_single_pc_opt(X, i, j, X_combo)
                # con_combo = jt.sqrt(pair_con[i, k] * pair_con[k, j])
                score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.transpose(0, 1)
    return X


def mgm_floyd_fast_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node

    def _comp_aff_score(x, k):
        return pygmtools.utils.compute_affinity_score(x, k, backend='jittor').unsqueeze(-1).unsqueeze(-1)

    mask1 = jt.arange(m).reshape(m, 1).repeat(1, m)
    mask2 = jt.arange(m).reshape(1, m).repeat(m, 1)
    mask = (mask1 < mask2).float()
    X_mask = mask.reshape(m, m, 1, 1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - jt.init.eye(m) * pair_aff
        norm = jt.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, jt.mean(pair_aff).item(), jt.mean(get_batch_pc_opt(X)).item()
        # ))

        X1 = X[:, k].reshape(m, 1, n, n).repeat(1, m, 1, 1).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n).repeat(m, 1, 1, 1).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = jt.bmm(X1, X2).reshape(m, m, n, n)

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)

        score_ori = aff_ori
        score_combo = aff_combo

        upt = (score_ori < score_combo).float()
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.transpose(0, 1).transpose(2, 3) * (1 - X_mask)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - jt.init.eye(m) * pair_aff
        norm = jt.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)

        X1 = X[:, k].reshape(m, 1, n, n).repeat(1, m, 1, 1).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n).repeat(m, 1, 1, 1).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = jt.bmm(X1, X2).reshape(m, m, n, n)

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)

        con_ori = jt.sqrt(pair_con)
        con1 = pair_con[:, k].reshape(m, 1).repeat(1, m)
        con2 = pair_con[k, :].reshape(1, m).repeat(m, 1)
        con_combo = jt.sqrt(con1 * con2)

        score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

        upt = (score_ori < score_combo).float()
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.transpose(0, 1).transpose(2, 3) * (1 - X_mask)
    return X


def _get_single_pc_opt(X, i, j, Xij=None):
    """
    CAO/Floyd helper function (compute consistency)
    :param X: (m, m, n, n) all the matching results
    :param i: index
    :param j: index
    :return: the consistency of X_ij
    """
    m, _, n, _ = X.size()
    if Xij is None:
        Xij = X[i, j]
    X1 = X[i, :].reshape(-1, n, n)
    X2 = X[:, j].reshape(-1, n, n)
    X_combo = jt.bmm(X1, X2)
    pair_con = 1 - jt.sum(jt.abs(Xij - X_combo)) / (2 * n * m)
    return pair_con


def _get_batch_pc_opt(X):
    """
    CAO/Floyd-fast helper function (compute consistency in batch)
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    m, _, n, _ = X.size()
    X1 = X.reshape(m, 1, m, n, n).repeat(1, m, 1, 1, 1).reshape(-1, n, n)  # X1[i, j, k] = X[i, k]
    X2 = X.reshape(1, m, m, n, n).repeat(m, 1, 1, 1, 1).transpose(1, 2).reshape(-1, n, n)  # X2[i, j, k] = X[k, j]
    X_combo = jt.bmm(X1, X2).reshape(m, m, m, n, n)
    X_ori = X.reshape(m, m, 1, n, n).repeat(1, 1, m, 1, 1)
    pair_con = 1 - jt.sum(jt.abs(X_combo - X_ori), dims=(2, 3, 4)) / (2 * n * m)
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
    Jittor implementation of Graduated Assignment for Multi-Graph Matching (with compatibility for 2GM and clustering)
    """

    num_graphs = A.shape[0]
    if ns is None:
        ns = jt.full((num_graphs,), A.shape[1], dtype=jt.int)
    n_indices = jt.cumsum(ns, dim=0)

    # build a super adjacency matrix A
    supA = jt.zeros((n_indices[-1].item(), n_indices[-1].item()))
    for i in range(num_graphs):
        start_n = (n_indices[i] - ns[i]).item()
        end_n = n_indices[i].item()
        supA[start_n:end_n, start_n:end_n] = A[i, :ns[i].item(), :ns[i].item()]

    # handle the type of n_univ
    if type(n_univ) is jt.Var:
        n_univ = n_univ.item()

    # randomly init U
    if U0 is None:
        U0 = jt.full((n_indices[-1].item(), n_univ), 1 / n_univ)
        U0 += jt.randn_like(U0) / 1000

    # init cluster_M if not given
    if cluster_M is None:
        cluster_M = jt.ones((num_graphs, num_graphs))

    # reshape W into supW
    supW = jt.zeros((n_indices[-1].item(), n_indices[-1].item()))
    for i, j in itertools.product(range(num_graphs), repeat=2):
        start_x = (n_indices[i] - ns[i]).item()
        end_x = n_indices[i].item()
        start_y = (n_indices[j] - ns[j]).item()
        end_y = n_indices[j].item()
        supW[start_x:end_x, start_y:end_y] = W[i, j, :ns[i].item(), :ns[j].item()]

    U = GAMGMJittorFunc.apply(
        bb_smooth,
        supA, supW, ns, n_indices, n_univ, num_graphs, U0,
        init_tau, min_tau, sk_gamma,
        sk_iter, max_iter, quad_weight,
        converge_thresh, outlier_thresh,
        verbose,
        cluster_M, projector, hung_iter
    )

    # build MultiMatchingResult
    result = pygmtools.utils.MultiMatchingResult(True, 'jittor')

    for i in range(num_graphs):
        start_n = n_indices[i] - ns[i]
        end_n = n_indices[i]
        result[i] = U[start_n.item():end_n.item()]

    return result


class GAMGMJittorFunc(jt.Function):
    """
    Jittor wrapper to support forward and backward pass (by black-box differentiation)
    """

    def execute(self, bb_smooth, supA, supW, ns, n_indices, n_univ, num_graphs, U0, *args):
        # save parameters
        self.bb_smooth = bb_smooth
        self.named_args = supA, supW, ns, n_indices, n_univ, num_graphs, U0
        self.list_args = args

        # real solver function
        U = gamgm_real(supA, supW, ns, n_indices, n_univ, num_graphs, U0, *args)

        # save result
        self.U = U
        return U

    def grad(self, dU):
        epsilon = 1e-8
        bb_smooth = self.bb_smooth
        supA, supW, ns, n_indices, n_univ, num_graphs, U0 = self.named_args
        args = self.list_args
        U = self.U

        for i, j in itertools.product(range(num_graphs), repeat=2):
            start_x = (n_indices[i] - ns[i]).item()
            end_x = n_indices[i].item()
            start_y = (n_indices[j] - ns[j]).item()
            end_y = n_indices[j].item()
            supW[start_x:end_x, start_y:end_y] += bb_smooth * jt.matmul(dU[start_x:end_x], dU[start_y:end_y].transpose(0, 1))

        U_prime = gamgm_real(supA, supW, ns, n_indices, n_univ, num_graphs, U0, *args)

        grad_supW = jt.zeros((n_indices[-1].item(), n_indices[-1].item()))
        for i, j in itertools.product(range(num_graphs), repeat=2):
            start_x = (n_indices[i] - ns[i]).item()
            end_x = n_indices[i].item()
            start_y = (n_indices[j] - ns[j]).item()
            end_y = n_indices[j].item()
            X = jt.matmul(U[start_x:end_x], U[start_y:end_y].transpose(0, 1))
            X_prime = jt.matmul(U_prime[start_x:end_x], U_prime[start_y:end_y].transpose(0, 1))
            grad_supW[start_x:end_x, start_y:end_y] = -(X - X_prime) / (bb_smooth + epsilon)

        return_list = [None, None, grad_supW] + [None] * (len(args) + 8 - 3)
        return tuple(return_list)


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
        with jt.enable_grad():
            for i in range(max_iter):
                # compact matrix form update of V
                UUt = jt.matmul(U, U.t())
                lastUUt = UUt
                # jittor does not accept array as the second parameter of repeat_interleave and jittor Var is based on numpy array
                import numpy as np
                cluster_weight = jt.Var(np.repeat(cluster_M, ns.long().data, axis=0))
                cluster_weight = jt.Var(np.repeat(cluster_weight, ns.long().data, axis=1))
                quad, chains = supA, [UUt * cluster_weight, supA, U]
                for matrix in chains:
                    quad = jt.matmul(quad, matrix)
                quad *= (quad_weight * 2)
                # quad = jt.chain_matmul(supA, UUt * cluster_weight, supA, U) * quad_weight * 2
                unary = jt.matmul(supW * cluster_weight, U)
                if verbose:
                    if projector == 'sinkhorn':
                        print_str = f'tau={sinkhorn_tau:.3e}'
                    else:
                        print_str = 'hungarian'
                    print(print_str + f' #iter={i}/{max_iter} '
                        f'quad score: {(quad * U).sum():.3e}, unary score: {(unary * U).sum():.3e}')
                V = (quad + unary) / num_graphs

                U_list = []
                if projector == 'hungarian':
                    n_start = 0
                    for n_end in n_indices:
                        if isinstance(n_start, Var):
                            n_start = n_start.item()
                        U_list.append(pygmtools.hungarian(V[n_start:n_end.item(), :n_univ], backend='jittor'))
                        n_start = n_end
                elif projector == 'sinkhorn':
                    if jt.all(ns == ns[0]):
                        if ns[0] <= n_univ:
                            U_list.append(
                                sinkhorn(
                                    V.reshape(num_graphs, -1, n_univ),
                                    max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True
                                ).reshape(-1, n_univ))
                        else:
                            U_list.append(
                                sinkhorn(
                                    V.reshape(num_graphs, -1, n_univ).transpose(1, 2),
                                    max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True
                                ).transpose(1, 2).reshape(-1, n_univ))
                    else:
                        V_list = []
                        n1 = []
                        n_start = 0
                        for n_end in n_indices:
                            if isinstance(n_start, Var):
                                n_start = n_start.item()
                            V_list.append(V[n_start:n_end.item(), :n_univ])
                            n1.append(n_end.item() - n_start)
                            n_start = n_end
                        V_batch = build_batch(V_list)
                        n1 = jt.Var(n1)
                        U = sinkhorn(V_batch, n1,
                                    max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True)
                        n_start = 0
                        for idx, n_end in enumerate(n_indices):
                            U_list.append(U[idx, :n_end.item() - n_start, :])
                            n_start = n_end.item()
                else:
                    raise NameError('Unknown projecter name: {}'.format(projector))

                U = jt.concat(U_list, dim=0)
                if num_graphs == 2:
                    U[:ns[0], :] = jt.init.eye(ns[0], n_univ)

                # calculate gap to discrete
                if projector == 'sinkhorn' and verbose:
                    U_list_hung = []
                    n_start = 0
                    for n_end in n_indices:
                        n_end = n_end.item()
                        U_list_hung.append(pygmtools.hungarian(V[n_start:n_end, :n_univ], backend='jittor'))
                        n_start = n_end
                    U_hung = jt.concat(U_list_hung, dim=0)
                    diff = jt.norm(jt.matmul(U, U.t()) - lastUUt).sum()
                    print(f'tau={sinkhorn_tau:.3e} #iter={i}/{max_iter} '
                          f'gap to discrete: {jt.mean(jt.abs(U - U_hung)).item():.3e}, '
                          f'iter diff: {diff.item():.3e}')

                if projector == 'hungarian' and outlier_thresh > 0:
                    U_hung = U
                    UUt = jt.matmul(U_hung, U_hung.t())
                    cluster_weight = jt.Var(np.repeat(cluster_M, ns.long().data, axis=0))
                    cluster_weight = jt.Var(np.repeat(cluster_weight, ns.long().data, axis=1))
                    quad, chains = supA, [UUt * cluster_weight, supA, U_hung]
                    for matrix in chains:
                        quad = jt.matmul(quad, matrix)
                    quad *= (quad_weight * 2)
                    unary = jt.matmul(supW * cluster_weight, U_hung)
                    max_vals = (unary + quad).max(dim=1)
                    U = U * (unary + quad > outlier_thresh)
                    if verbose:
                        print(f'hungarian #iter={i}/{max_iter} '
                            f'unary+quad score thresh={outlier_thresh:.3f}, #>thresh={jt.sum(max_vals > outlier_thresh)}/{max_vals.shape[0]}'
                            f' min:{max_vals.min():.4f}, mean:{max_vals.mean():.4f}, median:{max_vals.median():.4f}, max:{max_vals.max():.4f}')

                if (jt.matmul(U, U.t()) - lastUUt).pow(2).sum().sqrt() < converge_thresh:
                    break

        if verbose: print('-' * 20)

        if i == max_iter - 1: # not converged
            if hung_iter:
                pass
            else:
                U_list = [pygmtools.hungarian(_, backend='jittor') for _ in U_list]
                U = jt.concat(U_list, dim=0)
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
                U_list = [pygmtools.hungarian(_, backend='jittor') for _ in U_list]
                U = jt.concat(U_list, dim=0)
                break

    return U


############################################
#          Neural Network Solvers          #
############################################

from pygmtools.jittor_modules import *


class PCA_GM_Net(Sequential):
    """
    Jittor implementation of PCA-GM and IPCA-GM network
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
                self.add_module('affinity_{}'.format(i), WeightedInnerProdAffinity(out_channel))
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            if i == self.gnn_layer - 2:  # only the second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), jt.nn.Linear(hidden_channel * 2, hidden_channel))
                if cross_iter_num <= 0:
                    self.add_module('affinity_{}'.format(i), WeightedInnerProdAffinity(hidden_channel))


    def execute(self, feat1, feat2, A1, A2, n1, n2, cross_iter_num, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(sinkhorn,
                                           dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb1, emb2 = feat1, feat2
        if cross_iter_num <= 0:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = self.layers[f'gnn_layer_{i}']
                emb1, emb2 = gnn_layer([A1, emb1], [A2, emb2])

                if i == self.gnn_layer - 2:
                    affinity = self.layers[f'affinity_{i}']
                    s = affinity(emb1, emb2)
                    s = _sinkhorn_func(s, n1, n2)

                    cross_graph = self.layers[f'cross_graph_{i}']
                    new_emb1 = cross_graph(jt.concat((emb1, jt.bmm(s, emb2)), dim=-1))
                    new_emb2 = cross_graph(jt.concat((emb2, jt.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2

            affinity = self.layers[f'affinity_{self.gnn_layer - 1}']
            s = affinity(emb1, emb2)
            s = _sinkhorn_func(s, n1, n2)

        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = self.layers[f'gnn_layer_{i}']
                emb1, emb2 = gnn_layer([A1, emb1], [A2, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = jt.zeros((emb1.shape[0], emb1.shape[1], emb2.shape[1]))

            for x in range(cross_iter_num):
                # cross-graph convolution in second last layer
                i = self.gnn_layer - 2
                cross_graph = self.layers[f'cross_graph_{i}']
                emb1 = cross_graph(jt.concat((emb1_0, jt.bmm(s, emb2_0)), dim=-1))
                emb2 = cross_graph(jt.concat((emb2_0, jt.bmm(s.transpose(1, 2), emb1_0)), dim=-1))

                # last layer
                i = self.gnn_layer - 1
                gnn_layer = self.layers[f'gnn_layer_{i}']
                emb1, emb2 = gnn_layer([A1, emb1], [A2, emb2])
                affinity = self.layers[f'affinity_{i}']
                s = affinity(emb1, emb2)
                s = _sinkhorn_func(s, n1, n2)

        return s


pca_gm_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc_jittor.pt',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1k4eBJ869uX7sN9TVTe67-8ZKRffpeBu8',
             'https://www.dropbox.com/scl/fi/gc7ekhxdeump5znzv8nnz/pca_gm_voc_jittor.pt?rlkey=h9fe2d3cfn4r1fumvaqjjia16&dl=1'],
             '112bb91bd0ccc573c3a936c49416d79e'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_willow_jittor.pt',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=15R3mdOR99g1LuSyv2IikRmlvy06ub7GQ',
                'https://www.dropbox.com/scl/fi/1irrb63jdz30m2ebj3lhy/pca_gm_willow_jittor.pt?rlkey=njkwysy6eh89wa4jgrl54v767&dl=1'],
                '72f4decf48eb5e00933699518563035a'),
    'voc-all': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc-all_jittor.pt',
                 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=17QvlZRAFcPBslaMCax9BVmQpoFMUWv5I',
                 'https://www.dropbox.com/scl/fi/yl8vymg3rc52n4wxr50gh/pca_gm_voc-all_jittor.pt?rlkey=niypnhmsve6md495j59psqs37&dl=1'],
                 '65cdf9ab437fa37c18eac147cb490c8f')
}


def pca_gm(feat1, feat2, A1, A2, n1, n2,
           in_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
           network, pretrain):
    """
    Jittor implementation of PCA-GM
    """
    if feat1 is None:
        forward_pass = False
    else:
        forward_pass = True
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers)
        if pretrain:
            if pretrain in pca_gm_pretrain_path:
                url, md5 = pca_gm_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'pca_gm_{pretrain}_jittor.pt', url, md5)
                _load_model(network, filename) 
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {pca_gm_pretrain_path.keys()}')

    if forward_pass:
        batch_size = feat1.shape[0]
        if n1 is None:
            n1 = jt.Var([feat1.shape[1]] * batch_size)
        if n2 is None:
            n2 = jt.Var([feat2.shape[1]] * batch_size)
        result = network(feat1, feat2, A1, A2, n1, n2, -1, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


ipca_gm_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ipca_gm_voc_jittor.pt',
            'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1B5W83efRL50C1D348xPJHaHoEXpAfKTL',
            'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/jittor_backend/ipca_gm_voc_jittor.pt'],
            '3a6dc7948c75d2e31781847941b5f2f6'),
    
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ipca_gm_willow_jittor.pt',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1iHSAY0d7Ufw9slYQjD_dEMkUB8SQM0kO',
                'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/jittor_backend/ipca_gm_willow_jittor.pt'],
                '5a1a5b783b9e7ba51579b724a26dccb4'),
}


def ipca_gm(feat1, feat2, A1, A2, n1, n2,
           in_channel, hidden_channel, out_channel, num_layers, cross_iter, sk_max_iter, sk_tau,
           network, pretrain):
    """
    Jittor implementation of IPCA-GM
    """
    if feat1 is None:
        forward_pass = False
    else:
        forward_pass = True
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers, cross_iter)
        if pretrain:
            if pretrain in ipca_gm_pretrain_path:
                url, md5 = ipca_gm_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'ipca_gm_{pretrain}_jittor.pt', url, md5)
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {ipca_gm_pretrain_path.keys()}')
    
    if forward_pass:
        batch_size = feat1.shape[0]
        if n1 is None:
            n1 = jt.Var([feat1.shape[1]] * batch_size)
        if n2 is None:
            n2 = jt.Var([feat2.shape[1]] * batch_size)
        result = network(feat1, feat2, A1, A2, n1, n2, cross_iter, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


class CIE_Net(Sequential):
    """
    Jittor implementation of CIE graph matching network
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
                self.add_module('affinity_{}'.format(i), WeightedInnerProdAffinity(out_channel))
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            if i == self.gnn_layer - 2:  # only the second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), jt.nn.Linear(hidden_channel * 2, hidden_channel))
                self.add_module('affinity_{}'.format(i), WeightedInnerProdAffinity(hidden_channel))

    def execute(self, feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(sinkhorn,
                                           dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb1, emb2 = feat_node1, feat_node2
        emb_edge1, emb_edge2 = feat_edge1, feat_edge2
        for i in range(self.gnn_layer):
            gnn_layer = self.layers[f'gnn_layer_{i}']
            # during forward process, the network structure will not change
            emb1, emb2, emb_edge1, emb_edge2 = gnn_layer([A1, emb1, emb_edge1], [A2, emb2, emb_edge2])

            if i == self.gnn_layer - 2:
                affinity = self.layers[f'affinity_{i}']
                s = affinity(emb1, emb2)
                s = _sinkhorn_func(s, n1, n2)

                cross_graph = self.layers[f'cross_graph_{i}']
                new_emb1 = cross_graph(jt.concat((emb1, jt.bmm(s, emb2)), dim=-1))
                new_emb2 = cross_graph(jt.concat((emb2, jt.bmm(s.transpose(1, 2), emb1)), dim=-1))
                emb1 = new_emb1
                emb2 = new_emb2

        affinity = self.layers[f'affinity_{self.gnn_layer - 1}']
        s = affinity(emb1, emb2)
        s = _sinkhorn_func(s, n1, n2)
        return s


cie_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_voc_jittor.pt',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1jjzbtXne_ppdg7M2jWEpye8piURDVidY',
             'https://www.dropbox.com/scl/fi/rual5ozkfrbe3205lzui3/cie_voc_jittor.pt?rlkey=zio8gca7qg8mc5a3murl6mew9&dl=1'],
             'dc398a5885c5d5894ed6667103d2ff18'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_willow_jittor.pt',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=11ftNCYBGnjGpFM3__oTCpBhOBabSU1Rv',
                'https://www.dropbox.com/scl/fi/iyigyphvuil8ch7cura0n/cie_willow_jittor.pt?rlkey=44wgqd4njgwmj8qdbe9fjxnez&dl=1'],
                'bef2c341f605669ed4211e8ff7b1fe0b'),
}


def cie(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2,
        in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
        network, pretrain):
    """
    Jittor implementation of CIE
    """
    if feat_node1 is None:
        forward_pass = False
    else:
        forward_pass = True
    if network is None:
        network = CIE_Net(in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers)
        if pretrain:
            if pretrain in cie_pretrain_path:
                url, md5 = cie_pretrain_path[pretrain]
                filename = pygmtools.utils.download(f'cie_{pretrain}_jittor.pt', url, md5)
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {cie_pretrain_path.keys()}')

    if forward_pass:
        batch_size = feat_node1.shape[0]
        if n1 is None:
            n1 = jt.Var([feat_node1.shape[1]] * batch_size)
        if n2 is None:
            n2 = jt.Var([feat_node1.shape[1]] * batch_size)
        result = network(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


class NGM_Net(Sequential):
    """
    Jittor implementation of NGM network
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
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        # self.classifier = nn.Linear(gnn_channels[-1] + sk_emb, 1)
        self.add_module('classifier', nn.Linear(gnn_channels[-1] + sk_emb, 1))

    def execute(self, K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau):
        _sinkhorn_func = functools.partial(sinkhorn,
                                           dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb = v0
        A = (K != 0)
        emb_K = K.unsqueeze(-1)

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = self.layers[f'gnn_layer_{i}']
            emb_K, emb = gnn_layer(A, emb_K, emb, n1, n2, sk_func=_sinkhorn_func)

        classifier = self.layers['classifier']
        v = classifier(emb)
        s = v.view(v.shape[0], n2max, -1).transpose(1, 2)

        return _sinkhorn_func(s, n1, n2, dummy_row=True)


ngm_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_voc_jittor.pt',
             'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/jittor_backend/ngm_voc_jittor.pt',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1_KZQPR6msYsMXupfrAgGgXT-zUXaGtmL'],
             '1c01a48ee2095b70da270da9d862a8c0'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_willow_jittor.pt',
                'https://raw.githubusercontent.com/heatingma/pygmtools-pretrained-models/main/jittor_backend/ngm_willow_jittor.pt',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1sLI7iC9kUyWm3xeByHvAMx_Hux8VAuP7'],
                'c23821751c895f79bbd038fa426ce259'),
}


def ngm(K, n1, n2, n1max, n2max, x0, gnn_channels, sk_emb, sk_max_iter, sk_tau, network, return_network, pretrain):
    """
    Jittor implementation of NGM
    """
    if K is None:
        forward_pass = False
    else:
        forward_pass = True
    if network is None:
        network = NGM_Net(gnn_channels, sk_emb)
        if pretrain:
            if pretrain in ngm_pretrain_path:
                url, md5 = ngm_pretrain_path[pretrain]
                try:
                    filename = pygmtools.utils.download(f'ngm_{pretrain}_jittor.pt', url, md5)
                except:
                    filename = os.path.dirname(__file__) + f'/temp/ngm_{pretrain}_jittor.pt'
                _load_model(network, filename)
            else:
                raise ValueError(f'Unknown pretrain tag. Available tags: {ngm_pretrain_path.keys()}')

    if forward_pass:
        batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
        v0 = v0 / jt.mean(v0)
        result = network(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau)
    else:
        result = None
    return result, network


#############################################
#              Utils Functions              #
#############################################

def inner_prod_aff_fn(feat1, feat2):
    """
    Jittor implementation of inner product affinity function
    """
    return jt.matmul(feat1, feat2.transpose(1, 2))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    Jittor implementation of Gaussian affinity function
    """
    feat1 = feat1.unsqueeze(2)
    feat2 = feat2.unsqueeze(1)
    return jt.exp(-((feat1 - feat2) ** 2).sum(dim=-1) / sigma)


def build_batch(input, return_ori_dim=False):
    """
    Jittor implementation of building a batched Var
    """
    _check_data_type(input[0], 'input', True)

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
        padded_ts.append(jt.nn.pad(t, pad_pattern, 'constant', 0))

    if return_ori_dim:
        return jt.stack(padded_ts, dim=0), tuple([jt.int64(_) for _ in ori_shape])
    else:
        return jt.stack(padded_ts, dim=0)


def dense_to_sparse(dense_adj):
    """
    Jittor implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn, ori_shape = build_batch([jt.nonzero(a) for a in dense_adj], return_ori_dim=True)
    nedges = ori_shape[0]
    edge_weight = build_batch([dense_adj[b][(conn[b, :, 0], conn[b, :, 1])] for b in range(batch_size)])
    return conn, edge_weight.unsqueeze(-1), nedges


def compute_affinity_score(X, K):
    """
    Jittor implementation of computing affinity score
    """
    b, n, _ = X.size()
    vx = X.transpose(1, 2).reshape(b, -1, 1)  # (b, n*n, 1)
    vxt = vx.transpose(1, 2)  # (b, 1, n*n)
    affinity = jt.bmm(jt.bmm(vxt, K), vx)
    return affinity


def to_numpy(input):
    """
    Jittor function to_numpy
    """
    return input.detach().numpy()

def from_numpy(input, device=None):
    """
    Jittor function from_numpy
    """
    return jt.Var(input)


def generate_isomorphic_graphs(node_num, graph_num, node_feat_dim):
    """
    Jittor implementation of generate_isomorphic_graphs
    """
    X_gt = jt.zeros((graph_num, node_num, node_num))
    X_gt[0, jt.arange(0, node_num, dtype=jt.int64), jt.arange(0, node_num, dtype=jt.int64)] = 1
    for i in range(graph_num):
        if i > 0:
            X_gt[i, jt.arange(0, node_num, dtype=jt.int64), jt.randperm(node_num)] = 1
    joint_X = X_gt.reshape(graph_num * node_num, node_num)
    X_gt = jt.matmul(joint_X, joint_X.t())
    X_gt = X_gt.reshape(graph_num, node_num, graph_num, node_num).permute(0, 2, 1, 3)
    A0 = jt.rand(node_num, node_num)
    A0[jt.arange(node_num), jt.arange(node_num)] = 0
    As = [A0]
    for i in range(graph_num):
        if i > 0:
            As.append(jt.matmul(jt.matmul(X_gt[i, 0], A0), X_gt[0, i]))
    if node_feat_dim > 0:
        F0 = jt.rand(node_num, node_feat_dim)
        Fs = [F0]
        for i in range(graph_num):
            if i > 0:
                Fs.append(jt.matmul(X_gt[i, 0], F0))
        return jt.stack(As, dim=0), X_gt, jt.stack(Fs, dim=0)
    else:
        return jt.stack(As, dim=0), X_gt

def permutation_loss(pred_dsmat: Var, gt_perm: Var, n1: Var, n2: Var) -> Var:
    """
    Jittor implementation of permutation_loss
    """
    batch_num = pred_dsmat.shape[0]

    pred_dsmat = pred_dsmat.float32()

    if not jt.all((pred_dsmat >= 0) * (pred_dsmat <= 1)):
        raise ValueError("pred_dsmat contains invalid numerical entries.")
    if not jt.all((gt_perm >= 0) * (gt_perm <= 1)):
        raise ValueError("gt_perm contains invalid numerical entries.")

    if n1 is None:
        n1 = jt.Var([pred_dsmat.shape[1] for _ in range(batch_num)])
    if n2 is None:
        n2 = jt.Var([pred_dsmat.shape[2] for _ in range(batch_num)])

    loss = jt.Var(0.)
    n_sum = jt.zeros_like(loss)
    for b in range(batch_num):
        loss += jt.nn.bce_loss(
            pred_dsmat[b, 0:n1[b].item(), 0:n2[b].item()],
            gt_perm[b, 0:n1[b].item(), 0:n2[b].item()],
            size_average=False).sum()
        n_sum += n1[b]

    return loss / n_sum

def _get_shape(input):
    """
    Jittor implementation of _get_shape
    """
    return input.shape

def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = jt.full((batch_num,), n1max, dtype=jt.int)
    if n2 is None:
        n2 = jt.full((batch_num,), n2max, dtype=jt.int)
    if n1max is None:
        n1max = jt.max(n1).item()
    if n2max is None:
        n2max = jt.max(n2).item()

    if not n1max * n2max == n1n2:
        raise ValueError('the input size of K does not match with n1max * n2max!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = jt.zeros((batch_num, int(n1max), int(n2max)), dtype=K.dtype)
        for b in range(batch_num):
            x0[b, 0:int(n1[b].item()), 0:int(n2[b].item())] = jt.Var(1.) / (n1[b] * n2[b])
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


def _check_data_type(input: Var, var_name, raise_err):
    """
    Jittor implementation of _check_data_type
    """
    if raise_err and type(input) is not Var:
        raise ValueError(f'Expected Jittor Var{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}.')
    return type(input) is Var

def _check_shape(input, dim_num):
    """
    Jittor implementation of _check_shape
    """
    return len(input.shape) == dim_num

def _aff_mat_from_node_edge_aff(node_aff: Var, edge_aff: Var, connectivity1: Var, connectivity2: Var,
                                n1, n2, ne1, ne2):
    """
    Jittor implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = jt.max(jt.max(connectivity1, dim=-1), dim=-1) + 1
        if n2 is None:
            n2 = jt.max(jt.max(connectivity2, dim=-1), dim=-1) + 1
        if ne1 is None:
            ne1 = jt.Var([edge_aff.shape[1]] * batch_size)
        if ne2 is None:
            ne2 = jt.Var([edge_aff.shape[2]] * batch_size)
    else:
        dtype = node_aff.dtype
        batch_size = node_aff.shape[0]
        if n1 is None:
            n1 = jt.Var([node_aff.shape[1]] * batch_size)
        if n2 is None:
            n2 = jt.Var([node_aff.shape[2]] * batch_size)

    n1max = int(max(n1).item())
    n2max = int(max(n2).item())
    ks = []
    for b in range(batch_size):
        k = jt.zeros((n2max, n1max, n2max, n1max), dtype=dtype)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:int(ne1[b])]
            conn2 = connectivity2[b][:int(ne2[b])]

            edge_indices = jt.concat([conn1.repeat_interleave(int(ne2[b]), dim=0), conn2.repeat(int(ne1[b]), 1)], dim=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b, :int(ne1[b]), :int(ne2[b])].reshape(-1)
        k = k.reshape(n2max * n1max, n2max * n1max)
        # node-wise affinity
        if node_aff is not None:
            k[jt.arange(n2max * n1max), jt.arange(n2max * n1max)] = node_aff[b].transpose(0, 1).reshape(-1)
        ks.append(k)

    return jt.stack(ks, dim=0)

def _squeeze(input, dim):
    """
    Jittor implementation of _squeeze
    """
    return input.squeeze(dim)

def _unsqueeze(input, dim):
    """
    Jittor implementation of _unsqueeze
    """
    return input.unsqueeze(dim)

def _transpose(input, dim1, dim2):
    """
    Jittor implementaiton of _transpose
    """
    return input.transpose(dim1, dim2)

def _mm(input1, input2):
    """
    Jittor implementation of _mm
    """
    return jt.matmul(input1, input2)

def _save_model(model, path):
    """
    Save Jittor model to a given path
    """
    if isinstance(model, jt.nn.DataParallel):
        model = model.module

    jt.save(model.state_dict(), path)

def _load_model(model, path):
    """
    Load Jittor model from a given path. Unmatched keys shall be shown in jittor warning.
    """
    module = model
    module.load_state_dict(jt.load(path))
