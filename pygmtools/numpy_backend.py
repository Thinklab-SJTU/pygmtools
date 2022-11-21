import itertools
import functools
import scipy.special
import scipy.optimize
import numpy as np
from multiprocessing import Pool
import pygmtools.utils

#############################################
#     Linear Assignment Problem Solvers     #
#############################################


def hungarian(s: np.ndarray, n1: np.ndarray=None, n2: np.ndarray=None,
              unmatch1: np.ndarray=None, unmatch2: np.ndarray=None,
              nproc: int=1) -> np.ndarray:
    """
    numpy implementation of Hungarian algorithm
    """
    batch_num = s.shape[0]

    perm_mat = -s
    if n1 is None:
        n1 = [None] * batch_num
    if n2 is None:
        n2 = [None] * batch_num
    if unmatch1 is not None:
        unmatch1 = -unmatch1
    else:
        unmatch1 = [None] * batch_num
    if unmatch2 is not None:
        unmatch2 = -unmatch2
    else:
        unmatch2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            perm_mat = [_ for _ in perm_mat]
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2, unmatch1, unmatch2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) for b in range(batch_num)])

    return perm_mat


def _hung_kernel(s: np.ndarray, n1=None, n2=None, unmatch1=None, unmatch2=None):
    """
    Hungarian kernel function by calling the linear sum assignment solver from Scipy.
    """
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    if unmatch1 is not None and unmatch2 is not None:
        upper_left = s[:n1, :n2]
        upper_right = np.full((n1, n1), float('inf'))
        np.fill_diagonal(upper_right, unmatch1[:n1])
        lower_left = np.full((n2, n2), float('inf'))
        np.fill_diagonal(lower_left, unmatch2[:n2])
        lower_right = np.zeros((n2, n1))

        large_cost_mat = np.concatenate((np.concatenate((upper_left, upper_right), axis=1),
                                         np.concatenate((lower_left, lower_right), axis=1)), axis=0)

        row, col = scipy.optimize.linear_sum_assignment(large_cost_mat)
        valid_idx = np.logical_and(row < n1, col < n2)
        row = row[valid_idx]
        col = col[valid_idx]
    else:
        row, col = scipy.optimize.linear_sum_assignment(s[:n1, :n2])
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat


def sinkhorn(s: np.ndarray, nrows: np.ndarray=None, ncols: np.ndarray=None,
             unmatchrows: np.ndarray=None, unmatchcols: np.ndarray=None,
             dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False) -> np.ndarray:
    """
    numpy implementation of Sinkhorn algorithm
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
        nrows = np.array([s.shape[1] for _ in range(batch_size)], dtype=np.int)
    if ncols is None:
        ncols = np.array([s.shape[2] for _ in range(batch_size)], dtype=np.int)

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if np.any(transposed_batch):
        s_t = s.transpose((0, 2, 1))
        s_t = np.concatenate((
            s_t[:, :s.shape[1], :],
            np.full((batch_size, s.shape[1], s.shape[2]-s.shape[1]), -float('inf'))), axis=2)
        s = np.where(transposed_batch.reshape(batch_size, 1, 1), s_t, s)

        new_nrows = np.where(transposed_batch, ncols, nrows)
        new_ncols = np.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = np.concatenate((
                unmatchrows, np.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]), -float('inf'))),
            axis=1)
            new_unmatchrows = np.where(transposed_batch.reshape(batch_size, 1), unmatchcols, unmatchrows_pad)[:, :unmatchrows.shape[1]]
            new_unmatchcols = np.where(transposed_batch.reshape(batch_size, 1), unmatchrows_pad, unmatchcols)
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
        log_s = np.concatenate((log_s, np.full(dummy_shape, -float('inf'))), axis=1)
        if unmatchrows is not None:
            unmatchrows = np.concatenate((unmatchrows, np.full((dummy_shape[0], dummy_shape[1]), -float('inf'))), axis=1)
        for b in range(batch_size):
            log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = np.full((log_s.shape[0], log_s.shape[1]+1, log_s.shape[2]+1), -float('inf'))
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        for b in range(batch_size):
            log_s[b, :nrows[b], ncols[b]] = unmatchrows[b, :nrows[b]]
            log_s[b, nrows[b], :ncols[b]] = unmatchcols[b, :ncols[b]]
    row_mask = np.zeros((batch_size, log_s.shape[1], 1), dtype=bool)
    col_mask = np.zeros((batch_size, 1, log_s.shape[2]), dtype=bool)
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
                log_sum = scipy.special.logsumexp(log_s, 2, keepdims=True)
                log_s = log_s - np.where(row_mask, log_sum, np.zeros_like(log_sum))
                log_s[np.isnan(log_s)] = -float('inf')
            else:
                log_sum = scipy.special.logsumexp(log_s, 1, keepdims=True)
                log_s = log_s - np.where(col_mask, log_sum, np.zeros_like(log_sum))
                log_s[np.isnan(log_s)] = -float('inf')

        ret_log_s = log_s
    else:
        ret_log_s = np.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), dtype=log_s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = scipy.special.logsumexp(log_s_b, 1, keepdims=True)
                    log_s_b = log_s_b - np.where(row_mask_b, log_sum, np.zeros_like(log_sum))
                else:
                    log_sum = scipy.special.logsumexp(log_s_b, 0, keepdims=True)
                    log_s_b = log_s_b - np.where(col_mask_b, log_sum, np.zeros_like(log_sum))

            ret_log_s[b, row_slice, col_slice] = log_s_b

    if unmatchrows is not None and unmatchcols is not None:
        nrows -= 1
        ncols -= 1
        for b in range(batch_size):
            ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
            ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
        ret_log_s = ret_log_s[:, :-1, :-1]

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

    if np.any(transposed_batch):
        s_t = ret_log_s.transpose((0, 2, 1))
        s_t = np.concatenate((
            s_t[:, :ret_log_s.shape[1], :],
            np.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2]-ret_log_s.shape[1]), -float('inf'))), axis=2)
        ret_log_s = np.where(transposed_batch.reshape(batch_size, 1, 1), s_t, ret_log_s)

    if transposed:
        ret_log_s = ret_log_s.transpose((0, 2, 1))

    return np.exp(ret_log_s)
  
#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################

def rrwm(K: np.ndarray, n1: np.ndarray, n2: np.ndarray, n1max, n2max, x0: np.ndarray,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> np.ndarray:
    """
    numpy implementation of RRWM algorithm. (Reweighted Random Walk Matching)
    """

    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)

    d = K.sum(axis=2, keepdims=True)
    dmax = d.max(axis=1, keepdims=True)
    K = K / (dmax + d.min() * 1e-5) # d.min() * 1e-5 for numerical reasons
    v = v0

    for i in range(max_iter):
        # random walk
        v = np.matmul(K, v)
        last_v = v
        n = np.linalg.norm(v, ord=1, axis=1, keepdims=True)
        v = v / n

        s = v.reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))
        s = beta * s / np.amax(s, axis=(1, 2), keepdims=True)
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter).transpose((0, 2, 1)).reshape((batch_num, n1n2, 1)) + \
            (1 - alpha) * v
        n = np.linalg.norm(v, ord=1, axis=1, keepdims=True)
        v = np.matmul(v, 1 / n)
        
        if np.linalg.norm((v - last_v).squeeze(axis=-1), ord='fro') < 1e-5: 
            break

    return v.reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))

def sm(K: np.ndarray, n1: np.ndarray, n2: np.ndarray, n1max, n2max, x0: np.ndarray,
       max_iter: int) -> np.ndarray:
    """
    numpy implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0

    for i in range(max_iter):
        v = np.matmul(K, v)
        n = np.linalg.norm(v, ord=2, axis=1)
        v = np.matmul(v, (1 / n).reshape((batch_num, 1, 1)))
        if np.linalg.norm((v - vlast).squeeze(), ord='fro') < 1e-5:
            break
        vlast = v

    x = v.reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))
    return x

def ipfp(K: np.ndarray, n1: np.ndarray, n2: np.ndarray, n1max, n2max, x0: np.ndarray,
         max_iter) -> np.ndarray:
    """
    numpy implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v

    def comp_obj_score(v1, K, v2):
        """返回v1(行向量) * K * v2(列向量)"""
        return np.matmul(np.matmul(v1.reshape((batch_num, 1, -1)), K), v2)

    for i in range(max_iter):
        cost = np.matmul(K, v).reshape((batch_num, n2max, n1max)).transpose((0, 2, 1))
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.transpose((0, 2, 1)).reshape((batch_num, -1, 1))
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = alpha / beta
        v = np.where(np.logical_or(beta <= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_sol = comp_obj_score(last_v, K, last_v)
        if np.max(np.abs(
                last_v_sol - np.matmul(cost.reshape((batch_num, 1, -1)), binary_sol.reshape((batch_num, -1, 1)))
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
        n1 = np.full(batch_num, n1max, dtype=np.int)
    if n2 is None:
        n2 = np.full(batch_num, n2max, dtype=np.int)
    if n1max is None:
        n1max = np.max(n1)
    if n2max is None:
        n2max = np.max(n2)

    assert n1max * n2max == n1n2, 'the input size of K does not match with n1max * n2max!'

    # initialize x0 (also v0)
    if x0 is None:
        x0 = np.zeros((batch_num, n1max, n2max), dtype=K.dtype)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = 1. / (n1[b] * n2[b])
    v0 = x0.transpose((0, 2, 1)).reshape((batch_num, n1n2, 1))

    return batch_num, n1, n2, n1max, n2max, n1n2, v0

############################################
#      Multi-Graph Matching Solvers        #
############################################
def cao_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):

    m, n = num_graph, num_node
    param_lambda = lambda_init

    def _comp_aff_score(x, k):
        return np.expand_dims(np.expand_dims(pygmtools.utils.compute_affinity_score(x, k, backend='numpy'),axis=-1),axis=-1)

    for iter in range(max_iter):
        if iter >= iter_boost:
            param_lambda = np.min([param_lambda * lambda_step, lambda_max])
        # pair_con = get_batch_pc_opt(X)
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                con_ori = _get_single_pc_opt(X, i, j)
                # con_ori = torch.sqrt(pair_con[i, j])
                if iter < iter_boost:
                    score_ori = aff_ori
                else:
                    score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
                X_upt = X[i, j]
                for k in range(m):
                    X_combo = np.matmul(X[i, k], X[k, j])
                    aff_combo = _comp_aff_score(X_combo, K[i, j]) / norm
                    con_combo = _get_single_pc_opt(X, i, j, X_combo)
                    # con_combo = torch.sqrt(pair_con[i, k] * pair_con[k, j])
                    if iter < iter_boost:
                        score_combo = aff_combo
                    else:
                        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda
                    if score_combo > score_ori:
                        X_upt = X_combo
                X[i, j] = X_upt
                X[j, i] = X_upt.swapaxes(0,1)
    return X

def cao_fast_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    r"""
    Numpy implementation of CAO solver in fast config (mode="pc")

    :param K: affinity matrix, (m, m, n*n, n*n)
    :param X: initial matching, (m, m, n, n)
    :param num_graph: number of graphs, int
    :param num_node: number of nodes, int
    :return: X, (m, m, n, n)
    """
    m, n = num_graph, num_node
    param_lambda = lambda_init

    def _comp_aff_score(x, k):
        return np.expand_dims(np.expand_dims(pygmtools.utils.compute_affinity_score(x, k, backend='numpy'),axis=-1),axis=-1)

    mask1 = np.arange(m).reshape(m, 1).repeat(m,axis=1)
    mask2 = np.arange(m).reshape(1, m).repeat(m,axis=0)
    mask = (mask1 < mask2).astype(float)
    X_mask = mask.reshape(m, m, 1, 1)

    for iter in range(max_iter):
        if iter >= iter_boost:
            param_lambda = np.min([param_lambda * lambda_step, lambda_max])

        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        X1 = X.reshape(m, 1, m, n, n)
        X1 = np.tile(X1,(1, m, 1, 1, 1)).reshape(-1, n, n)  # X1[i,j,k] = X[i,k]
        X2 = X.reshape(1, m, m, n, n)
        X2 = np.tile(X2,(m, 1, 1, 1, 1)).swapaxes(1, 2).reshape(-1, n, n)  # X2[i,j,k] = X[k,j]
        X_combo = np.matmul(X1, X2).reshape(m, m, m, n, n) # X_combo[i,j,k] = X[i, k] * X[k, j]

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        pair_con = _get_batch_pc_opt(X)
        con_ori = np.sqrt(pair_con)

        K_repeat = np.repeat(K.reshape(m, m, 1, n * n, n * n),m,axis=2).reshape(-1, n * n, n * n)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K_repeat) / norm).reshape(m, m, m)
        con1 = pair_con.reshape(m, 1, m)
        con1 = np.tile(con1,(1, m, 1))  # con1[i,j,k] = pair_con[i,k]
        con2 = pair_con.reshape(1, m, m)
        con2 = np.tile(con2,(m, 1, 1)).swapaxes(1,2)  # con2[i,j,k] = pair_con[j,k]
        con_combo = np.sqrt(con1 * con2)

        if iter < iter_boost:
            score_ori = aff_ori
            score_combo = aff_combo
        else:
            score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
            score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda
        
        idx = np.argmax(score_combo,axis=-1)
        score_combo = np.max(score_combo, axis=-1)
        
        assert np.all(score_combo >= score_ori), np.min(score_combo - score_ori)
        X_upt = X_combo[mask1, mask2, idx, :, :]
        X = X_upt * X_mask + X_upt.swapaxes(0,1).swapaxes(2,3) * X_mask.swapaxes(0,1) + X * (1 - X_mask - X_mask.swapaxes(0, 1))
        assert np.all(X.swapaxes(0,1).swapaxes(2,3) == X)
    return X

def mgm_floyd_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node

    def _comp_aff_score(x, k):
        return np.expand_dims(np.expand_dims(pygmtools.utils.compute_affinity_score(x, k, backend='numpy'),axis=-1),axis=-1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, torch.mean(pair_aff).item(), torch.mean(get_batch_pc_opt(X)).item()
        # ))

        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                score_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                X_combo = np.matmul(X[i, k], X[k, j])
                score_combo = _comp_aff_score(X_combo, K[i, j]) / norm

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.swapaxes(0, 1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)
        for i in range(m):
            for j in range(m):
                if i >= j:
                    continue
                aff_ori = _comp_aff_score(X[i, j], K[i, j]) / norm
                con_ori = _get_single_pc_opt(X, i, j)
                # con_ori = torch.sqrt(pair_con[i, j])
                score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda

                X_combo = np.matmul(X[i, k], X[k, j])
                aff_combo = _comp_aff_score(X_combo, K[i, j]) / norm
                con_combo = _get_single_pc_opt(X, i, j, X_combo)
                # con_combo = torch.sqrt(pair_con[i, k] * pair_con[k, j])
                score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.swapaxes(0,1)
    return X

def mgm_floyd_fast_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node

    def _comp_aff_score(x, k):
        return np.expand_dims(np.expand_dims(pygmtools.utils.compute_affinity_score(x, k, backend='numpy'),axis=-1),axis=-1)

    mask1 = np.arange(m).reshape(m, 1).repeat(m,axis=1)
    mask2 = np.arange(m).reshape(1, m).repeat(m,axis=0)
    mask = (mask1 < mask2).astype(float)
    X_mask = mask.reshape(m, m, 1, 1)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        # print("iter:{} aff:{:.4f} con:{:.4f}".format(
        #     k, torch.mean(pair_aff).item(), torch.mean(get_batch_pc_opt(X)).item()
        # ))

        X1 = X[:, k].reshape(m, 1, n, n)
        X1 = np.tile(X1,(1, m, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n)
        X2 = np.tile(X2,(m, 1, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = np.matmul(X1, X2).reshape(m, m, n, n)

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)

        score_ori = aff_ori
        score_combo = aff_combo

        upt = (score_ori < score_combo).astype(float)
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.swapaxes(0,1).swapaxes(2, 3) * (1 - X_mask)

    for k in range(m):
        pair_aff = _comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - np.eye(m) * pair_aff
        norm = np.max(pair_aff)

        pair_con = _get_batch_pc_opt(X)

        X1 = X[:, k].reshape(m, 1, n, n)
        X1 = np.tile(X1,(1, m, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[i, k]
        X2 = X[k, :].reshape(1, m, n, n)
        X2 = np.tile(X2,(m, 1, 1, 1)).reshape(-1, n, n)  # X[i, j] = X[j, k]
        X_combo = np.matmul(X1, X2).reshape(m, m, n, n)

        aff_ori = (_comp_aff_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)
        aff_combo = (_comp_aff_score(X_combo.reshape(-1, n, n), K.reshape(-1, n * n, n * n)) / norm).reshape(m, m)

        con_ori = np.sqrt(pair_con)
        con1 = pair_con[:, k].reshape(m, 1).repeat(m,axis=1)
        con2 = pair_con[k, :].reshape(1, m).repeat(m,axis=0)
        con_combo = np.sqrt(con1 * con2)

        score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
        score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda

        upt = (score_ori < score_combo).astype(float)
        upt = (upt * mask).reshape(m, m, 1, 1)
        X = X * (1.0 - upt) + X_combo * upt
        X = X * X_mask + X.swapaxes(0,1).swapaxes(2, 3) * (1 - X_mask)
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
    X1 = X[i, :].reshape(-1, n, n)
    X2 = X[:, j].reshape(-1, n, n)
    X_combo = np.matmul(X1, X2)
    pair_con = 1 - np.sum(np.abs(Xij - X_combo)) / (2 * n * m)
    return pair_con

def _get_batch_pc_opt(X):
    """
    CAO/Floyd-fast helper function (compute consistency in batch)
    :param X: (m, m, n, n) all the matching results
    :return: (m, m) the consistency of X
    """
    m = X.shape[0]
    n = X.shape[2]
    X1 = X.reshape(m, 1, m, n, n)
    X1 = np.tile(X1,(1, m, 1, 1, 1)).reshape(-1, n, n)  # X1[i, j, k] = X[i, k]
    X2 = X.reshape(1, m, m, n, n)
    X2 = np.tile(X2,(m, 1, 1, 1, 1)).swapaxes(1,2).reshape(-1, n, n)  # X2[i, j, k] = X[k, j]
    X_combo = np.matmul(X1, X2).reshape(m, m, m, n, n)
    X_ori = X.reshape(m, m, 1, n, n)
    X_ori = np.tile(X_ori,(1, 1, m, 1, 1))
    pair_con = 1 - np.sum(np.abs(X_combo - X_ori), axis=(2, 3, 4)) / (2 * n * m)
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
    Numpy implementation of Graduated Assignment for Multi-Graph Matching (with compatibility for 2GM and clustering)
    """
    num_graphs = A.shape[0]
    if ns is None:
        ns = np.full((num_graphs,), A.shape[1], dtype='i4')
    n_indices = np.cumsum(ns, axis=0)

    # build a super adjacency matrix A
    supA = np.zeros((n_indices[-1], n_indices[-1]))
    for i in range(num_graphs):
        start_n = n_indices[i] - ns[i]
        end_n = n_indices[i]
        supA[start_n:end_n, start_n:end_n] = A[i, :ns[i], :ns[i]]

    # handle the type of n_univ
    if type(n_univ) is np.ndarray:
        n_univ = n_univ.item()

    # randomly init U
    if U0 is None:
        U0 = np.full((n_indices[-1], n_univ), 1 / n_univ)
        U0 += np.random.rand(n_indices[-1], n_univ) / 1000

    # init cluster_M if not given
    if cluster_M is None:
        cluster_M = np.ones((num_graphs, num_graphs))

    # reshape W into supW
    supW = np.zeros((n_indices[-1], n_indices[-1]))
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

    result = pygmtools.utils.MultiMatchingResult(True, 'numpy')

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
            UUt = np.matmul(U, U.T)
            lastUUt = UUt
            cluster_weight = np.repeat(cluster_M, ns.astype('i4'), axis=0)
            cluster_weight = np.repeat(cluster_weight, ns.astype('i4'), axis=1)
            quad = np.matmul(np.matmul(np.matmul(supA, UUt * cluster_weight), supA), U) * quad_weight * 2
            unary = np.matmul(supW * cluster_weight, U)
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
                    U_list.append(pygmtools.hungarian(V[n_start:n_end, :n_univ], backend='numpy'))
                    n_start = n_end
            elif projector == 'sinkhorn':
                if np.all(ns == ns[0]):
                    if ns[0] <= n_univ:
                        U_list.append(
                            sinkhorn(
                                V.reshape(num_graphs, -1, n_univ),
                                max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True
                            ).reshape(-1, n_univ))
                    else:
                        U_list.append(
                            sinkhorn(
                                V.reshape(num_graphs, -1, n_univ).swapaxes(1, 2),
                                max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True
                            ).swapaxes(1, 2).reshape(-1, n_univ))
                else:
                    V_list = []
                    n1 = []
                    n_start = 0
                    for n_end in n_indices:
                        V_list.append(V[n_start:n_end, :n_univ])
                        n1.append(n_end - n_start)
                        n_start = n_end
                    V_batch = build_batch(V_list)
                    n1 = np.ndarray(n1)
                    U = sinkhorn(V_batch, n1,
                                 max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True)
                    n_start = 0
                    for idx, n_end in enumerate(n_indices):
                        U_list.append(U[idx, :n_end - n_start, :])
                        n_start = n_end
            else:
                raise NameError('Unknown projecter name: {}'.format(projector))

            U = np.concatenate(U_list, axis=0)
            if num_graphs == 2:
                U[:ns[0], :] = np.eye(ns[0], n_univ)

            # calculate gap to discrete
            if projector == 'sinkhorn' and verbose:
                U_list_hung = []
                n_start = 0
                for n_end in n_indices:
                    U_list_hung.append(pygmtools.hungarian(V[n_start:n_end, :n_univ], backend='numpy'))
                    n_start = n_end
                U_hung = np.concatenate(U_list_hung, axis=0)
                diff = np.linalg.norm(np.matmul(U, U.t()) - lastUUt)
                print(f'tau={sinkhorn_tau:.3e} #iter={i}/{max_iter} '
                      f'gap to discrete: {np.mean(np.abs(U - U_hung)):.3e}, iter diff: {diff:.3e}')

            if projector == 'hungarian' and outlier_thresh > 0:
                U_hung = U
                UUt = np.matmul(U_hung, U_hung.t())
                cluster_weight = np.repeat(cluster_M, ns.astype('i4'), axis=0)
                cluster_weight = np.repeat(cluster_weight, ns.astype('i4'), axis=1)
                quad = np.linalg.multi_dot(supA, UUt * cluster_weight, supA, U_hung) * quad_weight * 2
                unary = np.matmul(supW * cluster_weight, U_hung)
                max_vals = (unary + quad).max(axis=1).values
                U = U * (unary + quad > outlier_thresh)
                if verbose:
                    print(f'hungarian #iter={i}/{max_iter} '
                          f'unary+quad score thresh={outlier_thresh:.3f}, #>thresh={np.sum(max_vals > outlier_thresh)}/{max_vals.shape[0]}'
                          f' min:{max_vals.min():.4f}, mean:{max_vals.mean():.4f}, median:{max_vals.median():.4f}, max:{max_vals.max():.4f}')

            if np.linalg.norm(np.matmul(U, U.T) - lastUUt) < converge_thresh:
                break

        if verbose: print('-' * 20)

        if i == max_iter - 1: # not converged
            if hung_iter:
                pass
            else:
                U_list = [pygmtools.hungarian(_, backend='numpy') for _ in U_list]
                U = np.concatenate(U_list, axis=0)
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
                U_list = [pygmtools.hungarian(_, backend='numpy') for _ in U_list]
                U = np.concatenate(U_list, axis=0)
                break

    return U

############################################
#          Neural Network Solvers          #
############################################

#############################################
#              Utils Functions              #
#############################################


def inner_prod_aff_fn(feat1, feat2):
    """
    numpy implementation of inner product affinity function
    """
    return np.matmul(feat1, feat2.swapaxes(1,2))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    numpy implementation of Gaussian affinity function
    """
    feat1 = np.expand_dims(feat1, axis=2)
    feat2 = np.expand_dims(feat2, axis=1)
    return np.exp(-((feat1 - feat2) ** 2).sum(axis=-1) / sigma)


def build_batch(input, return_ori_dim=False):
    """
    numpy implementation of building a batched np.ndarray
    """
    assert type(input[0]) == np.ndarray
    it = iter(input)
    t = next(it)
    max_shape = list(t.shape)
    ori_shape = tuple([[_] for _ in max_shape])
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
        return np.stack(padded_ts, axis=0), ori_shape
    else:
        return np.stack(padded_ts, axis=0)


def dense_to_sparse(dense_adj):
    """
    numpy implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn, ori_dim = build_batch([np.stack(np.nonzero(a), axis=1) for a in dense_adj], return_ori_dim=True)
    nedges = ori_dim[0]
    edge_weight = build_batch([dense_adj[b][(conn[b, :, 0], conn[b, :, 1])] for b in range(batch_size)])
    return conn, np.expand_dims(edge_weight, axis=-1), nedges

def compute_affinity_score(X, K):
    """
    Numpy implementation of computing affinity score
    """
    b, n, _ = X.shape
    vx = X.swapaxes(1,2).reshape(b, -1, 1)  # (b, n*n, 1)
    vxt = vx.swapaxes(1, 2)  # (b, 1, n*n)
    affinity = np.squeeze(np.squeeze(np.matmul(np.matmul(vxt, K), vx),axis=-1),axis=-1)
    return affinity

def to_numpy(input):
    """
    identity function
    """
    return input


def from_numpy(input, device):
    """
    identity function
    """
    return input

def generate_isomorphic_graphs(node_num, graph_num, node_feat_dim=0):
    """
    Numpy implementation of generate_isomorphic_graphs
    """
    X_gt = np.zeros((graph_num, node_num, node_num))
    X_gt[0, np.arange(0, node_num, dtype='i4'), np.arange(0, node_num, dtype='i4')] = 1
    for i in range(graph_num):
        if i > 0:
            X_gt[i, np.arange(0, node_num, dtype='i4'), np.random.permutation(node_num)] = 1
    joint_X = X_gt.reshape(graph_num * node_num, node_num)
    X_gt = np.matmul(joint_X, joint_X.T)
    X_gt = X_gt.reshape(graph_num, node_num, graph_num, node_num).transpose(0, 2, 1, 3)
    A0 = np.random.rand(node_num, node_num)
    A0[np.arange(node_num),np.arange(node_num)] = 0
    As = [A0]
    for i in range(graph_num):
        if i > 0:
            As.append(np.matmul(np.matmul(X_gt[i, 0], A0), X_gt[0, i]))
    if node_feat_dim > 0:
        F0 = np.random.rand(node_num, node_feat_dim)
        Fs = [F0]
        for i in range(graph_num):
            if i > 0:
                Fs.append(np.matmul(X_gt[i, 0], F0))
        return np.stack(As,axis=0), X_gt, np.stack(Fs,axis=0)
    else:
        return np.stack(As,axis=0), X_gt
"""
def permutation_loss(pred_dsmat:np.ndarray, gt_perm: np.ndarray, n1: np.ndarray, n2:np.ndarray) -> np.ndarray:

    #Numpy implementation of permutation_loss

    batch_num = pred_dsmat.shape[0]

    pred_dsmat = pred_dsmat.to(dtype='f')

    if not np.all((pred_dsmat >= 0) * (pred_dsmat <= 1)):
        raise ValueError("pred_dsmat contains invalid numerical entries.")
    if not np.all((gt_perm >= 0) * (gt_perm <= 1)):
        raise ValueError("gt_perm contains invalid numerical entries.")

    if n1 is None:
        n1 = np.array([pred_dsmat.shape[1] for _ in range(batch_num)])
    if n2 is None:
        n2 = np.array([pred_dsmat.shape[2] for _ in range(batch_num)])

    loss = np.array(0.)
    n_sum = np.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(n1[b]), slice(n2[b])]
        loss += array.nn.functional.binary_cross_entropy(
            pred_dsmat[batch_slice],
            gt_perm[batch_slice],
            reduction='sum')
        n_sum += n1[b].to(n_sum.dtype).to(pred_dsmat.device)

    return loss / n_sum
"""
def _aff_mat_from_node_edge_aff(node_aff: np.ndarray, edge_aff: np.ndarray, connectivity1: np.ndarray, connectivity2: np.ndarray,
                                n1, n2, ne1, ne2):
    """
    numpy implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = np.amax(connectivity1, axis=(1, 2)).copy() + 1
        if n2 is None:
            n2 = np.amax(connectivity2, axis=(1, 2)).copy() + 1
        if ne1 is None:
            ne1 = [edge_aff.shape[1]] * batch_size
        if ne2 is None:
            ne2 = [edge_aff.shape[2]] * batch_size
    else:
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
        k = np.zeros((n2max, n1max, n2max, n1max), dtype=dtype)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:ne1[b]]
            conn2 = connectivity2[b][:ne2[b]]
            edge_indices = np.concatenate([conn1.repeat(ne2[b], axis=0), np.tile(conn2, (ne1[b], 1))], axis=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b, :ne1[b], :ne2[b]].reshape(-1)
        k = k.reshape((n2max * n1max, n2max * n1max))
        # node-wise affinity
        if node_aff is not None:
            k[np.arange(n2max * n1max), np.arange(n2max * n1max)] = node_aff[b].T.reshape(-1)
        ks.append(k)

    return np.stack(ks, axis=0)


def _check_data_type(input: np.ndarray, var_name=None):
    """
    numpy implementation of _check_data_type
    """
    if type(input) is not np.ndarray:
        raise ValueError(f'Expected numpy ndarray{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}. Perhaps the wrong backend?')


def _check_shape(input: np.ndarray, dim_num):
    """
    numpy implementation of _check_shape
    """
    return len(input.shape) == dim_num


def _get_shape(input: np.ndarray):
    """
    numpy implementation of _get_shape
    """
    return input.shape


def _squeeze(input: np.ndarray, dim):
    """
    numpy implementation of _squeeze
    """
    return np.squeeze(input, axis=dim)


def _unsqueeze(input: np.ndarray, dim):
    """
    numpy implementation of _unsqueeze
    """
    return np.expand_dims(input, axis=dim)


def _transpose(input: np.ndarray, dim1, dim2):
    """
    numpy implementation of _transpose
    """
    return np.swapaxes(input, dim1, dim2)

def _mm(input1: np.ndarray, input2: np.ndarray):
    """
    numpy implementation of _mm
    """
    return np.matmul(input1, input2)
