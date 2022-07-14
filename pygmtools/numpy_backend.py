import scipy.special
import scipy.optimize
import numpy as np
from multiprocessing import Pool


#############################################
#     Linear Assignment Problem Solvers     #
#############################################


def hungarian(s: np.ndarray, n1: np.ndarray=None, n2: np.ndarray=None, nproc: int=1) -> np.ndarray:
    """
    numpy implementation of Hungarian algorithm
    """
    batch_num = s.shape[0]

    perm_mat = -s
    if n1 is None:
        n1 = [None] * batch_num
    if n2 is None:
        n2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            perm_mat = [_ for _ in perm_mat]
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    return perm_mat


def _hung_kernel(s: np.ndarray, n1=None, n2=None):
    """
    Hungarian kernel function by calling the linear sum assignment solver from Scipy.
    """
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    row, col = scipy.optimize.linear_sum_assignment(s[:n1, :n2])
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    return perm_mat


def sinkhorn(s: np.ndarray, nrows: np.ndarray=None, ncols: np.ndarray=None,
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

    # operations are performed on log_s
    s = s / tau

    if dummy_row:
        assert s.shape[2] >= s.shape[1]
        dummy_shape = list(s.shape)
        dummy_shape[1] = s.shape[2] - s.shape[1]
        ori_nrows = nrows
        nrows = ncols
        s = np.concatenate((s, np.full(dummy_shape, -float('inf'))), axis=1)
        for b in range(batch_size):
            s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
            s[b, nrows[b]:, :] = -float('inf')
            s[b, :, ncols[b]:] = -float('inf')

    if batched_operation:
        log_s = s

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = scipy.special.logsumexp(log_s, 2, keepdims=True)
                log_s = log_s - log_sum
                log_s[np.isnan(log_s)] = -float('inf')
            else:
                log_sum = scipy.special.logsumexp(log_s, 1, keepdims=True)
                log_s = log_s - log_sum
                log_s[np.isnan(log_s)] = -float('inf')

        ret_log_s = log_s
    else:
        ret_log_s = np.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), dtype=s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s = s[b, row_slice, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = scipy.special.logsumexp(log_s, 1, keepdims=True)
                    log_s = log_s - log_sum
                else:
                    log_sum = scipy.special.logsumexp(log_s, 0, keepdims=True)
                    log_s = log_s - log_sum

            ret_log_s[b, row_slice, col_slice] = log_s

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
    numpy implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
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

        # reweighted jump
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


#############################################
#              Utils Functions              #
#############################################


def inner_prod_aff_fn(feat1, feat2):
    """
    numpy implementation of inner product affinity function
    """
    return np.matmul(feat1, feat2.transpose((0, 2, 1)))


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
            ne2 = [edge_aff.shape[1]] * batch_size
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


def _check_data_type(input: np.ndarray):
    """
    numpy implementation of _check_data_type
    """
    if type(input) is not np.ndarray:
        raise ValueError(f'Expected numpy ndarray, but got {type(input)}. Perhaps the wrong backend?')


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
