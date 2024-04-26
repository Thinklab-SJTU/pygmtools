# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np
from multiprocessing import Pool

import pygmtools.utils
from pygmtools.numpy_backend import _hung_kernel


#############################################
#     Linear Assignment Problem Solvers     #
#############################################


def hungarian(s: tf.Tensor, n1: tf.Tensor=None, n2: tf.Tensor=None,
              unmatch1: tf.Tensor=None, unmatch2: tf.Tensor=None,
              nproc: int=1) -> tf.Tensor:
    """
    Tensorflow implementation of Hungarian algorithm
    """

    device = s.device
    batch_num = s.shape[0]

    with tf.device('/cpu:0'):
        perm_mat = tf.stop_gradient(s).numpy() * -1
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
            perm_mat = tnp.stack(mapresult.get())
    else:
        perm_mat = tnp.stack([_hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) for b in range(batch_num)])


    with tf.device(device):
        perm_mat = tf.convert_to_tensor(perm_mat)

    return perm_mat


def sinkhorn(s: tf.Tensor, nrows: tf.Tensor=None, ncols: tf.Tensor=None,
             unmatchrows: tf.Tensor=None, unmatchcols: tf.Tensor=None,
             dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False) -> tf.Tensor:
    """
    Tensorflow implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = tf.transpose(s, perm=[0, 2, 1])
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True

    if nrows is None:
        nrows = tf.constant([s.shape[1] for _ in range(batch_size)])
    if ncols is None:
        ncols = tf.constant([s.shape[2] for _ in range(batch_size)])

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if tf.reduce_any(transposed_batch):
        s_t = tf.transpose(s, perm=[0, 2, 1])
        s_t = tf.concat([
            s_t[:, :s.shape[1], :],
            tf.fill([batch_size, s.shape[1], s.shape[2] - s.shape[1]], -float('inf'))], axis=2)
        s = tf.where(tf.reshape(transposed_batch, [batch_size, 1, 1]), s_t, s)

        new_nrows = tf.where(transposed_batch, ncols, nrows)
        new_ncols = tf.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = tf.concat([
                unmatchrows,
                tf.fill([batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]], -float('inf'))],
            axis=1)
            new_unmatchrows = tf.where(tf.reshape(transposed_batch, [batch_size, 1]), unmatchcols, unmatchrows_pad)[:, :unmatchrows.shape[1]]
            new_unmatchcols = tf.where(tf.reshape(transposed_batch, [batch_size, 1]), unmatchrows_pad, unmatchcols)
            unmatchrows = new_unmatchrows
            unmatchcols = new_unmatchcols

    # operations are performed on log_s
    log_s = tf.Variable(s / tau)
    if unmatchrows is not None and unmatchcols is not None:
        unmatchrows = unmatchrows / tau
        unmatchcols = unmatchcols / tau

    if dummy_row:
        if not log_s.shape[2] >= log_s.shape[1]:
            raise RuntimeError('Error in Sinkhorn with dummy row')
        dummy_shape = list(log_s.shape)
        dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
        ori_nrows = nrows
        nrows = tf.constant(ncols)
        log_s = tf.Variable(tf.concat([log_s, tf.cast(tf.fill(dummy_shape, -float('inf')), dtype=log_s.dtype)], axis=1))
        if unmatchrows is not None:
            unmatchrows = tf.concat([unmatchrows, tf.cast(tf.fill((dummy_shape[0], dummy_shape[1]), -float('inf')), dtype=log_s.dtype)], axis=1)

        for b in range(batch_size):
            f = tf.cast(tf.fill([nrows[b] - ori_nrows[b], ncols[b]], -100), dtype=log_s.dtype)
            log_s[b, ori_nrows[b]:nrows[b], :ncols[b]].assign(f)

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = tf.Variable(tf.cast(tf.fill((log_s.shape[0], log_s.shape[1] + 1, log_s.shape[2] + 1), -float('inf')), dtype=log_s.dtype))
        new_log_s[:, :-1, :-1].assign(log_s)
        log_s = new_log_s
        for b in range(batch_size):
            log_s[b, :nrows[b], ncols[b]].assign(unmatchrows[b, :nrows[b]])
            log_s[b, nrows[b], :ncols[b]].assign(unmatchcols[b, :ncols[b]])
    row_mask = tf.Variable(tf.zeros([batch_size, log_s.shape[1], 1], dtype=tf.bool))
    col_mask = tf.Variable(tf.zeros([batch_size, 1, log_s.shape[2]], dtype=tf.bool))
    for b in range(batch_size):
        f = tf.fill([nrows[b]], True)
        row_mask[b, :nrows[b], 0].assign(f)
        f = tf.fill([ncols[b]], True)
        col_mask[b, 0, :ncols[b]].assign(f)
    if unmatchrows is not None and unmatchcols is not None:
        ncols += 1
        nrows += 1

    if batched_operation:
        for b in range(batch_size):
            f = tf.fill([log_s.shape[1]-nrows[b], log_s.shape[2]], -float('inf'))
            log_s[b, nrows[b]:, :].assign(f)
            f = tf.fill([log_s.shape[1], log_s.shape[2]-ncols[b]], -float('inf'))
            log_s[b, :, ncols[b]:].assign(f)

        for i in range(max_iter):
            if i % 2 == 0:
                log_sum = tf.reduce_logsumexp(log_s, 2, keepdims=True)
                log_s = tf.Variable(log_s - tf.where(row_mask, log_sum, tf.zeros_like(log_sum)))
                if tf.reduce_any(tf.math.is_nan(log_s)):
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

            else:
                log_sum = tf.reduce_logsumexp(log_s, 1, keepdims=True)
                log_s = tf.Variable(log_s - tf.where(col_mask, log_sum, tf.zeros_like(log_sum)))
                if tf.reduce_any(tf.math.is_nan(log_s)):
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

        ret_log_s = log_s
    else:
        ret_log_s = tf.Variable(tf.cast(tf.fill([batch_size, log_s.shape[1], log_s.shape[2]], -float('inf')), dtype=log_s.dtype))

        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    log_sum = tf.reduce_logsumexp(log_s_b, 1, keepdims=True)
                    log_s_b = log_s_b - tf.where(row_mask_b, log_sum, tf.zeros_like(log_sum))
                else:
                    log_sum = tf.reduce_logsumexp(log_s_b, 0, keepdims=True)
                    log_s_b = log_s_b - tf.where(col_mask_b, log_sum, tf.zeros_like(log_sum))

            ret_log_s[b, row_slice, col_slice].assign(log_s_b)

    if unmatchrows is not None and unmatchcols is not None:
        ncols -= 1
        nrows -= 1
        for b in range(batch_size):
            f = tf.cast(tf.fill([nrows[b]+1], -float('inf')), dtype=ret_log_s.dtype)
            ret_log_s[b, :nrows[b] + 1, ncols[b]].assign(f)
            f = tf.cast(tf.fill([ncols[b]], -float('inf')), dtype=ret_log_s.dtype)
            ret_log_s[b, nrows[b], :ncols[b]].assign(f)
        ret_log_s = tf.Variable(ret_log_s[:, :-1, :-1])

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = tf.Variable(ret_log_s[:, :-dummy_shape[1]])
        for b in range(batch_size):
            f = tf.cast(ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] - float('inf'), dtype=ret_log_s.dtype)
            ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]].assign(f)

    if tf.reduce_any(transposed_batch):
        s_t = tf.transpose(ret_log_s, perm=[0, 2, 1])
        s_t = tf.concat([
            s_t[:, :ret_log_s.shape[1], :],
            tf.fill((batch_size, ret_log_s.shape[1], ret_log_s.shape[2] - ret_log_s.shape[1]), -float('inf'))],
            axis=2)
        ret_log_s = tf.where(tf.reshape(transposed_batch, [batch_size, 1, 1]), s_t, ret_log_s)

    if transposed:
        ret_log_s = tf.transpose(ret_log_s, perm=[0, 2, 1])

    return tf.math.exp(ret_log_s)



#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################


def rrwm(K: tf.Tensor, n1: tf.Tensor, n2: tf.Tensor, n1max, n2max, x0: tf.Tensor,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> tf.Tensor:
    """
    Tensorflow implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
    d = tf.reduce_sum(K, axis=2, keepdims=True)
    dmax = tf.reduce_max(d, axis=1, keepdims=True)
    K = K / (dmax + tf.reduce_min(d) * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = tf.matmul(K, v)
        last_v = v
        n = tf.norm(v, ord=1, axis=1, keepdims=True)
        v = v / n

        # reweighted jump
        s = tf.transpose(tf.reshape(v, [batch_num, n2max, n1max]), [0, 2, 1])
        s = beta * s / tf.reduce_max(s, axis=[1,2], keepdims=True)
        v = alpha * tf.reshape(tf.transpose(sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True), [0, 2, 1]), [batch_num, n1n2, 1]) + \
            (1 - alpha) * v
        n = tf.norm(v, ord=1, axis=1, keepdims=True)
        v = tf.matmul(v, 1 / n)

        if tf.norm(v - last_v) < 1e-5:
            break

    return tf.transpose(tf.reshape(v, [batch_num, n2max, n1max]), [0, 2, 1])




def sm(K: tf.Tensor, n1: tf.Tensor, n2: tf.Tensor, n1max, n2max, x0: tf.Tensor,
       max_iter: int) -> tf.Tensor:
    """
    Tensorflow implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for i in range(max_iter):
        v = tf.matmul(K, v)
        n = tf.norm(v, ord=2, axis=1)
        v = tf.matmul(v, tf.reshape(1/n, [batch_num, 1, 1]))
        if tf.norm(v - vlast) < 1e-5:
            break
        vlast = v

    x = tf.transpose(tf.reshape(v, [batch_num, n2max, n1max]), [0, 2, 1])
    return x

def ipfp(K: tf.Tensor, n1: tf.Tensor, n2: tf.Tensor, n1max, n2max, x0: tf.Tensor,
         max_iter) -> tf.Tensor:
    """
    Tensorflow implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v
    best_v = v
    best_obj = -1

    def comp_obj_score(v1, K, v2):
        return tf.matmul(tf.matmul(tf.reshape(v1, [batch_num, 1, -1]), K), v2)

    for i in range(max_iter):
        cost = tf.transpose(tf.reshape(tf.matmul(K, v), [batch_num, n2max, n1max]), [0, 2, 1])
        binary_sol = hungarian(cost, n1, n2)
        binary_v = tf.reshape(tf.transpose(binary_sol, [0, 2, 1]), [batch_num, -1, 1])
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = - alpha / beta
        v = tf.where(tf.math.logical_or(beta >= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_obj = comp_obj_score(last_v, K, last_v)

        current_obj = comp_obj_score(binary_v, K, binary_v)
        best_v = tf.where(current_obj > best_obj, binary_v, best_v)
        best_obj = tf.where(current_obj > best_obj, current_obj, best_obj)

        if tf.reduce_max(tf.abs(last_v_obj - current_obj) / last_v_obj) < 1e-3:
            break
        last_v = v

    pred_x = tf.transpose(tf.reshape(best_v, [batch_num, n2max, n1max]), [0, 2, 1])
    return pred_x


def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = tf.cast(tf.fill((batch_num,), n1max), dtype=tf.int32)
    if n2 is None:
        n2 = tf.cast(tf.fill((batch_num,), n2max), dtype=tf.int32)
    if n1max is None:
        n1max = tf.reduce_max(n1)
    if n2max is None:
        n2max = tf.reduce_max(n2)

    if not n1max * n2max == n1n2:
        raise ValueError('the input size of K does not match with n1max * n2max!')

    # initialize x0 (also v0)
    if x0 is None:
        with tf.device(K.device):
            x0 = tf.Variable(tf.cast(tf.zeros([batch_num, n1max, n2max]), dtype=K.dtype))
        for b in range(batch_num):
            f = tf.cast(tf.fill([n1[b], n2[b]], 1) / (n1[b] * n2[b]), dtype=tf.float32)
            x0[b, 0:n1[b], 0:n2[b]].assign(f)
    v0 = tf.reshape(tf.transpose(x0, perm=[0, 2, 1]), [batch_num, n1n2, 1])

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


#############################################
#              Utils Functions              #
#############################################


def inner_prod_aff_fn(feat1, feat2):
    """
    Tensorflow implementation of inner product affinity function
    """
    return tf.matmul(feat1, tf.transpose(feat2,perm=[0, 2, 1]))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    Tensorflow implementation of Gaussian affinity function
    """
    feat1 = tf.expand_dims(feat1, 2)
    feat2 = tf.expand_dims(feat2, 1)
    return tf.math.exp(-tf.reduce_sum((feat1 - feat2) ** 2,axis=-1) / sigma)


def build_batch(input, return_ori_dim=False):
    """
    Tensorflow implementation of building a batched tensor
    """
    _check_data_type(input[0], 'input', True)
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
        pad_pattern = np.zeros((len(max_shape), 2), dtype=np.int64)
        pad_pattern[:, 1] = max_shape - np.array(t.shape)
        padded_ts.append(tnp.pad(t, pad_pattern, 'constant', constant_values=0))

    if return_ori_dim:
        with tf.device(device):
            return tf.stack(padded_ts, axis=0), tuple([tf.constant(_, dtype=tf.int64) for _ in ori_shape])
    else:
        return tf.stack(padded_ts, axis=0)

def dense_to_sparse(dense_adj):
    """
    Tensorflow implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn, ori_shape = build_batch([tf.where(a) for a in dense_adj], return_ori_dim=True)
    nedges = ori_shape[0]
    edge_weight = build_batch([tf.gather_nd(indices = conn[b, :, 0:2], params = dense_adj[b]) for b in range(batch_size)])
    return conn, tf.expand_dims(edge_weight, axis=-1), nedges


def compute_affinity_score(X, K):
    """
    Tensorflow implementation of computing affinity score
    """
    b, n, _ = X.shape
    vx = tf.reshape(tf.transpose(X, perm=[0, 2, 1]), [b, -1, 1])  # (b, n*n, 1)
    vxt = tf.transpose(vx,perm=[0, 2, 1])  # (b, 1, n*n)
    affinity = tf.matmul(tf.matmul(vxt, K), vx)
    return affinity


def to_numpy(input):
    """
    Tensorflow function to_numpy
    """
    with tf.device("/cpu:0"):
        return tf.stop_gradient(input).numpy()


def from_numpy(input, device):
    """
    Tensorflow function from_numpy
    """
    if device is None:
        return tf.convert_to_tensor(input)
    else:
        with tf.device(device):
            return tf.convert_to_tensor(input)


def generate_isomorphic_graphs(node_num, graph_num, node_feat_dim):
    """
    Tensorflow implementation of generate_isomorphic_graphs
    """
    indices = tf.stack([tf.fill(node_num, 0), tf.range(node_num), tf.range(node_num)], axis=1)
    updates = tf.fill(node_num, 1.)
    X_gt = tf.scatter_nd(indices, updates, [graph_num, node_num, node_num])
    for i in range(graph_num):
        if i > 0:
            indices = tf.stack([tf.fill(node_num, i), tf.range(node_num), tf.random.shuffle(tf.range(node_num))], axis=1)
            updates = tf.fill(node_num, 1.)
            X_gt = tf.tensor_scatter_nd_update(X_gt, indices, updates)
    joint_X = tf.reshape(X_gt, [graph_num * node_num, node_num])
    X_gt = tf.matmul(joint_X, joint_X, transpose_b=True)
    X_gt = tf.transpose(tf.reshape(X_gt, [graph_num, node_num, graph_num, node_num]), perm=[0, 2, 1, 3])
    A0 = tf.random.uniform(shape=[node_num, node_num])
    A0 = A0 - tf.linalg.diag(tf.linalg.diag_part(A0))
    As = [A0]
    for i in range(graph_num):
        if i > 0:
            As.append(tf.matmul(tf.matmul(X_gt[i, 0], A0), X_gt[0, i]))
    if node_feat_dim > 0:
        F0 = tf.random.uniform(shape=[node_num, node_feat_dim])
        Fs = [F0]
        for i in range(graph_num):
            if i > 0:
                Fs.append(tf.matmul(X_gt[i, 0], F0))
        return tf.stack(As, axis=0), X_gt, tf.stack(Fs, axis=0)
    else:
        return tf.stack(As, axis=0), X_gt


def permutation_loss(pred_dsmat: tf.Tensor, gt_perm: tf.Tensor, n1: tf.Tensor, n2: tf.Tensor) -> tf.Tensor:
    """
    Tensorflow implementation of permutation_loss
    """
    batch_num = pred_dsmat.shape[0]

    pred_dsmat = tf.cast(pred_dsmat, dtype=tf.float32)

    if not tf.reduce_all(tf.math.logical_and((pred_dsmat >= 0), (pred_dsmat <= 1))):
        raise ValueError("pred_dsmat contains invalid numerical entries.")
    if not tf.reduce_all(tf.math.logical_and((gt_perm >= 0), (gt_perm <= 1))):
        raise ValueError("gt_perm contains invalid numerical entries.")

    if n1 is None:
        n1 = tf.constant([pred_dsmat.shape[1] for _ in range(batch_num)])
    if n2 is None:
        n2 = tf.constant([pred_dsmat.shape[2] for _ in range(batch_num)])

    loss = tf.Variable(0.)
    n_sum = tf.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(n1[b]), slice(n2[b])]
        bce = tf.losses.BinaryCrossentropy()
        loss.assign_add(tf.cast(tf.size(gt_perm[batch_slice]), dtype=tf.float32) * bce(
            gt_perm[batch_slice],
            pred_dsmat[batch_slice]))
        n_sum += tf.cast(n1[b], dtype=n_sum.dtype)

    return loss / n_sum


def _aff_mat_from_node_edge_aff(node_aff: tf.Tensor, edge_aff: tf.Tensor, connectivity1: tf.Tensor, connectivity2: tf.Tensor,
                                n1, n2, ne1, ne2):
    """
    Tensorflow implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        device = edge_aff.device
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = tf.reduce_max(tf.reduce_max(connectivity1, axis=-1), axis=-1) + 1
        if n2 is None:
            n2 = tf.reduce_max(tf.reduce_max(connectivity2, axis=-1), axis=-1) + 1
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
        with tf.device(device):
            k = tf.zeros([n2max, n1max, n2max, n1max], dtype=dtype)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:ne1[b]]
            conn2 = connectivity2[b][:ne2[b]]
            edge_indices = tf.concat([tf.repeat(conn1, ne2[b], axis=0), tf.tile(conn2, [ne1[b], 1])], axis=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = tf.stack([edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]], axis=1) # indices: start_g2, start_g1, end_g2, end_g1
            updates = tf.reshape(edge_aff[b, :ne1[b], :ne2[b]], [-1])
            k = tf.tensor_scatter_nd_update(k, edge_indices, updates)
        k = tf.reshape(k, [n2max * n1max, n2max * n1max])
        # node-wise affinity
        if node_aff is not None:
            #k_diag = tf.linalg.diag_part(k)
            #k_diag = tf.reshape(tf.transpose(node_aff[b]), [-1])
            indices = tf.stack([tf.range(n2max * n1max), tf.range(n2max * n1max)], axis=1)
            updates = tf.reshape(tf.transpose(node_aff[b]),[-1])
            k = tf.tensor_scatter_nd_update(k, indices, updates)
        ks.append(k)

    return tf.stack(ks, axis=0)


def _check_data_type(input: tf.Tensor, var_name, raise_err):
    """
    Tensorflow implementation of _check_data_type
    """
    if raise_err and not tf.is_tensor(input):
        raise ValueError(f'Expected TensorFlow Tensor{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}.')
    return tf.is_tensor(input)


def _check_shape(input, dim_num):
    """
    Tensorflow implementation of _check_shape
    """
    return len(input.shape) == dim_num


def _get_shape(input):
    """
    Tensorflow implementation of _get_shape
    """
    return input.shape


def _squeeze(input, axis):
    """
    Tensorflow implementation of _squeeze
    """
    return tf.squeeze(input, axis)


def _unsqueeze(input, axis):
    """
    Tensorflow implementation of _unsqueeze
    """
    return tf.expand_dims(input,axis)


def _transpose(input, dim1, dim2):
    """
    Tensorflow implementaiton of _transpose
    """
    p = np.arange(len(input.shape))
    p[dim1], p[dim2] = p[dim2], p[dim1]
    return tf.transpose(input, perm=p)


def _mm(input1, input2):
    """
    Tensorflow implementation of _mm
    """
    return tf.matmul(input1, input2)


