import numpy as np
import jittor as jt
from jittor import Var
import pygmtools.utils
from multiprocessing import Pool
from pygmtools.numpy_backend import _hung_kernel

#############################################
#     Linear Assignment Problem Solvers     #
#############################################

def hungarian(s: Var, n1: Var=None, n2: Var=None, nproc: int=1) -> Var:
    """
    Jittor implementation of Hungarian algorithm
    """
    # device = s.device
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

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack([_hung_kernel(perm_mat[b], n1[b], n2[b]) for b in range(batch_num)])

    perm_mat = jt.Var(perm_mat)

    return perm_mat

def sinkhorn(s: Var, nrows: Var=None, ncols: Var=None,
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
        transposed = True

    if nrows is None:
        # nrows = [s.shape[1] for _ in range(batch_size)]
        nrows = jt.Var([s.shape[1] for _ in range(batch_size)])
    if ncols is None:
        ncols = [s.shape[2] for _ in range(batch_size)]
        ncols = jt.Var([s.shape[2] for _ in range(batch_size)], device=s.device)

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

    # operations are performed on log_s
    s = s / tau

    if dummy_row:
        assert s.shape[2] >= s.shape[1]
        dummy_shape = list(s.shape)
        dummy_shape[1] = s.shape[2] - s.shape[1]
        ori_nrows = nrows
        nrows = ncols
        s = jt.concat((s, jt.full(dummy_shape, -float('inf'))), dim=1)
        for b in range(batch_size):
            s[b, int(ori_nrows[b]):int(nrows[b]), :int(ncols[b])] = -100
            s[b, int(nrows[b]):, :] = -float('inf')
            s[b, :, int(ncols[b]):] = -float('inf')

    if batched_operation:
        log_s = s

        for i in range(max_iter):
            if i % 2 == 0:
                m = log_s.max(2, keepdims=True)  #optimized logsumexp
                log_sum = jt.nn.logsumexp(log_s - m, 2, keepdim=True) + m
                log_s = log_s - log_sum
                log_s[jt.isnan(log_s)] = -float('inf')
            else:
                m = log_s.max(1, keepdims=True)
                log_sum = jt.nn.logsumexp(log_s - m, 1, keepdim=True) + m                
                log_s = log_s - log_sum
                log_s[jt.isnan(log_s)] = -float('inf')

        ret_log_s = log_s
    else:
        ret_log_s = jt.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), dtype=s.dtype)
        
        for b in range(batch_size):
            r,c = nrows[b],ncols[b]
            if not isinstance(nrows[b],int):
                r = int(nrows[b].item())
            if not isinstance(ncols[b],int):
                c = int(ncols[b].item())
            log_s = s[b, 0:r, 0:c]
            for i in range(max_iter):
                if i % 2 == 0:
                    m = log_s.max(1, keepdims=True)
                    log_sum = jt.nn.logsumexp(log_s - m, 1, keepdim=True) + m
                    log_s = log_s - log_sum
                else:
                    m = log_s.max(0, keepdims=True)
                    log_sum = jt.nn.logsumexp(log_s - m, 0, keepdim=True) + m
                    log_s = log_s - log_sum
            ret_log_s[b, 0:r, 0:c] = log_s

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
    dmax = d.max(dim=1, keepdims=True) #.values
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = jt.bmm(K, v)
        last_v = v
        n = jt.norm(v, p=1, dim=1, keepdim=True)
        v = v / n

        # reweighted jump
        s = v.view((batch_num, int(n2max), int(n1max))).transpose(1, 2)
        s = beta * s / s.max(dim=1, keepdims=True).max(dim=2, keepdims=True)
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter).transpose(1, 2).reshape(batch_num, n1n2, 1) + \
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
        # if jt.norm(v - vlast) < 1e-5: # Wrong with norm: lack of Frobenius norm 
        #     break
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

    def comp_obj_score(v1, K, v2):
        return jt.bmm(jt.bmm(v1.view(batch_num, 1, -1), K), v2)

    for i in range(max_iter):
        cost = jt.bmm(K, v).reshape((batch_num, int(n2max), int(n1max))).transpose(1, 2)
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.transpose(1, 2).view(batch_num, -1, 1)
        alpha = comp_obj_score(v, K, binary_v - v)  # + jt.mm(k_diag.view(1, -1), (binary_sol - v).view(-1, 1))
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = alpha / beta
        cond = jt.logical_or(beta <= 0, t0 >= 1)
        if cond.shape != binary_v.shape:
            cond = cond.expand(binary_v.shape)
        v = jt.where(cond, binary_v, v + t0 * (binary_v - v))
        last_v_sol = comp_obj_score(last_v, K, last_v)
        if jt.max(jt.abs(
                last_v_sol - jt.bmm(cost.reshape(batch_num, 1, -1), binary_sol.reshape(batch_num, -1, 1))
        ) / last_v_sol) < 1e-3:
            break
        last_v = v

    pred_x = binary_sol
    return pred_x


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
    assert type(input[0]) == jt.Var
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
    # return input.detach().cpu().numpy()
    # return input.detach().numpy()
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
    X_gt = jt.zeros(graph_num, node_num, node_num)
    X_gt[0, jt.arange(0, node_num, dtype=jt.int64), jt.arange(0, node_num, dtype=jt.int64)] = 1
    for i in range(graph_num):
        if i > 0:
            X_gt[i, jt.arange(0, node_num, dtype=jt.int64), jt.randperm(node_num)] = 1
    joint_X = X_gt.reshape(graph_num * node_num, node_num)
    X_gt = jt.mm(joint_X, joint_X.t())
    X_gt = X_gt.reshape(graph_num, node_num, graph_num, node_num).permute(0, 2, 1, 3)
    A0 = jt.rand(node_num, node_num)
    jt.diagonal(A0)[:] = 0
    As = [A0]
    for i in range(graph_num):
        if i > 0:
            As.append(jt.mm(jt.mm(X_gt[i, 0], A0), X_gt[0, i]))
    if node_feat_dim > 0:
        F0 = jt.rand(node_num, node_feat_dim)
        Fs = [F0]
        for i in range(graph_num):
            if i > 0:
                Fs.append(jt.mm(X_gt[i, 0], F0))
        return jt.stack(As, dim=0), X_gt, jt.stack(Fs, dim=0)
    else:
        return jt.stack(As, dim=0), X_gt


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
        n1 = jt.full((batch_num,), n1max, dtype=jt.int, device=K.device)
    if n2 is None:
        n2 = jt.full((batch_num,), n2max, dtype=jt.int, device=K.device)
    if n1max is None:
        n1max = jt.max(n1).item()
    if n2max is None:
        n2max = jt.max(n2).item()

    assert n1max * n2max == n1n2, 'the input size of K does not match with n1max * n2max!'

    # initialize x0 (also v0)
    if x0 is None:
        x0 = jt.zeros((batch_num, int(n1max), int(n2max)), dtype=K.dtype)#, device=K.device)
        for b in range(batch_num):
            x0[b, 0:int(n1[b].item()), 0:int(n2[b].item())] = jt.Var(1.) / (n1[b] * n2[b])
    v0 = x0.transpose(1, 2).reshape(batch_num, n1n2, 1)

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


def _check_data_type(input: Var):
    """
    Jittor implementation of _check_data_type
    """
    if type(input) is not Var:
        raise ValueError(f'Expected Jittor Var, but got {type(input)}. Perhaps the wrong backend?')

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
        # device = edge_aff.device
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = jt.max(jt.max(connectivity1, dim=-1).values, dim=-1).values + 1
        if n2 is None:
            n2 = jt.max(jt.max(connectivity2, dim=-1).values, dim=-1).values + 1
        if ne1 is None:
            ne1 = [edge_aff.shape[1]] * batch_size
        if ne2 is None:
            ne2 = [edge_aff.shape[1]] * batch_size
    else:
        # device = node_aff.device
        dtype = node_aff.dtype
        batch_size = node_aff.shape[0]
        if n1 is None:
            n1 = [node_aff.shape[1]] * batch_size
        if n2 is None:
            n2 = [node_aff.shape[2]] * batch_size

    # print(n1.max(),type(n1))
    n1max = int(max(n1).item())
    n2max = int(max(n2).item())
    ks = []
    for b in range(batch_size):
        k = jt.zeros((n2max, n1max, n2max, n1max), dtype=dtype)#, device=device)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:int(ne1[b])]
            conn2 = connectivity2[b][:int(ne2[b])]
            # print(conn1, ne2[b])
            edge_indices = jt.concat([conn1.repeat_interleave(int(ne2[b]), dim=0), conn2.repeat(int(ne1[b]), 1)], dim=1) # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3], edge_indices[:, 1]) # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b, :int(ne1[b]), :int(ne2[b])].reshape(-1)
        k = k.reshape(n2max * n1max, n2max * n1max)
        # node-wise affinity
        if node_aff is not None:
            # k_diag = jt.diag(k)
            # k_diag[:] = node_aff[b].transpose(0, 1).reshape(-1)
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
