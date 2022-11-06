from multiprocessing import Pool
import numpy as np
import mindspore

#############################################
#     Linear Assignment Problem Solvers     #
#############################################

from pygmtools.numpy_backend import _hung_kernel


def hungarian(s: mindspore.Tensor, n1: mindspore.Tensor=None, n2: mindspore.Tensor=None,
              unmatch1: mindspore.Tensor=None, unmatch2: mindspore.Tensor=None,
              nproc: int=1) -> mindspore.Tensor:
    """
    mindspore implementation of Hungarian algorithm
    """
    #device = s.device
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

    perm_mat = mindspore.Tensor(perm_mat)

    return perm_mat


# def sinkhorn(s: mindspore.Tensor, nrows: mindspore.Tensor=None, ncols: mindspore.Tensor=None,
#              unmatchrows: mindspore.Tensor=None, unmatchcols: mindspore.Tensor=None,
#              dummy_row: bool=False, max_iter: int=10, tau: float=1., batched_operation: bool=False) -> mindspore.Tensor:
#     """
#     mindspore implementation of Sinkhorn algorithm
#     """
#     batch_size = s.shape[0]
#
#     if s.shape[2] >= s.shape[1]:
#         transposed = False
#     else:
#         s = s.transpose(1, 2)
#         nrows, ncols = ncols, nrows
#         unmatchrows, unmatchcols = unmatchcols, unmatchrows
#         transposed = True
#
#     if nrows is None:
#         nrows = torch.tensor([s.shape[1] for _ in range(batch_size)], device=s.device)
#     if ncols is None:
#         ncols = torch.tensor([s.shape[2] for _ in range(batch_size)], device=s.device)
#
#     # ensure that in each dimension we have nrow < ncol
#     transposed_batch = nrows > ncols
#     if torch.any(transposed_batch):
#         s_t = s.transpose(1, 2)
#         s_t = torch.cat((
#             s_t[:, :s.shape[1], :],
#             torch.full((batch_size, s.shape[1], s.shape[2]-s.shape[1]), -float('inf'), device=s.device)), dim=2)
#         s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, s)
#
#         new_nrows = torch.where(transposed_batch, ncols, nrows)
#         new_ncols = torch.where(transposed_batch, nrows, ncols)
#         nrows = new_nrows
#         ncols = new_ncols
#
#         if unmatchrows is not None and unmatchcols is not None:
#             unmatchrows_pad = torch.cat((
#                 unmatchrows,
#                 torch.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]), -float('inf'), device=s.device)),
#             dim=1)
#             new_unmatchrows = torch.where(transposed_batch.view(batch_size, 1), unmatchcols, unmatchrows_pad)[:, :unmatchrows.shape[1]]
#             new_unmatchcols = torch.where(transposed_batch.view(batch_size, 1), unmatchrows_pad, unmatchcols)
#             unmatchrows = new_unmatchrows
#             unmatchcols = new_unmatchcols
#
#     # operations are performed on log_s
#     log_s = s / tau
#     if unmatchrows is not None and unmatchcols is not None:
#         unmatchrows = unmatchrows / tau
#         unmatchcols = unmatchcols / tau
#
#     if dummy_row:
#         assert log_s.shape[2] >= log_s.shape[1]
#         dummy_shape = list(log_s.shape)
#         dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
#         ori_nrows = nrows
#         nrows = ncols.clone()
#         log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf'), device=log_s.device, dtype=log_s.dtype)), dim=1)
#         if unmatchrows is not None:
#             unmatchrows = torch.cat((unmatchrows, torch.full((dummy_shape[0], dummy_shape[1]), -float('inf'), device=log_s.device, dtype=log_s.dtype)), dim=1)
#         for b in range(batch_size):
#             log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
#
#     # assign the unmatch weights
#     if unmatchrows is not None and unmatchcols is not None:
#         new_log_s = torch.full((log_s.shape[0], log_s.shape[1]+1, log_s.shape[2]+1), -float('inf'), device=log_s.device, dtype=log_s.dtype)
#         new_log_s[:, :-1, :-1] = log_s
#         log_s = new_log_s
#         for b in range(batch_size):
#             log_s[b, :nrows[b], ncols[b]] = unmatchrows[b, :nrows[b]]
#             log_s[b, nrows[b], :ncols[b]] = unmatchcols[b, :ncols[b]]
#     row_mask = torch.zeros(batch_size, log_s.shape[1], 1, dtype=torch.bool, device=log_s.device)
#     col_mask = torch.zeros(batch_size, 1, log_s.shape[2], dtype=torch.bool, device=log_s.device)
#     for b in range(batch_size):
#         row_mask[b, :nrows[b], 0] = 1
#         col_mask[b, 0, :ncols[b]] = 1
#     if unmatchrows is not None and unmatchcols is not None:
#         ncols += 1
#         nrows += 1
#
#     if batched_operation:
#         for b in range(batch_size):
#             log_s[b, nrows[b]:, :] = -float('inf')
#             log_s[b, :, ncols[b]:] = -float('inf')
#
#         for i in range(max_iter):
#             if i % 2 == 0:
#                 log_sum = torch.logsumexp(log_s, 2, keepdim=True)
#                 log_s = log_s - torch.where(row_mask, log_sum, torch.zeros_like(log_sum))
#                 assert not torch.any(torch.isnan(log_s))
#             else:
#                 log_sum = torch.logsumexp(log_s, 1, keepdim=True)
#                 log_s = log_s - torch.where(col_mask, log_sum, torch.zeros_like(log_sum))
#                 assert not torch.any(torch.isnan(log_s))
#
#         ret_log_s = log_s
#     else:
#         ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=log_s.device, dtype=log_s.dtype)
#
#         for b in range(batch_size):
#             row_slice = slice(0, nrows[b])
#             col_slice = slice(0, ncols[b])
#             log_s_b = log_s[b, row_slice, col_slice]
#             row_mask_b = row_mask[b, row_slice, :]
#             col_mask_b = col_mask[b, :, col_slice]
#
#             for i in range(max_iter):
#                 if i % 2 == 0:
#                     log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
#                     log_s_b = log_s_b - torch.where(row_mask_b, log_sum, torch.zeros_like(log_sum))
#                 else:
#                     log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
#                     log_s_b = log_s_b - torch.where(col_mask_b, log_sum, torch.zeros_like(log_sum))
#
#             ret_log_s[b, row_slice, col_slice] = log_s_b
#
#     if unmatchrows is not None and unmatchcols is not None:
#         ncols -= 1
#         nrows -= 1
#         for b in range(batch_size):
#             ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
#             ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
#         ret_log_s = ret_log_s[:, :-1, :-1]
#
#     if dummy_row:
#         if dummy_shape[1] > 0:
#             ret_log_s = ret_log_s[:, :-dummy_shape[1]]
#         for b in range(batch_size):
#             ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')
#
#     if torch.any(transposed_batch):
#         s_t = ret_log_s.transpose(1, 2)
#         s_t = torch.cat((
#             s_t[:, :ret_log_s.shape[1], :],
#             torch.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2]-ret_log_s.shape[1]), -float('inf'), device=log_s.device)), dim=2)
#         ret_log_s = torch.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)
#
#     if transposed:
#         ret_log_s = ret_log_s.transpose(1, 2)
#
#     return torch.exp(ret_log_s)