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
    device = s.device
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

    perm_mat = mindspore.Tensor(perm_mat).to(device)

    return perm_mat