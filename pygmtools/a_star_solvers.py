"""
**Neural network-based** graph matching solvers. It is recommended to integrate these networks as modules into your
existing deep learning pipeline (either supervised, unsupervised or reinforcement learning).
"""

# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib
import pygmtools
import numpy as np
from pygmtools.utils import NOT_IMPLEMENTED_MSG, from_numpy, \
    _check_shape, _get_shape, _unsqueeze, _squeeze, _check_data_type

def a_star(feat1, feat2, A1, A2, n1=None, n2=None,network=None, 
           return_network=False, pretrain='AIDS700nef',backend=None,**kwargs):
    if backend is None:
        backend = pygmtools.BACKEND
    non_batched_input = False
    if feat1 is not None: # if feat1 is None, this function skips the forward pass and only returns a network object
        for _ in (feat1, feat2, A1, A2):
            _check_data_type(_, backend)

        if all([_check_shape(_, 2, backend) for _ in (feat1, feat2, A1, A2)]):
            feat1, feat2, A1, A2 = [_unsqueeze(_, 0, backend) for _ in (feat1, feat2, A1, A2)]
            if type(n1) is int: n1 = from_numpy(np.array([n1]), backend=backend)
            if type(n2) is int: n2 = from_numpy(np.array([n2]), backend=backend)
            non_batched_input = True
        elif all([_check_shape(_, 3, backend) for _ in (feat1, feat2, A1, A2)]):
            non_batched_input = False
        else:
            raise ValueError(
                f'the input arguments feat1, feat2, A1, A2 are expected to be all 2-dimensional or 3-dimensional, got '
                f'feat1:{len(_get_shape(feat1, backend))}dims, feat2:{len(_get_shape(feat2, backend))}dims, '
                f'A1:{len(_get_shape(A1, backend))}dims, A2:{len(_get_shape(A2, backend))}dims!')

        if not (_get_shape(feat1, backend)[0] == _get_shape(feat2, backend)[0] == _get_shape(A1, backend)[0] == _get_shape(A2, backend)[0])\
                or not (_get_shape(feat1, backend)[1] == _get_shape(A1, backend)[1] == _get_shape(A1, backend)[2])\
                or not (_get_shape(feat2, backend)[1] == _get_shape(A2, backend)[1] == _get_shape(A2, backend)[2])\
                or not (_get_shape(feat1, backend)[2] == _get_shape(feat2, backend)[2]):
            raise ValueError(
                f'the input dimensions do not match. Got feat1:{_get_shape(feat1, backend)}, '
                f'feat2:{_get_shape(feat2, backend)}, A1:{_get_shape(A1, backend)}, A2:{_get_shape(A2, backend)}!')
    if n1 is not None: _check_data_type(n1, 'n1', backend)
    if n2 is not None: _check_data_type(n2, 'n2', backend)

    args = (feat1, feat2, A1, A2, n1, n2, network, pretrain)
    try:
        mod = importlib.import_module(f'pygmtools.{backend}_backend')
        fn = mod.a_star
    except (ModuleNotFoundError, AttributeError):
        raise NotImplementedError(
            NOT_IMPLEMENTED_MSG.format(backend)
        )

    result = fn(*args,**kwargs)
    match_mat = _squeeze(result[0], 0, backend) if non_batched_input else result[0]
    if return_network:
        return match_mat, result[1]
    else:
        return match_mat








                
      