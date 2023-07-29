# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import pygmtools as pygm

# Some test utils functions
def data_from_numpy(*data, device=None):
    return_list = []
    for d in data:
        return_list.append(pygm.utils.from_numpy(d, device))
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]


def data_to_numpy(*data):
    return_list = []
    for d in data:
        return_list.append(pygm.utils.to_numpy(d))
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]
    