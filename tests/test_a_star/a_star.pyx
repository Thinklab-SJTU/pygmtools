# distutils: language = c++
import torch
import numpy as np
cimport cython
cimport numpy as np
#from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

cdef extern from "priority_queue.hpp":
    cdef cppclass TreeNode:
        TreeNode()
        TreeNode(long)
        #TreeNode(vector[pair[long, long]], double, long)
        pair[vector[long], vector[long]] x_indices
        double gplsh
        long idx

    cdef cppclass tree_node_priority_queue:
        tree_node_priority_queue(...) # get Cython to accept any arguments and let C++ deal with getting them right
        void push(TreeNode)
        TreeNode top()
        void pop()
        bool empty()
        long size()

#cdef class TreeNode:
#    cdef public x_indices
#    cdef public double gplsh
#    cdef public long idx
#
#    def __init__(self, x_indices, double gplsh, long idx):
#        self.x_indices = x_indices
#        self.gplsh = gplsh
#        self.idx = idx

#cdef (double, long) key_func(TreeNode ele):
#cdef tuple key_func(TreeNode ele):
#    cdef double gplsh = ele.gplsh
#    cdef long idx = ele.idx
#    return gplsh, -idx

#cdef bool comp_func(TreeNode a, TreeNode b):
#    # note - no protection if allocated memory isn't long enough
#    return a.gplsh < a.gplsh

@cython.boundscheck(False)
@cython.wraparound(False)
def a_star(
        data,
        k,
        vector[long] ns_1,
        vector[long] ns_2,
        net_pred_func,
        heuristic_func,
        bool net_pred=True,
        long beam_width=0,
        double trust_fact=1.,
        long no_pred_size=0,
):
    # declare static dtypes
    cdef long batch_num, b, n1, n2, _n2, ns_1b, ns_2b, max_ns_1, max_ns_2, extra_n2_cnt
    cdef vector[long] tree_size
    cdef double h_p, g_p
    cdef vector[tree_node_priority_queue] open_set
    cdef tree_node_priority_queue cur_set
    cdef TreeNode selected, new_node
    cdef vector[bool] stop_flags
    cdef bool flag

    batch_num = k.shape[0]

    max_ns_1 = max(ns_1)
    max_ns_2 = max(ns_2)

    open_set = vector[tree_node_priority_queue](batch_num)
    tree_size = vector[long](batch_num)
    for b in range(batch_num):
        open_set[b].push(TreeNode())
    ret_x = torch.zeros(batch_num, max_ns_1+1, max_ns_2+1, device=k.device)
    x_dense = torch.zeros(max_ns_1+1, max_ns_2+1, device=k.device)
    stop_flags = vector[bool](batch_num, 0)
    while not all(stop_flags):
        for b in range(batch_num):
            ns_1b = ns_1[b]
            ns_2b = ns_2[b]

            if stop_flags[b] == 1:
                continue

            selected = open_set[b].top()
            open_set[b].pop()
            #selected_x_indices = torch.tensor(selected.x_indices, dtype=torch.long).reshape(-1, 2)
            if selected.idx == ns_1b:
                stop_flags[b] = 1
                #indices = selected_x_indices
                #v = torch.ones(indices.shape[0], device=k.device)
                #x = torch.sparse.FloatTensor(indices.t(), v, x_size).to_dense()
                ret_x[b][selected.x_indices] = 1
                continue

            if beam_width > 0:
                cur_set = tree_node_priority_queue()
            flag = False
            for n2 in range(ns_2b + 1):
                if n2 != ns_2b and is_in(n2, selected.x_indices.second):
                    continue
                if selected.idx + 1 == ns_1b:
                    flag = True
                    extra_n2_cnt = 0
                    for _n2 in range(ns_2b):
                        if _n2 != n2 and not is_in(_n2, selected.x_indices.second):
                            extra_n2_cnt += 1
                    new_node = TreeNode(ns_1b + extra_n2_cnt)
                    n1 = 0
                    for _ in range(selected.idx):
                        new_node.x_indices.first[n1] = selected.x_indices.first[n1]
                        new_node.x_indices.second[n1] = selected.x_indices.second[n1]
                        n1 += 1
                    new_node.x_indices.first[n1] = selected.idx
                    new_node.x_indices.second[n1] = n2
                    n1 += 1
                    for _n2 in range(ns_2b):
                        if _n2 != n2 and not is_in(_n2, selected.x_indices.second):
                            new_node.x_indices.first[n1] = ns_1b
                            new_node.x_indices.second[n1] = _n2
                            n1 += 1
                else:
                    new_node = TreeNode(selected.idx + 1)
                    n1 = 0
                    for _ in range(selected.idx):
                        new_node.x_indices.first[n1] = selected.x_indices.first[n1]
                        new_node.x_indices.second[n1] = selected.x_indices.second[n1]
                        n1 += 1
                    new_node.x_indices.first[n1] = selected.idx
                    new_node.x_indices.second[n1] = n2
                    n1 += 1

                x_dense[:] = 0
                x_dense[new_node.x_indices] = 1

                g_p = comp_ged(x_dense, k[b])

                if net_pred:
                    if selected.idx + 1 == ns_1b or trust_fact <= 0. or ns_1b - (selected.idx + 1) < no_pred_size:
                        h_p = 0
                    else:
                        h_p = net_pred_func(data, x_dense)
                else:
                    if selected.idx + 1 == ns_1b or trust_fact <= 0. or ns_1b - (selected.idx + 1) < no_pred_size:
                        h_p = 0
                    else:
                        h_p = heuristic_func(k[b], ns_1b, ns_2b, x_dense)

                new_node.gplsh = g_p + h_p * trust_fact
                new_node.idx = selected.idx + 1

                if beam_width > 0:
                    cur_set.push(new_node)
                else:
                    open_set[b].push(new_node)
                    tree_size[b] += 1
                if(flag):
                    break
                    
            if beam_width > 0:
                for i in range(min(beam_width, cur_set.size())):
                    open_set[b].push(cur_set.top())
                    cur_set.pop()
                    tree_size[b] += 1

    return ret_x, tree_size


cdef double comp_ged(_x, _k):
    return torch.mm(torch.mm(_x.reshape( 1, -1), _k), _x.reshape( -1, 1))

cdef bool is_in (long inp, vector[long] vec):
    cdef unsigned long i
    cdef bool ret = False
    for i in range(vec.size()):
        if inp == vec[i]:
            ret = True
            break
    return ret
