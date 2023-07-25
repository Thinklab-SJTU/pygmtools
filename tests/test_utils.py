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
from tqdm import tqdm

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


def test_networkx(graph_num_nodes):
    """
    Test the RRWM algorithm on pairs of isomorphic graphs using NetworkX
    
    :param graph_num_nodes: list, the numbers of nodes in the graphs to test
    """
    for num_node in tqdm(graph_num_nodes):
        As_b, X_gt = pygm.utils.generate_isomorphic_graphs(num_node)
        A1 = As_b[0]
        A2 = As_b[1]
        G1 = pygm.utils.to_networkx(A1)
        G2 = pygm.utils.to_networkx(A2)
        K = pygm.utils.build_aff_mat_from_networkx(G1, G2)
        X = pygm.rrwm(K, n1=num_node, n2=num_node)
        accuracy = (pygm.utils.to_numpy(pygm.hungarian(X, num_node, num_node)) * X_gt).sum() / X_gt.sum()
        assert accuracy == 1, f'When testing the networkx function with rrwm algorithm, there is an error in accuracy, \
                                and the accuracy is {accuracy}, the num_node is {num_node},.'

def test_graphml(graph_num_nodes):
    """
    Test the RRWM algorithm on pairs of isomorphic graphs using graphml
    
    :param graph_num_nodes: list, the numbers of nodes in the graphs to test
    """
    filename = 'examples/data/test_graphml_{}.graphml'
    filename_1 = filename.format(1)
    filename_2 = filename.format(2)
    for num_node in tqdm(graph_num_nodes):
        As_b, X_gt = pygm.utils.generate_isomorphic_graphs(num_node)
        A1 = As_b[0]
        A2 = As_b[1]
        pygm.utils.to_graphml(A1, filename_1)
        pygm.utils.to_graphml(A2, filename_2)
        K = pygm.utils.build_aff_mat_from_graphml(filename_1, filename_2)
        X = pygm.rrwm(K, n1=num_node, n2=num_node)
        accuracy = (pygm.utils.to_numpy(pygm.hungarian(X, num_node, num_node)) * X_gt).sum() / X_gt.sum()
        assert accuracy == 1, f'When testing the graphml function with rrwm algorithm, there is an error in accuracy, \
                                and the accuracy is {accuracy}, the num_node is {num_node},.'

if __name__ == '__main__':
    test_graphml(list(range(10, 30, 2)))
    test_networkx(list(range(10, 30, 2)))