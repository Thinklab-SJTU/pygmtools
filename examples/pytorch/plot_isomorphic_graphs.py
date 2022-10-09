# coding: utf-8
"""
========================================
Introduction: Matching Isomorphic Graphs
========================================
This example is an introduction to ``pygmtools`` which shows how to match isomorphic graphs.
Isomorphic graphs means graphs whose structure are identical, but the node correspondence is unknown.
"""

# Author: Runzhong Wang
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 6

import torch # pytorch backend
import pygmtools as pygm #
import networkx as nx # for plotting networks
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)

##############################################################################
# Generate two isomorphic graphs
# ------------------------------------
#

num_nodes = 10
X_gt = torch.zeros(num_nodes, num_nodes)
X_gt[torch.arange(0, num_nodes, dtype=torch.int64), torch.randperm(num_nodes)] = 1
A1 = torch.rand(num_nodes, num_nodes)
A1 = (A1 + A1.t() > 1.) * (A1 + A1.t()) / 2
A2 = torch.mm(torch.mm(X_gt.t(), A1), X_gt)
n1 = torch.tensor([num_nodes])
n2 = torch.tensor([num_nodes])

##############################################################################
# Visualize the graphs
# ----------------------
#
plt.figure(figsize=(8, 4))
G1 = nx.from_numpy_array(A1.numpy())
G2 = nx.from_numpy_array(A2.numpy())
pos1 = nx.spring_layout(G1)
pos2 = nx.spring_layout(G2)
plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G1, pos=pos1)
plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2)

##############################################################################
# These two graphs look dissimilar because they are not aligned. We then align these two graphs
# by graph matching.
#
# Build affinity matrix
# ----------------------
#
conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

plt.figure(figsize=(4, 4))
plt.title('Affinity Matrix')
plt.imshow(K.numpy(), cmap='Blues')

##############################################################################
# Solve graph matching problem by RRWM solver
# -------------------------------------------
#
# See :func:`~pygmtools.classic_solvers.rrwm`.
#
X = pygm.rrwm(K, n1, n2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('RRWM\'s Soft Matching Matrix')
plt.imshow(X.numpy(), cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')

##############################################################################
# Get the discrete matching matrix
# ---------------------------------
#
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
plt.imshow(X.numpy(), cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')

##############################################################################
# Align the original graphs
# --------------------------
#
plt.figure(figsize=(8, 4))
ax1 = plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G1, pos=pos1)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2)
for i in range(num_nodes):
    j = torch.argmax(X[i]).item()
    con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red")
    plt.gca().add_artist(con)

align_A2 = torch.mm(torch.mm(X, A2), X.t())
plt.figure(figsize=(8, 4))
ax1 = plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G1, pos=pos1)
ax2 = plt.subplot(1, 2, 2)
plt.title('Aligned Graph 2')
align_pos2 = {}
for i in range(num_nodes):
    j = torch.argmax(X[i]).item()
    align_pos2[j] = pos1[i]
    con = ConnectionPatch(xyA=pos1[i], xyB=align_pos2[j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red")
    plt.gca().add_artist(con)
nx.draw_networkx(G2, pos=align_pos2)

##############################################################################
# Other solvers are also available
# ---------------------------------
#
# Classic IPFP solver :func:`~pygmtools.classic_solvers.ipfp`:
#
X = pygm.ipfp(K, n1, n2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('IPFP Matching Matrix')
plt.imshow(X.numpy(), cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')

##############################################################################
# NGM neural network solver :func:`~pygmtools.neural_solvers.ngm`:
#
with torch.set_grad_enabled(False):
    X = pygm.ngm(K, n1, n2, pretrain='voc')
    X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('NGM Matching Matrix')
plt.imshow(X.numpy(), cmap='Blues')
plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')
