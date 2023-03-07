# coding: utf-8
"""
========================================
Matching Image Keypoints by QAP Solvers
========================================

This example shows how to match image keypoints by graph matching solvers provided by ``pygmtools``.
These solvers follow the Quadratic Assignment Problem formulation and can generally work out-of-box.
The matched images can be further processed for other downstream tasks.
"""

# Author: Runzhong Wang <runzhong.wang@sjtu.edu.cn>
#         Wenzheng Pan <pwz1121@sjtu.edu.cn>
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 5

##############################################################################
# .. note::
#     The following solvers support QAP formulation, and are included in this example:
#
#     * :func:`~pygmtools.classic_solvers.rrwm` (classic solver)
#
#     * :func:`~pygmtools.classic_solvers.ipfp` (classic solver)
#
#     * :func:`~pygmtools.classic_solvers.sm` (classic solver)
#
#     * :func:`~pygmtools.neural_solvers.ngm` (neural network solver)
#
import numpy as np # numpy backend
import cv2 as cv
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
from PIL import Image
pygm.BACKEND = 'numpy' # set numpy as backend for pygmtools

##############################################################################
# Load the images
# ----------------
# Images are from the Willow Object Class dataset (this dataset also available with the Benchmark of ``pygmtools``,
# see :class:`~pygmtools.dataset.WillowObject`).
#
# The images are resized to 256x256.
#
obj_resize = (256, 256)
img1 = Image.open('../data/willow_duck_0001.png')
img2 = Image.open('../data/willow_duck_0002.png')
kpts1 = np.array(sio.loadmat('../data/willow_duck_0001.mat')['pts_coord'])
kpts2 = np.array(sio.loadmat('../data/willow_duck_0002.mat')['pts_coord'])
kpts1[0] = kpts1[0] * obj_resize[0] / img1.size[0]
kpts1[1] = kpts1[1] * obj_resize[1] / img1.size[1]
kpts2[0] = kpts2[0] * obj_resize[0] / img2.size[0]
kpts2[1] = kpts2[1] * obj_resize[1] / img2.size[1]
img1 = img1.resize(obj_resize, resample=Image.BILINEAR)
img2 = img2.resize(obj_resize, resample=Image.BILINEAR)

##############################################################################
# Visualize the images and keypoints
#
def plot_image_with_graph(img, kpt, A=None):
    plt.imshow(img)
    plt.scatter(kpt[0], kpt[1], c='w', edgecolors='k')
    if A is not None:
        for x, y in zip(np.nonzero(A)[0], np.nonzero(A)[1]):
            plt.plot((kpt[0, x], kpt[0, y]), (kpt[1, x], kpt[1, y]), 'k-')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Image 1')
plot_image_with_graph(img1, kpts1)
plt.subplot(1, 2, 2)
plt.title('Image 2')
plot_image_with_graph(img2, kpts2)

##############################################################################
# Build the graphs
# -----------------
# Graph structures are built based on the geometric structure of the keypoint set. In this example,
# we refer to `Delaunay triangulation <https://en.wikipedia.org/wiki/Delaunay_triangulation>`_.
#
def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.T)
    A = np.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

A1 = delaunay_triangulation(kpts1)
A2 = delaunay_triangulation(kpts2)

##############################################################################
# We encode the length of edges as edge features
#
A1 = ((np.expand_dims(kpts1, 1) - np.expand_dims(kpts1, 2)) ** 2).sum(axis=0) * A1
A1 = (A1 / A1.max()).astype(np.float32)
A2 = ((np.expand_dims(kpts2, 1) - np.expand_dims(kpts2, 2)) ** 2).sum(axis=0) * A2
A2 = (A2 / A2.max()).astype(np.float32)

##############################################################################
# Visualize the graphs
#
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Image 1 with Graphs')
plot_image_with_graph(img1, kpts1, A1)
plt.subplot(1, 2, 2)
plt.title('Image 2 with Graphs')
plot_image_with_graph(img2, kpts2, A2)

##############################################################################
# Extract node features
# ----------------------
# Let's adopt the SIFT method to extract node features.
#
np_img1 = np.array(img1, dtype=np.float32)
np_img2 = np.array(img2, dtype=np.float32)

def detect_sift(img):
    sift = cv.SIFT_create() 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img8bit = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    kpt = sift.detect(img8bit, None) 
    kpt, feat = sift.compute(img8bit, kpt) 
    return kpt, feat

sift_kpts1, feat1 = detect_sift(np_img1)
sift_kpts2, feat2 = detect_sift(np_img2)
sift_kpts1 = np.round(cv.KeyPoint_convert(sift_kpts1).T).astype(int)
sift_kpts2 = np.round(cv.KeyPoint_convert(sift_kpts2).T).astype(int)

##############################################################################
# Normalize the features
#
num_features = feat1.shape[1]
feat1 = feat1 / np.expand_dims(np.linalg.norm(feat1, axis=1), 1).repeat(128, axis=1)
feat2 = feat2 / np.expand_dims(np.linalg.norm(feat2, axis=1), 1).repeat(128, axis=1)

##############################################################################
# Extract node features by nearest interpolation
#
rounded_kpts1 = np.round(kpts1).astype(int)
rounded_kpts2 = np.round(kpts2).astype(int)

idx_1, idx_2 = [], []
for i in range(rounded_kpts1.shape[1]):
    y1 = np.where(sift_kpts1[1] == sift_kpts1[1][np.abs(sift_kpts1[1] - rounded_kpts1[1][i]).argmin()])
    y2 = np.where(sift_kpts2[1] == sift_kpts2[1][np.abs(sift_kpts2[1] - rounded_kpts2[1][i]).argmin()])
    t1 = sift_kpts1[0][y1]
    t2 = sift_kpts2[0][y2]
    x1 = np.where(sift_kpts1[0] == t1[np.abs(t1 - rounded_kpts1[0][i]).argmin()])
    x2 = np.where(sift_kpts2[0] == t2[np.abs(t2 - rounded_kpts2[0][i]).argmin()])
    idx_1.append(np.intersect1d(x1, y1)[0])
    idx_2.append(np.intersect1d(x2, y2)[0])

node1 = feat1[idx_1, :] # shape: NxC
node2 = feat2[idx_2, :] # shape: NxC

##############################################################################
# Build affinity matrix
# ----------------------
# We follow the formulation of Quadratic Assignment Problem (QAP):
#
# .. math::
#
#     &\max_{\mathbf{X}} \ \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})\\
#     s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}
#
# where the first step is to build the affinity matrix (:math:`\mathbf{K}`)
#
conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1) # set affinity function
K = pygm.utils.build_aff_mat(node1, edge1, conn1, node2, edge2, conn2, edge_aff_fn=gaussian_aff)

##############################################################################
# Visualization of the affinity matrix. For graph matching problem with :math:`N` nodes, the affinity matrix
# has :math:`N^2\times N^2` elements because there are :math:`N^2` edges in each graph.
#
# .. note::
#     The diagonal elements are node affinities, the off-diagonal elements are edge features.
#
plt.figure(figsize=(4, 4))
plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
plt.imshow(K, cmap='Blues')

##############################################################################
# Solve graph matching problem by RRWM solver
# -------------------------------------------
# See :func:`~pygmtools.classic_solvers.rrwm` for the API reference.
#
X = pygm.rrwm(K, kpts1.shape[1], kpts2.shape[1])

##############################################################################
# The output of RRWM is a soft matching matrix. Hungarian algorithm is then adopted to reach a discrete matching matrix.
#
X = pygm.hungarian(X)

##############################################################################
# Plot the matching
# ------------------
# The correct matchings are marked by green, and wrong matchings are marked by red. In this example, the nodes are
# ordered by their ground truth classes (i.e. the ground truth matching matrix is a diagonal matrix).
#
plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by RRWM')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = np.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# Solve by other solvers
# -----------------------
# We could also do a quick benchmarking of other solvers on this specific problem.
#
# IPFP solver
# ^^^^^^^^^^^
# See :func:`~pygmtools.classic_solvers.ipfp` for the API reference.
#
X = pygm.ipfp(K, kpts1.shape[1], kpts2.shape[1])

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by IPFP')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = np.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# SM solver
# ^^^^^^^^^^^
# See :func:`~pygmtools.classic_solvers.sm` for the API reference.
#
X = pygm.sm(K, kpts1.shape[1], kpts2.shape[1])
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by SM')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = np.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# NGM solver
# ^^^^^^^^^^^
# See :func:`~pygmtools.neural_solvers.ngm` for the API reference.
#
# .. note::
#     The NGM solvers are pretrained on a different problem setting, so their performance may seem inferior.
#     To improve their performance, you may change the way of building affinity matrices, or try finetuning
#     NGM on the new problem.
#
# The NGM solver pretrained on Willow dataset:
#
X = pygm.ngm(K, kpts1.shape[1], kpts2.shape[1], pretrain='willow')
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by NGM (willow pretrain)')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = np.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# The NGM solver pretrained on VOC dataset:
#
X = pygm.ngm(K, kpts1.shape[1], kpts2.shape[1], pretrain='voc')
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by NGM (voc pretrain)')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = np.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)
