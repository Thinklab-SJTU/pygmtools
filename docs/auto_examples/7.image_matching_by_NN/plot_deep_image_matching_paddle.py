# coding: utf-8
"""
==================================================================================
Paddle Backend Example: Matching Image Keypoints by Graph Matching Neural Networks
==================================================================================

This example shows how to match image keypoints by neural network-based graph matching solvers.
These graph matching solvers are designed to match two individual graphs. The matched images
can be further passed to tackle downstream tasks.
"""

# Author: Runzhong Wang <runzhong.wang@sjtu.edu.cn>
#         Wenzheng Pan <pwz1121@sjtu.edu.cn>
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 3

##############################################################################
# .. note::
#     The following solvers are based on matching two individual graphs, and are included in this example:
#
#     * :func:`~pygmtools.neural_solvers.pca_gm` (neural network solver)
#
#     * :func:`~pygmtools.neural_solvers.ipca_gm` (neural network solver)
#
#     * :func:`~pygmtools.neural_solvers.cie` (neural network solver)
#
import paddle # paddle backend
from paddle.vision.models import vgg16
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc
import itertools
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
pygm.set_backend('paddle') # set default backend for pygmtools

paddle.device.set_device('cpu') # paddle sets device globally

##############################################################################
# Predicting Matching by Graph Matching Neural Networks
# ------------------------------------------------------
# In this section we show how to do predictions (inference) by graph matching neural networks.
# Let's take PCA-GM (:func:`~pygmtools.neural_solvers.pca_gm`) as an example.
#
# Load the images
# ^^^^^^^^^^^^^^^^
# Images are from the Willow Object Class dataset (this dataset also available with the Benchmark of ``pygmtools``,
# see :class:`~pygmtools.dataset.WillowObject`).
#
# The images are resized to 256x256.
#
obj_resize = (256, 256)
img1 = Image.open('../data/willow_duck_0001.png')
img2 = Image.open('../data/willow_duck_0002.png')
kpts1 = paddle.to_tensor(sio.loadmat('../data/willow_duck_0001.mat')['pts_coord'])
kpts2 = paddle.to_tensor(sio.loadmat('../data/willow_duck_0002.mat')['pts_coord'])
kpts1[0] = kpts1[0] * obj_resize[0] / img1.size[0]
kpts1[1] = kpts1[1] * obj_resize[1] / img1.size[1]
kpts2[0] = kpts2[0] * obj_resize[0] / img2.size[0]
kpts2[1] = kpts2[1] * obj_resize[1] / img2.size[1]
img1 = img1.resize(obj_resize, resample=Image.BILINEAR)
img2 = img2.resize(obj_resize, resample=Image.BILINEAR)
paddle_img1 = paddle.to_tensor(np.array(img1, dtype=np.float32) / 256).transpose((2, 0, 1)).unsqueeze(0) # shape: BxCxHxW
paddle_img2 = paddle.to_tensor(np.array(img2, dtype=np.float32) / 256).transpose((2, 0, 1)).unsqueeze(0) # shape: BxCxHxW

##############################################################################
# Visualize the images and keypoints
#
def plot_image_with_graph(img, kpt, A=None):
    plt.imshow(img)
    plt.scatter(kpt[0], kpt[1], c='w', edgecolors='k')
    if A is not None:
        for idx in paddle.nonzero(A, as_tuple=False):
            plt.plot((kpt[0, idx[0]], kpt[0, idx[1]]), (kpt[1, idx[0]], kpt[1, idx[1]]), 'k-')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Image 1')
plot_image_with_graph(img1, kpts1)
plt.subplot(1, 2, 2)
plt.title('Image 2')
plot_image_with_graph(img2, kpts2)

##############################################################################
# Build the graphs
# ^^^^^^^^^^^^^^^^^
# Graph structures are built based on the geometric structure of the keypoint set. In this example,
# we refer to `Delaunay triangulation <https://en.wikipedia.org/wiki/Delaunay_triangulation>`_.
#
def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = paddle.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

A1 = delaunay_triangulation(kpts1)
A2 = delaunay_triangulation(kpts2)

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
# Extract node features via CNN
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Deep graph matching solvers can be fused with CNN feature extractors, to build an end-to-end learning pipeline.
#
# In this example, let's adopt the deep graph solvers based on matching two individual graphs.
# The image features are based on two intermediate layers from the VGG16 CNN model, following
# existing deep graph matching papers (such as :func:`~pygmtools.neural_solvers.pca_gm`)
#
# Let's firstly fetch the VGG16 model:
#
vgg16_cnn = vgg16(batch_norm=True) # vgg16_bn

##############################################################################
# List of layers of VGG16:
#
print(vgg16_cnn.features)

##############################################################################
# Let's define the CNN feature extractor, which outputs the features of ``layer (30)`` and
# ``layer (37)``
#
class CNNNet(paddle.nn.Layer):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = paddle.nn.Sequential(*[_ for _ in vgg16_module.features[:31]])
        self.edge_layers = paddle.nn.Sequential(*[_ for _ in vgg16_module.features[31:38]])

    def forward(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global

##############################################################################
# Download pretrained CNN weights (from `ThinkMatch <https://github.com/Thinklab-SJTU/ThinkMatch>`_),
# load the weights and then extract the CNN features
#
cnn = CNNNet(vgg16_cnn)
path = pygm.utils.download('vgg16_pca_voc_paddle.pdparams', 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1rIb_fPx20a4Q1GGlUsF8lAY1XNCyGO6L')
cnn.set_dict(paddle.load(path))
with paddle.set_grad_enabled(False):
    feat1_local, feat1_global = cnn(paddle_img1)
    feat2_local, feat2_global = cnn(paddle_img2)

##############################################################################
# Normalize the features
#
def l2norm(node_feat):
    return paddle.nn.functional.local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)

feat1_local = l2norm(feat1_local)
feat1_global = l2norm(feat1_global)
feat2_local = l2norm(feat2_local)
feat2_global = l2norm(feat2_global)

##############################################################################
# Up-sample the features to the original image size and concatenate
#
feat1_local_upsample = paddle.nn.functional.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat1_global_upsample = paddle.nn.functional.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat2_local_upsample = paddle.nn.functional.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat2_global_upsample = paddle.nn.functional.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat1_upsample = paddle.concat((feat1_local_upsample, feat1_global_upsample), axis=1)
feat2_upsample = paddle.concat((feat2_local_upsample, feat2_global_upsample), axis=1)
num_features = feat1_upsample.shape[1]

##############################################################################
# Visualize the extracted CNN feature (dimensionality reduction via principle component analysis)
#
pca_dim_reduc = PCAdimReduc(n_components=3, whiten=True)
feat_dim_reduc = pca_dim_reduc.fit_transform(
    np.concatenate((
        feat1_upsample.transpose((0, 2, 3, 1)).reshape((-1, num_features)).numpy(),
        feat2_upsample.transpose((0, 2, 3, 1)).reshape((-1, num_features)).numpy()
    ), axis=0)
)
feat_dim_reduc = feat_dim_reduc / np.max(np.abs(feat_dim_reduc), axis=0, keepdims=True) / 2 + 0.5
feat1_dim_reduc = feat_dim_reduc[:obj_resize[0] * obj_resize[1], :]
feat2_dim_reduc = feat_dim_reduc[obj_resize[0] * obj_resize[1]:, :]

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Image 1 with CNN features')
plot_image_with_graph(img1, kpts1, A1)
plt.imshow(feat1_dim_reduc.reshape((obj_resize[1], obj_resize[0], 3)), alpha=0.5)
plt.subplot(1, 2, 2)
plt.title('Image 2 with CNN features')
plot_image_with_graph(img2, kpts2, A2)
plt.imshow(feat2_dim_reduc.reshape((obj_resize[1], obj_resize[0], 3)), alpha=0.5)

##############################################################################
# Extract node features by nearest interpolation
#
rounded_kpts1 = paddle.cast(paddle.round(kpts1), dtype='int64')
rounded_kpts2 = paddle.cast(paddle.round(kpts2), dtype='int64')

node1 = feat1_upsample.transpose((2, 3, 0, 1))[rounded_kpts1[1], rounded_kpts1[0]][:, 0]
node2 = feat2_upsample.transpose((2, 3, 0, 1))[rounded_kpts2[1], rounded_kpts2[0]][:, 0]

##############################################################################
# Call PCA-GM matching model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.neural_solvers.pca_gm` for the API reference.
#
X = pygm.pca_gm(node1, node2, A1, A2, pretrain='voc')
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by PCA-GM')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = paddle.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# Matching images with other neural networks
# -------------------------------------------
# The above pipeline also works for other deep graph matching networks. Here we give examples of
# :func:`~pygmtoools.neural_solvers.ipca_gm` and :func:`~pygmtoools.neural_solvers.cie`.
#
# Matching by IPCA-GM model
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.neural_solvers.ipca_gm` for the API reference.
#
path = pygm.utils.download('vgg16_ipca_voc_paddle.pdparams', 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1h_VEmlfMAeBszoR0DvMr6EPXdNVTfTgf')
cnn.set_dict(paddle.load(path))

with paddle.set_grad_enabled(False):
    feat1_local, feat1_global = cnn(paddle_img1)
    feat2_local, feat2_global = cnn(paddle_img2)

##############################################################################
# Normalize the features
#
def l2norm(node_feat):
    return paddle.nn.functional.local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)

feat1_local = l2norm(feat1_local)
feat1_global = l2norm(feat1_global)
feat2_local = l2norm(feat2_local)
feat2_global = l2norm(feat2_global)

##############################################################################
# Up-sample the features to the original image size and concatenate
#
feat1_local_upsample = paddle.nn.functional.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat1_global_upsample = paddle.nn.functional.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat2_local_upsample = paddle.nn.functional.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat2_global_upsample = paddle.nn.functional.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat1_upsample = paddle.concat((feat1_local_upsample, feat1_global_upsample), axis=1)
feat2_upsample = paddle.concat((feat2_local_upsample, feat2_global_upsample), axis=1)
num_features = feat1_upsample.shape[1]

##############################################################################
# Extract node features by nearest interpolation
#
rounded_kpts1 = paddle.cast(paddle.round(kpts1), dtype='int64')
rounded_kpts2 = paddle.cast(paddle.round(kpts2), dtype='int64')

node1 = feat1_upsample.transpose((2, 3, 0, 1))[rounded_kpts1[1], rounded_kpts1[0]][:, 0]
node2 = feat2_upsample.transpose((2, 3, 0, 1))[rounded_kpts2[1], rounded_kpts2[0]][:, 0]

##############################################################################
# Build edge features as edge lengths
#
kpts1_dis = (kpts1.unsqueeze(0) - kpts1.unsqueeze(1))
kpts1_dis = paddle.norm(kpts1_dis, p=2, axis=2).detach()
kpts2_dis = (kpts2.unsqueeze(0) - kpts2.unsqueeze(1))
kpts2_dis = paddle.norm(kpts2_dis, p=2, axis=2).detach()

Q1 = paddle.exp(-kpts1_dis / obj_resize[0])
Q2 = paddle.exp(-kpts2_dis / obj_resize[0])

##############################################################################
# Matching by IPCA-GM model
#
X = pygm.ipca_gm(node1, node2, A1, A2, pretrain='voc')
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by IPCA-GM')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = paddle.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# Matching by CIE model
# ^^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.neural_solvers.cie` for the API reference.
#
path = pygm.utils.download('vgg16_cie_voc_paddle.pdparams', 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=18MwP3nuMkqDiiwRd_y6rlFmtjKi9THb-')
cnn.set_dict(paddle.load(path))

with paddle.set_grad_enabled(False):
    feat1_local, feat1_global = cnn(paddle_img1)
    feat2_local, feat2_global = cnn(paddle_img2)

##############################################################################
# Normalize the features
#
def l2norm(node_feat):
    return paddle.nn.functional.local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)

feat1_local = l2norm(feat1_local)
feat1_global = l2norm(feat1_global)
feat2_local = l2norm(feat2_local)
feat2_global = l2norm(feat2_global)

##############################################################################
# Up-sample the features to the original image size and concatenate
#
feat1_local_upsample = paddle.nn.functional.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat1_global_upsample = paddle.nn.functional.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat2_local_upsample = paddle.nn.functional.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat2_global_upsample = paddle.nn.functional.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
feat1_upsample = paddle.concat((feat1_local_upsample, feat1_global_upsample), axis=1)
feat2_upsample = paddle.concat((feat2_local_upsample, feat2_global_upsample), axis=1)
num_features = feat1_upsample.shape[1]

##############################################################################
# Extract node features by nearest interpolation
#
rounded_kpts1 = paddle.cast(paddle.round(kpts1), dtype='int64')
rounded_kpts2 = paddle.cast(paddle.round(kpts2), dtype='int64')

node1 = feat1_upsample.transpose((2, 3, 0, 1))[rounded_kpts1[1], rounded_kpts1[0]][:, 0]
node2 = feat2_upsample.transpose((2, 3, 0, 1))[rounded_kpts2[1], rounded_kpts2[0]][:, 0]

##############################################################################
# Build edge features as edge lengths
#
kpts1_dis = (kpts1.unsqueeze(1) - kpts1.unsqueeze(2))
kpts1_dis = paddle.norm(kpts1_dis, p=2, axis=0).detach()
kpts2_dis = (kpts2.unsqueeze(1) - kpts2.unsqueeze(2))
kpts2_dis = paddle.norm(kpts2_dis, p=2, axis=0).detach()

Q1 = paddle.exp(-kpts1_dis / obj_resize[0]).unsqueeze(-1).cast('float32')
Q2 = paddle.exp(-kpts2_dis / obj_resize[0]).unsqueeze(-1).cast('float32')

##############################################################################
# Call CIE matching model
#
X = pygm.cie(node1, node2, A1, A2, Q1, Q2, pretrain='voc')
X = pygm.hungarian(X)

plt.figure(figsize=(8, 4))
plt.suptitle('Image Matching Result by CIE')
ax1 = plt.subplot(1, 2, 1)
plot_image_with_graph(img1, kpts1, A1)
ax2 = plt.subplot(1, 2, 2)
plot_image_with_graph(img2, kpts2, A2)
for i in range(X.shape[0]):
    j = paddle.argmax(X[i]).item()
    con = ConnectionPatch(xyA=kpts1[:, i], xyB=kpts2[:, j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="red" if i != j else "green")
    plt.gca().add_artist(con)

##############################################################################
# Training a deep graph matching model
# -------------------------------------
# In this section, we show how to build a deep graph matching model which supports end-to-end training.
# For the image matching problem considered here, the model is composed of a CNN feature extractor and
# a learnable matching module. Take the PCA-GM model as an example.
#
# .. note::
#     This simple example is intended to show you how to do the basic forward and backward pass when
#     training an end-to-end deep graph matching neural network. A 'more formal' deep learning pipeline
#     should involve asynchronized data loader, batched operations, CUDA support and so on, which are
#     all omitted in consideration of simplicity. You may refer to `ThinkMatch <https://github.com/Thinklab-SJTU/ThinkMatch>`_
#     which is a research protocol with all these advanced features.
#
# Let's firstly define the neural network model. By calling :func:`~pygmtools.utils.get_network`,
# it will simply return the network object.
#
class GMNet(paddle.nn.Layer):
    def __init__(self):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain=False) # fetch the network object
        self.cnn = CNNNet(vgg16_cnn)

    def forward(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)

        # upsample feature map
        feat1_local_upsample = paddle.nn.functional.interpolate(feat1_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_global_upsample = paddle.nn.functional.interpolate(feat1_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_local_upsample = paddle.nn.functional.interpolate(feat2_local, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat2_global_upsample = paddle.nn.functional.interpolate(feat2_global, (obj_resize[1], obj_resize[0]), mode='bilinear')
        feat1_upsample = paddle.concat((feat1_local_upsample, feat1_global_upsample), axis=1)
        feat2_upsample = paddle.concat((feat2_local_upsample, feat2_global_upsample), axis=1)

        # assign node features
        rounded_kpts1 = paddle.cast(paddle.round(kpts1), dtype='int64')
        rounded_kpts2 = paddle.cast(paddle.round(kpts2), dtype='int64')
        node1 = feat1_upsample.transpose((2, 3, 0, 1))[rounded_kpts1[1], rounded_kpts1[0]][:, 0]
        node2 = feat2_upsample.transpose((2, 3, 0, 1))[rounded_kpts2[1], rounded_kpts2[0]][:, 0]

        # PCA-GM matching layers
        X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        return X

model = GMNet()

##############################################################################
# Define optimizer
# ^^^^^^^^^^^^^^^^^
#
optim = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-3)

##############################################################################
# Forward pass
# ^^^^^^^^^^^^^
#
X = model(paddle_img1, paddle_img2, kpts1, kpts2, A1, A2)

##############################################################################
# Compute loss
# ^^^^^^^^^^^^^
# In this example, the ground truth matching matrix is a diagonal matrix. We calculate the loss function via
# :func:`~pygmtools.utils.permutation_loss`
#
X_gt = paddle.eye(X.shape[0])
loss = pygm.utils.permutation_loss(X, X_gt)
print(f'loss={loss.item():.4f}')

##############################################################################
# Backward Pass
# ^^^^^^^^^^^^^^
#
loss.backward()

##############################################################################
# Visualize the gradients
#
plt.figure(figsize=(4, 4))
plt.title('Gradient Sizes of PCA-GM and VGG16 layers')
plt.gca().set_xlabel('Layer Index')
plt.gca().set_ylabel('Average Gradient Size')
grad_size = []
for param in model.parameters():
    if param.grad is not None:
        grad_size.append(paddle.abs(param.grad).mean().item())
print(grad_size)
plt.stem(grad_size)

##############################################################################
# Update the model parameters. A deep learning pipeline should iterate the forward pass
# and backward pass steps until convergence.
#
optim.step()
optim.clear_grad()

##############################################################################
# .. note::
#     This example supports both GPU and CPU, and the online documentation is built by a CPU-only machine.
#     The efficiency will be significantly improved if you run this code on GPU.
#