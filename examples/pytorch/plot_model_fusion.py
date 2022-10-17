# coding: utf-8
"""
========================================
Model Fusion by Graph Matching
========================================
This example shows how to fuse different models into a single model by ``pygmtools``.
Model fusion aims to fuse multiple models into one, such that the fused model could have higher performance.
In this example, the given models are trained on MNIST data from different distributions, and the fused model 
should fuse the knowledge in both the input models and can reach higher accuracy when testing.
"""

# Author: Chang Liu
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 1

##############################################################################
# .. note::
#     The following solvers support are included in this example:
#
#     * :func:`~pygmtools.classic_solvers.sm` (classic solver)
#
#     * :func:`~pygmtools.linear_solvers.hungarian` (linear solver)
#

import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append("../data")
import utils
import pygmtools as pygm

pygm.BACKEND = 'pytorch'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##############################################################################
# Load the trained models to be fused
# ------------------------------------
#

model1 = utils.SimpleNet()
model2 = utils.SimpleNet()
model1.load_state_dict(torch.load('../data/example_model_fusion_1.dat', map_location=device))
model2.load_state_dict(torch.load('../data/example_model_fusion_2.dat', map_location=device))
model1.to(device)
model2.to(device)
test_dataset = torchvision.datasets.MNIST(
    root='../data/mnist_data',  # the directory to store the dataset
    train=False,  # the dataset is used to test
    transform=transforms.ToTensor(),  # the dataset is in the form of tensors
    download=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False)
print(model1)

##############################################################################
# Test the input models
# ------------------------------------
#

with torch.no_grad():
    n_correct1 = 0
    n_correct2 = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs1 = model1(images)
        outputs2 = model2(images)
        _, predictions1 = torch.max(outputs1, 1)
        _, predictions2 = torch.max(outputs2, 1)
        n_samples += labels.shape[0]
        n_correct1 += (predictions1 == labels).sum().item()
        n_correct2 += (predictions2 == labels).sum().item()
    acc1 = 100 * n_correct1 / n_samples
    acc2 = 100 * n_correct2 / n_samples

##############################################################################
# Fuse the input models
# ------------------------------------
# Start the time counter, set the ensemble step as 0.2, which denotes fused model = 80% model1 + 20% model2.
#

img = Image.open('../data/model_fusion.png')
plt.imshow(img)
plt.axis('off')
ensemble_step = 0.2
st_time = time.perf_counter()

##############################################################################
# Get the affinity (similarity) metrix between model1 and model2.
#

affinity, params = utils.graph_matching_fusion([model1, model2])

##############################################################################
# Align the channels of model1 & model2 by maximize the affinity (similarity) via graph matching algorithms.
#

n1 = params[0]
n2 = params[1]
solution = pygm.sm(affinity, n1, n2)
solution = pygm.hungarian(solution)

##############################################################################
# Get the average weights of the two model after alignment.
#

fused_weights = utils.align(solution, ensemble_step, [model1, model2], params)

##############################################################################
# Test the fused model
# ------------------------------------
#

fused_model = utils.SimpleNet()
state_dict = fused_model.state_dict()
for idx, (key, _) in enumerate(state_dict.items()):
    state_dict[key] = fused_weights[idx]
fused_model.load_state_dict(state_dict)
test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = fused_model(data)
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
test_loss /= len(test_loader.dataset)
end_time = time.perf_counter()

##############################################################################
# Output the results
# ------------------------------------
#

print(f'time consumed for model fusion: {end_time - st_time}')
gm_acc = 100. * correct / len(test_loader.dataset)
print(f'model1 accuracy = {acc1}%, model2 accuracy = {acc2}%')
print("fused model accuracy: {}%".format(gm_acc))
