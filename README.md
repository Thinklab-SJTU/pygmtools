<img src="https://pygmtools.readthedocs.io/en/latest/_static/images/pygmtools_logo.svg" alt="pygmtools: Python Graph Matching Tools" width="800"/>

[![PyPi version](https://badgen.net/pypi/v/pygmtools/)](https://pypi.org/pypi/pygmtools/)
[![PyPI pyversions](https://img.shields.io/badge/dynamic/json?color=blue&label=python&query=info.requires_python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpygmtools%2Fjson)](https://pypi.python.org/pypi/pygmtools/)
[![Downloads](https://pepy.tech/badge/pygmtools)](https://pepy.tech/project/pygmtools)
[![Documentation Status](https://readthedocs.org/projects/pygmtools/badge/?version=latest)](https://pygmtools.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Thinklab-SJTU/pygmtools/branch/main/graph/badge.svg?token=Q68XTY0N0C)](https://codecov.io/gh/Thinklab-SJTU/pygmtools)
[![discord channel](https://img.shields.io/discord/1028701206526304317.svg?&color=blueviolet&label=discord)](https://discord.gg/8m6n7rRz9T)
[![QQ group](https://img.shields.io/badge/QQ%20group-696401889-blue)](https://qm.qq.com/cgi-bin/qm/qr?k=QolXYJn_M5ilDEM9e2jEjlPnJ02Ktabd&jump_from=webapi&authKey=6zG6D/Js4YF5h5zj778aO5MDKOXBwPFi8gQ4LsXJN8Hn1V8uCVGV81iT4J/FjPGT)
[![GitHub stars](https://img.shields.io/github/stars/Thinklab-SJTU/pygmtools.svg?style=social&label=Star&maxAge=8640)](https://GitHub.com/Thinklab-SJTU/pygmtools/stargazers/) 

-----------------------------------------

``pygmtools`` (Python Graph Matching Tools) provides graph matching solvers in Python and is easily accessible via:

```bash
$ pip install pygmtools
```

Official documentation: https://pygmtools.readthedocs.io

Source code: https://github.com/Thinklab-SJTU/pygmtools

Graph matching is a fundamental yet challenging problem in pattern recognition, data mining, and others.
Graph matching aims to find node-to-node correspondence among multiple graphs, by solving an NP-hard combinatorial
optimization problem.

Doing graph matching in Python used to be difficult, and this library wants to make researchers' lives easier. 
To highlight, ``pygmtools`` has the following features:

* *Support various solvers*, including traditional combinatorial solvers (including linear, quadratic, and multi-graph) 
  and novel deep learning-based solvers;
* *Support various backends*, including ``numpy`` which is universally accessible, and some state-of-the-art deep 
  learning architectures with GPU support: 
  ``pytorch``, ``paddle``, ``jittor``, ``Tensorflow``; 
* *Deep learning friendly*, the operations are designed to best preserve the gradient during computation and batched 
  operations support for the best performance.
  
## Installation

You can install the stable release on PyPI:

```bash
$ pip install pygmtools
```

or get the latest version by running:

```bash
$ pip install -U https://github.com/Thinklab-SJTU/pygmtools/archive/master.zip # with --user for user install (no root)
```

Now the pygmtools is available with the ``numpy`` backend.

The following packages are required, and shall be automatically installed by ``pip``:

```
Python >= 3.7
requests >= 2.25.1
scipy >= 1.4.1
Pillow >= 7.2.0
numpy >= 1.18.5
easydict >= 1.7
appdirs >= 1.4.4
tqdm >= 4.64.1
wget >= 3.2
```
  
## Available Graph Matching Solvers
This library offers user-friendly API for the following solvers:

* [Two-Graph Matching Solvers](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.classic_solvers.html)
    * Linear assignment solvers including the differentiable soft 
      [Sinkhorn algorithm](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.linear_solvers.sinkhorn.html) [1], 
      and the exact solver [Hungarian](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.linear_solvers.hungarian.html) [2].
    * Soft and differentiable quadratic assignment solvers, including [spectral graph matching](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.classic_solvers.sm.html) [3] 
      and [random-walk-based graph matching](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.classic_solvers.rrwm.html) [4].
    * Discrete (non-differentiable) quadratic assignment solver 
      [integer projected fixed point method](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.classic_solvers.ipfp.html) [5]. 
* [Multi-Graph Matching Solvers](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.multi_graph_solvers.html)
    * [Composition based Affinity Optimization (CAO) solver](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.multi_graph_solvers.cao.html) [6] 
      by optimizing the affinity score, meanwhile gradually infusing the consistency.
    * Multi-Graph Matching based on 
      [Floyd shortest path algorithm](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.multi_graph_solvers.mgm_floyd.html) [7].
    * [Graduated-assignment based multi-graph matching solver](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.multi_graph_solvers.gamgm.html) [8][9]
      by graduated annealing of Sinkhorn’s temperature.
* [Neural Graph Matching Solvers](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.neural_solvers.html)
    * Intra-graph and cross-graph embedding based neural graph matching solvers 
      [PCA-GM](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.neural_solvers.pca_gm.html) 
      and [IPCA-GM](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.neural_solvers.ipca_gm.html) [10]
      for matching individual graphs.
    * [Channel independent embedding (CIE)](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.neural_solvers.cie.html) [11]
      based neural graph matching solver for matching individual graphs.
    * [Neural graph matching solver (NGM)](https://pygmtools.readthedocs.io/en/latest/api/_autosummary/pygmtools.neural_solvers.ngm.html) [12]
      for the general quadratic assignment formulation.

## Available Backends
This library is designed to support multiple backends with the same set of API. 
Please follow the official instructions to install your backend.

The following backends are available:

* [Numpy](https://numpy.org/) (**default** backend, CPU only)

<img src="https://pygmtools.readthedocs.io/en/latest/_images/numpy_logo.png" alt="numpy logo" width="200"/>

* [PyTorch](https://pytorch.org/) (GPU friendly, deep learning friendly)

<img src="https://pygmtools.readthedocs.io/en/latest/_images/pytorch_logo.png" alt="pytorch logo" width="200"/>

* [Jittor](https://github.com/Jittor/Jittor) (GPU friendly, JIT support, deep learning friendly)

<img src="https://pygmtools.readthedocs.io/en/latest/_images/jittor_logo.png" alt="jittor logo" width="200"/>

* [PaddlePaddle](https://www.paddlepaddle.org.cn/en) (GPU friendly, deep learning friendly)

<img src="https://pygmtools.readthedocs.io/en/latest/_images/paddle_logo.png" alt="paddle logo" width="200"/>

* [Tensorflow](https://tensorflow.google.cn/) (GPU friendly, deep learning friendly)

<img src="https://pygmtools.readthedocs.io/en/latest/_images/tensorflow_logo.png" alt="tensorflow logo" width="200"/>

### Development status (0.3.6)

|                     | Numpy | PyTorch | Jittor | PaddlePaddle | Tensorflow | MindSpore |
| ------------------- | ----- | ------- | ------ | ------------ | ---------- | --------- |
| Linear Solvers      | ✔     | ✔       | ✔      | ✔            | ✔         | 🧱        |
| Classic Solvers     | ✔     | ✔       | ✔      | ✔            | ✔         | 🧱        |
| Multi-Graph Solvers | ✔    | ✔       | ✔      | ✔            | 📆         | 📆        |
| Neural Solvers      | ✔    | ✔       | ✔      | ✔           | 📆         | 📆        |
| Examples Gallery    | ✔    | ✔       | ✔      | ✔           | 📆         | 📆        |

✔: Supported; 🧱: Work in progress; 📆: Planned for future versions (contributions welcomed!).

For more details, please [read the documentation](https://pygmtools.readthedocs.io/en/latest/guide/get_started.html#install-other-backends).

## Pretrained Models

The library includes several neural network solvers. The pretrained models shall be automatically downloaded upon 
needed from Google Drive. If you are experiencing issues accessing Google Drive, please download the pretrained models
manually and put them at ``~/.cache/pygmtools`` (for Linux).

Available at:
[[google drive]](https://drive.google.com/drive/folders/1O7vkIW8QXBJsNsHUIRiSw91HJ_0FAzu_?usp=sharing)
[[baidu drive]](https://pan.baidu.com/s/1MvzfM52NJeLWx2JXbbc6HA?pwd=x8bv)

## The Deep Graph Matching Benchmark

``pygmtools`` is also featured with a standard data interface of several graph matching benchmarks. Please read 
[the corresponding documentation](https://pygmtools.readthedocs.io/en/latest/guide/benchmark.html) for details.

We also maintain a repository containing non-trivial implementation of deep graph matching models, please check out
[ThinkMatch](https://thinkmatch.readthedocs.io/) if you are interested!

## Chat with the Community

If you have any questions, or if you are experiencing any issues, feel free to raise an issue on GitHub. 

We also offer the following chat rooms if you are more comfortable with them:

* Discord (for English speakers): 
  
  [![discord](https://discordapp.com/api/guilds/1028701206526304317/widget.png?style=banner2)](https://discord.gg/8m6n7rRz9T)

* QQ Group (for Chinese speakers)/QQ群(中文用户): 696401889
  
  [![ThinkMatch/pygmtools交流群](http://pub.idqqimg.com/wpa/images/group.png)](https://qm.qq.com/cgi-bin/qm/qr?k=NlPuwwvaFaHzEWD8w7jSOTzoqSLIM80V&jump_from=webapi&authKey=chI2htrWDujQed6VtVid3V1NXEoJvwz3MVwruax6x5lQIvLsC8BmpmzBJOCzhtQd)

## Contributing
Any contributions/ideas/suggestions from the community is welcomed! Before starting your contribution, please read the
[Contributing Guide](https://github.com/Thinklab-SJTU/pygmtools/blob/main/CONTRIBUTING.md).

## Developers and Maintainers

``pygmtools`` is currently developed and maintained by members from [ThinkLab](http://thinklab.sjtu.edu.cn) at 
Shanghai Jiao Tong University. 

## References
<!--MLA style references-->

[1] Sinkhorn, Richard, and Paul Knopp. "Concerning nonnegative matrices and doubly stochastic matrices." Pacific Journal of Mathematics 21.2 (1967): 343-348.

[2] Munkres, James. "Algorithms for the assignment and transportation problems." Journal of the society for industrial and applied mathematics 5.1 (1957): 32-38.

[3] Leordeanu, Marius, and Martial Hebert. "A spectral technique for correspondence problems using pairwise constraints." International Conference on Computer Vision (2005).

[4] Cho, Minsu, Jungmin Lee, and Kyoung Mu Lee. "Reweighted random walks for graph matching." European conference on Computer vision. Springer, Berlin, Heidelberg, 2010.

[5] Leordeanu, Marius, Martial Hebert, and Rahul Sukthankar. "An integer projected fixed point method for graph matching and map inference." Advances in neural information processing systems 22 (2009).

[6] Yan, Junchi, et al. "Multi-graph matching via affinity optimization with graduated consistency regularization." IEEE transactions on pattern analysis and machine intelligence 38.6 (2015): 1228-1242.

[7] Jiang, Zetian, Tianzhe Wang, and Junchi Yan. "Unifying offline and online multi-graph matching via finding shortest paths on supergraph." IEEE transactions on pattern analysis and machine intelligence 43.10 (2020): 3648-3663.

[8] Solé-Ribalta, Albert, and Francesc Serratosa. "Graduated assignment algorithm for multiple graph matching based on a common labeling." International Journal of Pattern Recognition and Artificial Intelligence 27.01 (2013): 1350001.

[9] Wang, Runzhong, Junchi Yan, and Xiaokang Yang. "Graduated assignment for joint multi-graph matching and clustering with application to unsupervised graph matching network learning." Advances in Neural Information Processing Systems 33 (2020): 19908-19919.

[10] Wang, Runzhong, Junchi Yan, and Xiaokang Yang. "Combinatorial learning of robust deep graph matching: an embedding based approach." IEEE Transactions on Pattern Analysis and Machine Intelligence (2020).

[11] Yu, Tianshu, et al. "Learning deep graph matching with channel-independent embedding and hungarian attention." International conference on learning representations. 2019.

[12] Wang, Runzhong, Junchi Yan, and Xiaokang Yang. "Neural graph matching network: Learning lawler’s quadratic assignment problem with extension to hypergraph and multiple-graph matching." IEEE Transactions on Pattern Analysis and Machine Intelligence (2021).
