=================================
Introduction and Guidelines
=================================

This page provides a brief introduction to graph matching and some guidelines for using ``pygmtools``.
If you are seeking some background information, this is the right place!

.. note::
    For more technical details, we recommend the following two surveys.

    About **learning-based** deep graph matching:
    Junchi Yan, Shuang Yang, Edwin Hancock. `"Learning Graph Matching and Related Combinatorial Optimization Problems." <https://www.ijcai.org/proceedings/2020/0694.pdf>`_ *IJCAI 2020*.

    About **non-learning** two-graph matching and multi-graph matching:
    Junchi Yan, Xu-Cheng Yin, Weiyao Lin, Cheng Deng, Hongyuan Zha, Xiaokang Yang. `"A Short Survey of Recent Advances in Graph Matching." <https://dl.acm.org/doi/10.1145/2911996.2912035>`_ *ICMR 2016*.


Why Graph Matching?
--------------------

Graph Matching (GM) is a fundamental yet challenging problem in pattern recognition, data mining, and others.
GM aims to find node-to-node correspondence among multiple graphs, by solving an NP-hard combinatorial problem.
Recently, there is growing interest in developing deep learning-based graph matching methods.

Compared to other straight-forward matching methods e.g. greedy matching, graph matching methods are more reliable
because it is based on an optimization form. Besides, graph matching methods exploit both node affinity and edge
affinity, thus graph matching methods are usually more robust to noises and outliers. The recent line of deep graph
matching methods also enables many graph matching solvers to be integrated into a deep learning pipeline.

Graph matching techniques have been applied to the following applications:

* `Bridging movie and synopses <https://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_A_Graph-Based_Framework_to_Bridge_Movies_and_Synopses_ICCV_2019_paper.pdf>`_

  .. image:: ../images/movie_synopses.png
     :width: 500

* `Image correspondence <https://arxiv.org/pdf/1911.11763.pdf>`_

  .. image:: ../images/superglue.png
     :width: 500

* `Model ensemble and federated learning <https://proceedings.mlr.press/v162/liu22k/liu22k.pdf>`_

  .. image:: ../images/federated_learning.png
     :width: 500

* `Molecules matching <https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Combinatorial_Learning_of_Graph_Edit_Distance_via_Dynamic_Embedding_CVPR_2021_paper.pdf>`_

  .. image:: ../images/molecules.png
     :width: 450

* and more...

If your task involves matching two or more graphs, you should try the solvers in ``pygmtools``!

What is Graph Matching?
------------------------

The Math Form
^^^^^^^^^^^^^^

Let's involve a little bit of math to better understand the graph matching pipeline.
In general, graph matching is of the following form, known as **Quadratic Assignment Problem (QAP)**:

.. math::

    &\max_{\mathbf{X}} \ \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})\\
    s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}

The notations are explained as follows:

* :math:`\mathbf{X}` is known as the **permutation matrix** which encodes the matching result. It is also the decision
  variable in graph matching problem. :math:`\mathbf{X}_{i,a}=1` means node :math:`i` in graph 1 is matched to node :math:`a` in graph 2,
  and :math:`\mathbf{X}_{i,a}=0` means non-matched. Without loss of generality, it is assumed that :math:`n_1\leq n_2.`
  :math:`\mathbf{X}` has the following constraints:

  * The sum of each row must be equal to 1: :math:`\mathbf{X}\mathbf{1} = \mathbf{1}`;
  * The sum of each column must be equal to, or smaller than 1: :math:`\mathbf{X}\mathbf{1} \leq \mathbf{1}`.

* :math:`\mathtt{vec}(\mathbf{X})` means the column-wise vectorization form of :math:`\mathbf{X}`.

* :math:`\mathbf{1}` means a column vector whose elements are all 1s.

* :math:`\mathbf{K}` is known as the **affinity matrix** which encodes the information of the input graphs.
  Both node-wise and edge-wise affinities are encoded in :math:`\mathbf{K}`:

  * The diagonal element :math:`\mathbf{K}_{i + a\times n_1, i + a\times n_1}` means the node-wise affinity of
    node :math:`i` in graph 1 and node :math:`a` in graph 2;
  * The off-diagonal element :math:`\mathbf{K}_{i + a\times n_1, j + b\times n_1}` means the edge-wise affinity of
    edge :math:`ij` in graph 1 and edge :math:`ab` in graph 2.

The Graph Matching Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Solving a real-world graph-matching problem can be divided into the following parts:

Part 1: Feature Extraction
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Extract node/edge features from the graphs you want to match. The features are used to measure the similarity
between nodes/edges and to build the affinity matrix which is essential in graph matching problems.

Part 2: Affinity Matrix Construction
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Build the affinity matrix from node/edge features and form the specific QAP problem.

Part 3: QAP Problem Solving
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Solve the resulting QAP problem (graph matching problem) with GM solvers.

Part 1 may be done by methods depending on your application, Part 2\&3 can be handled by ``pygmtools``.
The following plot illustrates a standard deep graph matching pipeline.

.. image:: ../images/QAP_illustration.png

Graph Matching Best Practice
-----------------------------

We need to understand the advantages and limitations of graph matching solvers. As discussed above, the major advantage
of graph matching solvers is that they are more robust to noises and outliers. Graph matching also utilizes edge
information, which is usually ignored in linear matching methods. The major drawback of graph matching
solvers is their efficiency and scalability since the optimization problem is NP-hard. Therefore, to decide which
matching method is most suitable, one needs to balance between the required matching accuracy and the affordable time
and memory cost according to his/her application.

.. note::

    Anyway, it does no harm to try graph matching first!

When to use pygmtools
^^^^^^^^^^^^^^^^^^^^^^

``pygmtools`` is recommended for the following cases, and you could benefit from the friendly API:

* If you want to integrate graph matching as a step of your pipeline (either learning or non-learning).

* If you want a quick benchmarking and profiling of the graph matching solvers available in ``pygmtools``.

* If you do not want to dive too deep into the algorithm details and do not need to modify the algorithm.

We offer the following guidelines for your reference:

* If you want to integrate graph matching solvers into your end-to-end supervised deep learning pipeline, try
  :mod:`~pygmtools.neural_solvers`.

* If no ground truth label is available for the matching step, try :mod:`~pygmtools.classic_solvers`.

* If there are multiple graphs to be jointly matched, try :mod:`~pygmtools.multi_graph_solvers`.

* If time and memory cost of the above methods are unacceptable for your task, try :mod:`~pygmtools.linear_solvers`.

When not to use pygmtools
^^^^^^^^^^^^^^^^^^^^^^^^^^

As a highly packed toolkit, ``pygmtools`` lacks some flexibilities in the implementation details, especially for
experts in graph matching. If you are researching new graph matching algorithms or developing next-generation deep
graph matching neural networks, ``pygmtools`` may not be suitable. We recommend
`ThinkMatch <https://github.com/Thinklab-SJTU/ThinkMatch>`_ as the protocol for academic research.

What's Next
------------
Please read the :doc:`get_started` guide.
