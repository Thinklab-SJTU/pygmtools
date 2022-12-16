===========================
Graph Matching Benchmark
===========================

**pygmtools** also provides a protocol to fairly compare existing deep graph matching algorithms under different datasets & experiment settings.
The ``Benchmark`` module provides a unified data interface and an evaluating platform for different datasets.

If you are interested in the performance and the full deep learning pipeline, please refer to our `ThinkMatch project <https://github.com/Thinklab-SJTU/ThinkMatch>`_.

Evaluation Metrics and Results
-------------------------------------

Our evaluation metrics include **matching_precision (p)**, **matching_recall (r)** and **f1_score (f1)**.
Also, to measure the reliability of the evaluation result, we define **coverage (cvg)** for each class in the dataset
as *the number of evaluated pairs in the class/number of all possible pairs* in the class. Therefore,
larger coverage refers to higher reliability.

An example of evaluation result (``p==r==f1`` because this evaluation does not involve partial matching/outliers):

::

    Matching accuracy
    Car: p = 0.8395±0.2280, r = 0.8395±0.2280, f1 = 0.8395±0.2280, cvg = 1.0000
    Duck: p = 0.7713±0.2255, r = 0.7713±0.2255, f1 = 0.7713±0.2255, cvg = 1.0000
    Face: p = 0.9656±0.0913, r = 0.9656±0.0913, f1 = 0.9656±0.0913, cvg = 0.2612
    Motorbike: p = 0.8821±0.1821, r = 0.8821±0.1821, f1 = 0.8821±0.1821, cvg = 1.0000
    Winebottle: p = 0.8929±0.1569, r = 0.8929±0.1569, f1 = 0.8929±0.1569, cvg = 0.9662
    average accuracy: p = 0.8703±0.1767, r = 0.8703±0.1767, f1 = 0.8703±0.1767
    Evaluation complete in 1m 55s


Available Datasets
--------------------
Dataset can be automatically downloaded and unzipped, but you can also download the dataset yourself,
and make sure it in the right path.

PascalVOC-Keypoint Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download `VOC2011 dataset <http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html>`_ and make sure it looks like ``data/PascalVOC/TrainVal/VOCdevkit/VOC2011``

#. Download keypoint annotation for VOC2011 from `Berkeley server <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz>`_ or `google drive <https://drive.google.com/open?id=1D5o8rmnY1-DaDrgAXSygnflX5c-JyUWR>`_ and make sure it looks like ``data/PascalVOC/annotations``

#. Download the `train/test split file <https://github.com/Thinklab-SJTU/ThinkMatch/raw/master/data/PascalVOC/voc2011_pairs.npz>`_ and make sure it looks like ``data/PascalVOC/voc2011_pairs.npz``

Please cite the following papers if you use PascalVOC-Keypoint dataset:

::

    @article{EveringhamIJCV10,
      title={The pascal visual object classes (voc) challenge},
      author={Everingham, Mark and Van Gool, Luc and Williams, Christopher KI and Winn, John and Zisserman, Andrew},
      journal={International Journal of Computer Vision},
      volume={88},
      pages={303–338},
      year={2010}
    }

    @inproceedings{BourdevICCV09,
      title={Poselets: Body part detectors trained using 3d human pose annotations},
      author={Bourdev, L. and Malik, J.},
      booktitle={International Conference on Computer Vision},
      pages={1365--1372},
      year={2009},
      organization={IEEE}
    }

Willow-Object-Class Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download `Willow-ObjectClass dataset <http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip>`_

#. Unzip the dataset and make sure it looks like ``data/WillowObject/WILLOW-ObjectClass``

Please cite the following paper if you use Willow-Object-Class dataset:

::

    @inproceedings{ChoICCV13,
      author={Cho, Minsu and Alahari, Karteek and Ponce, Jean},
      title = {Learning Graphs to Match},
      booktitle = {International Conference on Computer Vision},
      pages={25--32},
      year={2013}
    }

CUB2011 Dataset
^^^^^^^^^^^^^^^^^^^

#. Download `CUB-200-2011 dataset <http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz>`_.

#. Unzip the dataset and make sure it looks like ``data/CUB_200_2011/CUB_200_2011``

Please cite the following report if you use CUB2011 dataset:

::

    @techreport{CUB2011,
      Title = {{The Caltech-UCSD Birds-200-2011 Dataset}},
      Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
      Year = {2011},
      Institution = {California Institute of Technology},
      Number = {CNS-TR-2011-001}
    }

IMC-PT-SparseGM Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Download the IMC-PT-SparseGM dataset from `google drive <https://drive.google.com/file/d/1Po9pRMWXTqKK2ABPpVmkcsOq-6K_2v-B/view?usp=sharing>`_ or `baidu drive (code: 0576) <https://pan.baidu.com/s/1hlJdIFp4rkiz1Y-gztyHIw>`_

#. Unzip the dataset and make sure it looks like ``data/IMC_PT_SparseGM/annotations``

Please cite the following papers if you use IMC-PT-SparseGM dataset:

::

    @article{JinIJCV21,
      title={Image Matching across Wide Baselines: From Paper to Practice},
      author={Jin, Yuhe and Mishkin, Dmytro and Mishchuk, Anastasiia and Matas, Jiri and Fua, Pascal and Yi, Kwang Moo and Trulls, Eduard},
      journal={International Journal of Computer Vision},
      pages={517--547},
      year={2021}
    }

SPair-71k Dataset
^^^^^^^^^^^^^^^^^^^^

#. Download `SPair-71k dataset <http://cvlab.postech.ac.kr/research/SPair-71k/>`_

#. Unzip the dataset and make sure it looks like ``data/SPair-71k``

Please cite the following papers if you use SPair-71k dataset:

::

    @article{min2019spair,
       title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
       author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
       journal={arXiv prepreint arXiv:1908.10543},
       year={2019}
    }

    @InProceedings{min2019hyperpixel,
       title={Hyperpixel Flow: Semantic Correspondence with Multi-layer Neural Features},
       author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
       booktitle={ICCV},
       year={2019}
    }

API Reference
------------------
See :doc:`the API doc of Benchmark module <../api/_autosummary/pygmtools.benchmark.Benchmark>` and
:doc:`the API doc of datasets <../api/_autosummary/pygmtools.dataset>` for details.


File Organization
------------------

* ``dataset.py``: The file includes 5 dataset classes, used to automatically download the dataset and process the dataset into a json file, and also save the training set and the testing set.
* ``benchmark.py``: The file includes Benchmark class that can be used to fetch data from the json file and evaluate prediction results.
* ``dataset_config.py``: The default dataset settings, mostly dataset path and classes.


Example
-----------

::

    import pygmtools as pygm
    from pygm.benchmark import Benchmark

    # Define Benchmark on PascalVOC.
    bm = Benchmark(name='PascalVOC', sets='train',
                   obj_resize=(256, 256), problem='2GM',
                   filter='intersection')

    # Random fetch data and ground truth.
    data_list, gt_dict, _ = bm.rand_get_data(cls=None, num=2)
