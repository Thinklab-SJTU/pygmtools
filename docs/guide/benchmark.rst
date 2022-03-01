===========================
Graph Matching Benchmark
===========================

**pygmtools** also provides a protocol to fairly compare existing deep graph matching algorithms under different datasets & experiment settings.
The ``Benchmark`` module provides a unified data interface and an evaluating platform for different datasets.
Currently, pygmtools supports 5 datasets:

* PascalVOC
* Willow-Object
* SPair-71k
* CUB2011
* IMC-PT-SparseGM

Files
----------

* ``dataset.py``: The file includes 5 dataset classes, used to automatically download dataset and process the dataset into a json file, and also save train set and test set.
* ``benchmark.py``: The file includes Benchmark class that can be used to fetch data from json file and evaluate prediction result.
* ``dataset_config.py``: Fixed dataset settings, mostly dataset path and classes.

Notes
----------

* Our evaluation metrics include **matching_precision (p)**, **matching_recall (r)** and **f1_score (f1)**.
  Also, to measure the reliability of the evaluation result, we define coverage (cvg) for each class in the dataset
  as *the number of evaluated pairs in the class / number of all possible pairs* in the class. Therefore,
  larger coverage refers to higher reliability.

* Dataset can be automatically downloaded and unzipped, but you can also download the dataset yourself,
  and make sure it in the right path. The expected dataset paths are listed as follows.

  ::

    # Pascal VOC 2011 dataset with keypoint annotations
    PascalVOC.ROOT_DIR = 'data/PascalVOC/TrainVal/VOCdevkit/VOC2011/'
    PascalVOC.KPT_ANNO_DIR = 'data/PascalVOC/annotations/'

    # Willow-Object Class dataset
    WillowObject.ROOT_DIR = 'data/WillowObject/WILLOW-ObjectClass'

    # CUB2011 dataset
    CUB2011.ROOT_PATH = 'data/CUB_200_2011/CUB_200_2011'

    # SWPair-71 Dataset
    SPair.ROOT_DIR = "data/SPair-71k"

    # IMC_PT_SparseGM dataset
    IMC_PT_SparseGM.ROOT_DIR_NPZ = 'data/IMC-PT-SparseGM/annotations'
    IMC_PT_SparseGM.ROOT_DIR_IMG = 'data/IMC-PT-SparseGM/images'

  Specifically, for PascalVOC, you should download the train/test split yourself, and make sure it looks like ``data/PascalVOC/voc2011_pairs.npz``

Example
-----------

::

    from pygmtools.benchmark import Benchmark

    # Define Benchmark on PascalVOC.
    bm = Benchmark(name='PascalVOC', sets='train',
                   obj_resize=(256, 256), problem='2GM',
                   filter='intersection')

    # Random fetch data and ground truth.
    data_list, gt_dict, _ = bm.rand_get_data(cls=None, num=2)

