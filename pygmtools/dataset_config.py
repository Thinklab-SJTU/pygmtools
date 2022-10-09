from easydict import EasyDict as edict

__C = edict()

dataset_cfg = __C
# Pascal VOC 2011 dataset with keypoint annotations
__C.PascalVOC = edict()
__C.PascalVOC.KPT_ANNO_DIR = 'data/PascalVOC/annotations/'  # keypoint annotation
__C.PascalVOC.ROOT_DIR = 'data/PascalVOC/TrainVal/VOCdevkit/VOC2011/'  # original VOC2011 dataset
__C.PascalVOC.SET_SPLIT = 'data/PascalVOC/voc2011_pairs.npz'  # set split path
__C.PascalVOC.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                         'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                         'tvmonitor']

# Willow-Object Class dataset
__C.WillowObject = edict()
__C.WillowObject.ROOT_DIR = 'data/WillowObject/WILLOW-ObjectClass'
__C.WillowObject.CLASSES = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
__C.WillowObject.KPT_LEN = 10
__C.WillowObject.TRAIN_NUM = 20
__C.WillowObject.SPLIT_OFFSET = 0
__C.WillowObject.TRAIN_SAME_AS_TEST = False
__C.WillowObject.RAND_OUTLIER = 0

# CUB2011 dataset
__C.CUB2011 = edict()
__C.CUB2011.ROOT_DIR = 'data/CUB_200_2011/CUB_200_2011'
__C.CUB2011.CLASS_SPLIT = 'ori' # choose from 'ori' (original split), 'sup' (super class) or 'all' (all birds as one class)

# SWPair-71 Dataset
__C.SPair = edict()
__C.SPair.ROOT_DIR = "data/SPair-71k"
__C.SPair.size = "large"
__C.SPair.CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
]
__C.SPair.TRAIN_DIFF_PARAMS = {}
__C.SPair.EVAL_DIFF_PARAMS = {}
__C.SPair.COMB_CLS = False

# IMC_PT_SparseGM dataset
__C.IMC_PT_SparseGM = edict()
__C.IMC_PT_SparseGM.CLASSES = {'train': ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior',
                                      'grand_place_brussels', 'hagia_sophia_interior', 'notre_dame_front_facade',
                                      'palace_of_westminster', 'pantheon_exterior', 'prague_old_town_square',
                                      'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey'],
                            'test': ['reichstag', 'sacre_coeur', 'st_peters_square']}
__C.IMC_PT_SparseGM.ROOT_DIR_NPZ = 'data/IMC-PT-SparseGM/annotations'
__C.IMC_PT_SparseGM.ROOT_DIR_IMG = 'data/IMC-PT-SparseGM/images'
__C.IMC_PT_SparseGM.TOTAL_KPT_NUM = 50

__C.CACHE_PATH = 'data/cache'
