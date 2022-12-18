r"""
The implementations of data loading and data processing.
"""

# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import requests
import os
import zipfile
import tarfile
from pygmtools.dataset_config import dataset_cfg
from pathlib import Path
from xml.etree.ElementTree import Element
from PIL import Image
from tqdm.auto import tqdm
from time import sleep
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import json
import scipy.io as sio
import glob
import random
from pygmtools.utils import download


VOC2011_KPT_NAMES = {
    'cat': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'bottle': ['L_Base', 'L_Neck', 'L_Shoulder', 'L_Top', 'R_Base', 'R_Neck',
               'R_Shoulder', 'R_Top'],
    'horse': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
              'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
              'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'motorbike': ['B_WheelCenter', 'B_WheelEnd', 'ExhaustPipeEnd',
                  'F_WheelCenter', 'F_WheelEnd', 'HandleCenter', 'L_HandleTip',
                  'R_HandleTip', 'SeatBase', 'TailLight'],
    'boat': ['Hull_Back_Bot', 'Hull_Back_Top', 'Hull_Front_Bot',
             'Hull_Front_Top', 'Hull_Mid_Left_Bot', 'Hull_Mid_Left_Top',
             'Hull_Mid_Right_Bot', 'Hull_Mid_Right_Top', 'Mast_Top', 'Sail_Left',
             'Sail_Right'],
    'tvmonitor': ['B_Bottom_Left', 'B_Bottom_Right', 'B_Top_Left',
                  'B_Top_Right', 'F_Bottom_Left', 'F_Bottom_Right', 'F_Top_Left',
                  'F_Top_Right'],
    'cow': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'chair': ['BackRest_Top_Left', 'BackRest_Top_Right', 'Leg_Left_Back',
              'Leg_Left_Front', 'Leg_Right_Back', 'Leg_Right_Front',
              'Seat_Left_Back', 'Seat_Left_Front', 'Seat_Right_Back',
              'Seat_Right_Front'],
    'car': ['L_B_RoofTop', 'L_B_WheelCenter', 'L_F_RoofTop', 'L_F_WheelCenter',
            'L_HeadLight', 'L_SideviewMirror', 'L_TailLight', 'R_B_RoofTop',
            'R_B_WheelCenter', 'R_F_RoofTop', 'R_F_WheelCenter', 'R_HeadLight',
            'R_SideviewMirror', 'R_TailLight'],
    'person': ['B_Head', 'HeadBack', 'L_Ankle', 'L_Ear', 'L_Elbow', 'L_Eye',
               'L_Foot', 'L_Hip', 'L_Knee', 'L_Shoulder', 'L_Toes', 'L_Wrist', 'Nose',
               'R_Ankle', 'R_Ear', 'R_Elbow', 'R_Eye', 'R_Foot', 'R_Hip', 'R_Knee',
               'R_Shoulder', 'R_Toes', 'R_Wrist'],
    'diningtable': ['Bot_Left_Back', 'Bot_Left_Front', 'Bot_Right_Back',
                    'Bot_Right_Front', 'Top_Left_Back', 'Top_Left_Front', 'Top_Right_Back',
                    'Top_Right_Front'],
    'dog': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
            'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
            'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'bird': ['Beak_Base', 'Beak_Tip', 'Left_Eye', 'Left_Wing_Base',
             'Left_Wing_Tip', 'Leg_Center', 'Lower_Neck_Base', 'Right_Eye',
             'Right_Wing_Base', 'Right_Wing_Tip', 'Tail_Tip', 'Upper_Neck_Base'],
    'bicycle': ['B_WheelCenter', 'B_WheelEnd', 'B_WheelIntersection',
                'CranksetCenter', 'F_WheelCenter', 'F_WheelEnd', 'F_WheelIntersection',
                'HandleCenter', 'L_HandleTip', 'R_HandleTip', 'SeatBase'],
    'train': ['Base_Back_Left', 'Base_Back_Right', 'Base_Front_Left',
              'Base_Front_Right', 'Roof_Back_Left', 'Roof_Back_Right',
              'Roof_Front_Middle'],
    'sheep': ['L_B_Elbow', 'L_B_Paw', 'L_EarBase', 'L_Eye', 'L_F_Elbow',
              'L_F_Paw', 'Nose', 'R_B_Elbow', 'R_B_Paw', 'R_EarBase', 'R_Eye',
              'R_F_Elbow', 'R_F_Paw', 'TailBase', 'Throat', 'Withers'],
    'aeroplane': ['Bot_Rudder', 'Bot_Rudder_Front', 'L_Stabilizer',
                  'L_WingTip', 'Left_Engine_Back', 'Left_Engine_Front',
                  'Left_Wing_Base', 'NoseTip', 'Nose_Bottom', 'Nose_Top',
                  'R_Stabilizer', 'R_WingTip', 'Right_Engine_Back',
                  'Right_Engine_Front', 'Right_Wing_Base', 'Top_Rudder'],
    'sofa': ['Back_Base_Left', 'Back_Base_Right', 'Back_Top_Left',
             'Back_Top_Right', 'Front_Base_Left', 'Front_Base_Right',
             'Handle_Front_Left', 'Handle_Front_Right', 'Handle_Left_Junction',
             'Handle_Right_Junction', 'Left_Junction', 'Right_Junction'],
    'pottedplant': ['Bottom_Left', 'Bottom_Right', 'Top_Back_Middle',
                    'Top_Front_Middle', 'Top_Left', 'Top_Right'],
    'bus': ['L_B_Base', 'L_B_RoofTop', 'L_F_Base', 'L_F_RoofTop', 'R_B_Base',
            'R_B_RoofTop', 'R_F_Base', 'R_F_RoofTop']
}


class PascalVOC:
    r"""
    Download and preprocess **PascalVOC Keypoint** dataset.

    :param sets: str, problem set, ``'train'`` for training set and ``'test'`` for testing set
    :param obj_resize: tuple, resized image size
    :param ds_dict: settings of dataset, containing at most 5 params(keys) for PascalVOC:

            * **KPT_ANNO_DIR**: str, directory of keypoint annotations

            * **ROOT_DIR**: str, directory of data

            * **SET_SPLIT**: str, set split path

            * **CLASSES**: list, data classes

            * **CACHE_PATH**: str, directory of data cache
    """
    def __init__(self, sets, obj_resize, **ds_dict):
        KPT_ANNO_DIR = dataset_cfg.PascalVOC.KPT_ANNO_DIR
        ROOT_DIR = dataset_cfg.PascalVOC.ROOT_DIR
        SET_SPLIT = dataset_cfg.PascalVOC.SET_SPLIT
        CLASSES = dataset_cfg.PascalVOC.CLASSES
        CACHE_PATH = dataset_cfg.CACHE_PATH
        if len(ds_dict.keys()) > 0:
            if 'CLASSES' in ds_dict.keys():
                CLASSES = ds_dict['CLASSES']
            if 'ROOT_DIR' in ds_dict.keys():
                ROOT_DIR = ds_dict['ROOT_DIR']
            if 'SET_SPLIT' in ds_dict.keys():
                SET_SPLIT = ds_dict['SET_SPLIT']
            if 'KPT_ANNO_DIR' in ds_dict.keys():
                KPT_ANNO_DIR = ds_dict['KPT_ANNO_DIR']
            if 'CACHE_PATH' in ds_dict.keys():
                CACHE_PATH = ds_dict['CACHE_PATH']

        VOC2011_anno_path = KPT_ANNO_DIR
        VOC2011_img_path = ROOT_DIR + 'JPEGImages'
        VOC2011_ori_anno_path = ROOT_DIR + 'Annotations'
        VOC2011_cache_path = CACHE_PATH

        self.VOC2011_set_path = SET_SPLIT
        self.dataset_dir = 'data/PascalVOC'
        if not os.path.exists(ROOT_DIR):
            assert ROOT_DIR == dataset_cfg.PascalVOC.ROOT_DIR, 'you should not change ROOT_DIR unless the data have been manually downloaded'
            self.download(url='http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
                          name='PascalVOC')

        if not os.path.exists(KPT_ANNO_DIR):
            assert KPT_ANNO_DIR == dataset_cfg.PascalVOC.KPT_ANNO_DIR, 'you should not change KPT_ANNO_DIR unless the annotations have been manually downloaded'
            self.download(
                url='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz',
                name='PascalVOC_anno')

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.sets = sets
        self.obj_resize = obj_resize
        self.suffix = 'pca'

        self.classes = CLASSES
        self.kpt_len = [len(VOC2011_KPT_NAMES[_]) for _ in self.classes]
        self.classes_kpts = {cls: len(VOC2011_KPT_NAMES[cls]) for cls in self.classes}
        self.anno_path = Path(VOC2011_anno_path)
        self.img_path = Path(VOC2011_img_path)
        self.ori_anno_path = Path(VOC2011_ori_anno_path)

        assert sets == 'train' or sets == 'test', 'No match found for dataset {}'.format(sets)
        cache_name = 'voc_db_' + sets + '.pkl'
        self.cache_path = Path(VOC2011_cache_path)
        self.cache_file = self.cache_path / cache_name
        if self.cache_file.exists():
            with self.cache_file.open(mode='rb') as f:
                self.xml_list = pickle.load(f)
            print('xml list loaded from {}'.format(self.cache_file))

        else:
            if self.sets != 'test':
                print('Caching xml list to {}...'.format(self.cache_file))
            self.cache_path.mkdir(exist_ok=True, parents=True)
            with np.load(self.VOC2011_set_path, allow_pickle=True) as f:
                self.xml_list = f[sets]
            before_filter = sum([len(k) for k in self.xml_list])
            self.__filter_list(self.xml_list)
            after_filter = sum([len(k) for k in self.xml_list])
            with self.cache_file.open(mode='wb') as f:
                pickle.dump(self.xml_list, f)
            print('Filtered {} images to {}. Annotation saved.'.format(before_filter, after_filter))

        self.process()

    def download(self, url=None, name=None, retries=5):
        r"""
        Automatically download PascalVOC dataset.

        :param url: str, web url of PascalVOC and PascalVOC annotation
        :param name: str, ``"PascalVOC"`` to download PascalVOC and ``"PascalVOC_anno"`` to download PascalVOC annotation
        """
        if retries <= 0:
            raise RuntimeError('Max Retries exceeded!')

        dirs = 'data/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        if name == "PascalVOC_anno":
            print('Downloading dataset annotation...')
            filename = "data/PascalVOC.tgz"
            download(filename=filename, url=url, to_cache=False)
            try:
                tar = tarfile.open(filename, "r")
            except tarfile.ReadError as err:
                print('Warning: Content error. Retrying...\n', err)
                os.remove(filename)
                return self.download(url, name, retries - 1)

            file_names = tar.getnames()
            print('Unzipping files...')
            sleep(0.5)
            for file_name in tqdm(file_names):
                tar.extract(file_name, "data/PascalVOC/")
            tar.close()
            os.remove(filename)

        if name == "PascalVOC":
            print('Downloading dataset PascalVOC...')
            filename = "data/PascalVOC.tar"
            download(filename=filename, url=url, to_cache=False)
            try:
                tar = tarfile.open(filename, "r")
            except tarfile.ReadError as err:
                print('Warning: Content error. Retrying...\n', err)
                os.remove(filename)
                return self.download(url, name, retries - 1)

            file_names = tar.getnames()
            print('Unzipping files...')
            sleep(0.5)
            for file_name in tqdm(file_names):
                tar.extract(file_name, "data/PascalVOC/")
            tar.close()
            os.remove(filename)
        return filename

    def __filter_list(self, a_xml_list):
        """
        Filter out ``'truncated'``, ``'occluded'`` and ``'difficult'`` images following the practice of previous works.
        In addition, this dataset has uncleaned label (in person category). They are omitted as suggested by README.
        """
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in a_xml_list[cls_id]:
                xml_comps = xml_name.split('/')[-1].strip('.xml').split('_')
                ori_xml_name = '_'.join(xml_comps[:-1]) + '.xml'
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), '{} does not exist.'.format(xml_file)
                tree = ET.parse(xml_file.open())
                root = tree.getroot()
                obj: Element = root.findall('object')[voc_idx - 1]

                difficult = obj.find('difficult')
                if difficult is not None:
                    difficult = int(difficult.text)
                occluded = obj.find('occluded')
                if occluded is not None:
                    occluded = int(occluded.text)
                truncated = obj.find('truncated')
                if truncated is not None:
                    truncated = int(truncated.text)
                if difficult or occluded or truncated:
                    to_del.append(xml_name)
                    continue

                    # Exclude uncleaned images
                if self.classes[cls_id] == 'person' and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
                    continue

                    # Exclude overlapping images in Willow
                    # if self.sets == 'train' and (self.classes[cls_id] == 'motorbike' or self.classes[cls_id] == 'car') \
                    #        and int(xml_comps[0]) == 2007:
                    #    to_del.append(xml_name)
                    #    continue

            for x in to_del:
                a_xml_list[cls_id].remove(x)

    def process(self):
        r"""
        Process the dataset and generate ``data-(size, size).json`` for preprocessed dataset, ``train.json`` for
        training set, and ``test.json`` for testing set.
        """
        train_file = os.path.join(self.dataset_dir, 'train.json')
        test_file = os.path.join(self.dataset_dir, 'test.json')
        img_file = os.path.join(self.dataset_dir, 'data-' + str(self.obj_resize) + '-' + self.suffix + '.json')
        if not (os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(img_file)):
            if not (os.path.exists(train_file) and os.path.exists(test_file)):
                list1 = []
                for x in range(len(self.xml_list)):
                    for xml_name in self.xml_list[x]:
                        tmp = xml_name.split('/')
                        tmp2 = tmp[1].split('.')
                        objID = tmp2[0] + '_' + tmp[0]
                        list1.append(objID)

                list2 = []
                if self.sets == 'train':
                    with np.load(self.VOC2011_set_path, allow_pickle=True) as f:
                        a_list = f['test']
                    self.__filter_list(a_list)
                    cache_name = 'voc_db_test.pkl'
                    cache_file = self.cache_path / cache_name
                    if not cache_file.exists():
                        with cache_file.open(mode='wb') as f:
                            pickle.dump(a_list, f)

                    for x in range(len(a_list)):
                        for xml_name in a_list[x]:
                            tmp = xml_name.split('/')
                            tmp2 = tmp[1].split('.')
                            objID = tmp2[0] + '_' + tmp[0]
                            list2.append(objID)
                    str1 = json.dumps(list1)
                    f1 = open(train_file, 'w')
                    f1.write(str1)
                    f1.close()
                    str2 = json.dumps(list2)
                    f2 = open(test_file, 'w')
                    f2.write(str2)
                    f2.close()
                else:
                    with np.load(self.VOC2011_set_path, allow_pickle=True) as f:
                        a_list = f['train']
                    self.__filter_list(a_list)
                    cache_name = 'voc_db_train.pkl'
                    cache_file = self.cache_path / cache_name
                    if not cache_file.exists():
                        with cache_file.open(mode='wb') as f:
                            pickle.dump(a_list, f)

                    for x in range(len(a_list)):
                        for xml_name in a_list[x]:
                            tmp = xml_name.split('/')
                            tmp2 = tmp[1].split('.')
                            objID = tmp2[0] + '_' + tmp[0]
                            list2.append(objID)
                    str1 = json.dumps(list1)
                    f1 = open(test_file, 'w')
                    f1.write(str1)
                    f1.close()
                    str2 = json.dumps(list2)
                    f2 = open(train_file, 'w')
                    f2.write(str2)
                    f2.close()
            else:
                if self.sets == 'train':
                    with np.load(self.VOC2011_set_path, allow_pickle=True) as f:
                        a_list = f['test']
                    self.__filter_list(a_list)
                    cache_name = 'voc_db_test.pkl'
                    cache_file = self.cache_path / cache_name
                    if not cache_file.exists():
                        with cache_file.open(mode='wb') as f:
                            pickle.dump(a_list, f)

                else:
                    with np.load(self.VOC2011_set_path, allow_pickle=True) as f:
                        a_list = f['train']
                    self.__filter_list(a_list)
                    cache_name = 'voc_db_train.pkl'
                    cache_file = self.cache_path / cache_name
                    if not cache_file.exists():
                        with cache_file.open(mode='wb') as f:
                            pickle.dump(a_list, f)

            if not os.path.exists(img_file):
                data_list = self.xml_list + a_list
                data_dict = dict()
                for x in range(len(data_list)):
                    for xml_name in data_list[x]:
                        tmp = xml_name.split('/')
                        tmp2 = tmp[1].split('.')
                        objID = tmp2[0] + '_' + tmp[0]
                        annotations = self.__get_anno_dict(xml_name)
                        data_dict[objID] = annotations

                data_str = json.dumps(data_dict)
                f3 = open(img_file, 'w')
                f3.write(data_str)
                f3.close()

    def __get_anno_dict(self, xml_name):
        """
        Get an annotation dict from xml file
        """
        xml_file = self.anno_path / xml_name
        assert xml_file.exists(), '{} does not exist.'.format(xml_file)

        tree = ET.parse(xml_file.open())
        root = tree.getroot()

        img_name = root.find('./image').text + '.jpg'
        img_file = self.img_path / img_name
        bounds = root.find('./visible_bounds').attrib
        cls = root.find('./category').text

        xmin = float(bounds['xmin'])
        ymin = float(bounds['ymin'])
        h = float(bounds['height'])
        w = float(bounds['width'])
        xmax = float(xmin) + float(w)
        ymax = float(ymin) + float(h)

        keypoint_list = []
        for keypoint in root.findall('./keypoints/keypoint'):
            attr = keypoint.attrib
            attr['x'] = (float(attr['x']) - xmin) * self.obj_resize[0] / w
            attr['y'] = (float(attr['y']) - ymin) * self.obj_resize[1] / h
            kpts_anno = dict()
            kpts_anno['labels'] = attr['name']
            kpts_anno['x'] = attr['x']
            kpts_anno['y'] = attr['y']
            keypoint_list.append(kpts_anno)

        anno_dict = dict()
        anno_dict['kpts'] = keypoint_list
        anno_dict['path'] = str(img_file)
        anno_dict['cls'] = cls
        anno_dict['bounds'] = [xmin, ymin, xmax, ymax]
        anno_dict['univ_size'] = len(VOC2011_KPT_NAMES[cls])

        return anno_dict


class WillowObject:
    r"""
        Download and preprocess **Willow Object Class** dataset.

        :param sets: str, problem set, ``'train'`` for training set and ``'test'`` for testing set
        :param obj_resize: tuple, resized image size
        :param ds_dict: settings of dataset, containing at most 6 params(keys) for WillowObject:

            * **ROOT_DIR**: str, directory of data

            * **CLASSES**: list, data classes

            * **TRAIN_NUM**: int, number of images for train in each class

            * **SPLIT_OFFSET**: int, offset when split train and testing set

            * **TRAIN_SAME_AS_TEST**: bool, whether to use same images for training and test

            * **RAND_OUTLIER**: int, number of added outliers in one image
    """
    def __init__(self, sets, obj_resize, **ds_dict):
        CLASSES = dataset_cfg.WillowObject.CLASSES
        KPT_LEN = dataset_cfg.WillowObject.KPT_LEN
        ROOT_DIR = dataset_cfg.WillowObject.ROOT_DIR
        TRAIN_NUM = dataset_cfg.WillowObject.TRAIN_NUM
        SPLIT_OFFSET = dataset_cfg.WillowObject.SPLIT_OFFSET
        TRAIN_SAME_AS_TEST = dataset_cfg.WillowObject.TRAIN_SAME_AS_TEST
        RAND_OUTLIER = dataset_cfg.WillowObject.RAND_OUTLIER
        URL = 'http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip'
        if len(ds_dict.keys()) > 0:
            if 'CLASSES' in ds_dict.keys():
                CLASSES = ds_dict['CLASSES']
            if 'ROOT_DIR' in ds_dict.keys():
                ROOT_DIR = ds_dict['ROOT_DIR']
            if 'TRAIN_NUM' in ds_dict.keys():
                TRAIN_NUM = ds_dict['TRAIN_NUM']
            if 'SPLIT_OFFSET' in ds_dict.keys():
                SPLIT_OFFSET = ds_dict['SPLIT_OFFSET']
            if 'TRAIN_SAME_AS_TEST' in ds_dict.keys():
                TRAIN_SAME_AS_TEST = ds_dict['TRAIN_SAME_AS_TEST']
            if 'RAND_OUTLIER' in ds_dict.keys():
                RAND_OUTLIER = ds_dict['RAND_OUTLIER']
            if 'URL' in ds_dict.keys():
                URL = ds_dict['URL']

        self.dataset_dir = 'data/WillowObject'
        if not os.path.exists(ROOT_DIR):
            assert ROOT_DIR == dataset_cfg.WillowObject.ROOT_DIR, 'you should not change ROOT_DIR unless the data have been manually downloaded'
            self.download(url=URL)

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.sets = sets
        self.obj_resize = obj_resize
        self.suffix = 'willow-' + str(RAND_OUTLIER)

        self.classes = CLASSES
        self.kpt_len = [KPT_LEN for _ in self.classes]

        self.root_path = Path(ROOT_DIR)
        self.obj_resize = obj_resize

        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        self.split_offset = SPLIT_OFFSET
        self.train_len = TRAIN_NUM
        self.train_same_as_test = TRAIN_SAME_AS_TEST
        self.rand_outlier = RAND_OUTLIER

        self.mat_list = []

        self.process()

    def download(self, url=None, retries=5):
        r"""
         Automatically download WillowObject dataset.

         :param url: str, web url of WillowObject
         """
        if retries <= 0:
            raise RuntimeError('Max Retries exceeded!')

        dirs = 'data/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        print('Downloading dataset WillowObject...')
        filename = "data/WILLOW.zip"
        download(filename=filename, url=url, to_cache=False)
        try:
            fz = zipfile.ZipFile(filename, "r")
        except zipfile.BadZipFile as err:
            print('Warning: Content error. Retrying...\n', err)
            os.remove(filename)
            return self.download(url, retries - 1)

        print('Unzipping files...')
        sleep(0.5)
        for file in tqdm(fz.namelist()):
            fz.extract(file, "data/WillowObject/")
        os.remove(filename)
        return filename

    def process(self):
        r"""
        Process the dataset and generate ``data-(size, size).json`` for preprocessed dataset, ``train.json`` for
        training set, and ``test.json`` for testing set.
        """
        train_file = os.path.join(self.dataset_dir, 'train.json')
        test_file = os.path.join(self.dataset_dir, 'test.json')
        img_file = os.path.join(self.dataset_dir, 'data-' + str(self.obj_resize) + '-' + self.suffix + '.json')

        data_list = []
        mat_list_ = []
        for cls_name in self.classes:
            assert type(cls_name) is str
            cls_mat_list = [p for p in (self.root_path / cls_name).glob('*.mat')]
            cls_mat_list.sort()
            if cls_name == 'Face':
                cls_mat_list.remove(self.root_path / cls_name / 'image_0160.mat')
                assert not self.root_path / cls_name / 'image_0160.mat' in cls_mat_list
            ori_len = len(cls_mat_list)
            assert ori_len > 0, 'No data found for WillowObject Class. Is the dataset installed correctly?'
            data_list.append(cls_mat_list)
            if self.split_offset % ori_len + self.train_len <= ori_len:
                if self.sets == 'train' and not self.train_same_as_test:
                    self.mat_list.append(
                        cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    )

                    mat_list_.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
                elif self.train_same_as_test:
                    self.mat_list.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )

                    mat_list_.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[:self.split_offset % ori_len] +
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len:]
                    )

                    mat_list_.append(
                        cls_mat_list[self.split_offset % ori_len: (self.split_offset + self.train_len) % ori_len]
                    )
            else:
                if self.sets == 'train' and not self.train_same_as_test:
                    self.mat_list.append(
                        cls_mat_list[:(self.split_offset + self.train_len) % ori_len - ori_len] +
                        cls_mat_list[self.split_offset % ori_len:]
                    )

                    mat_list_.append(
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )
                elif self.train_same_as_test:
                    self.mat_list.append(
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )

                    mat_list_.append(
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )
                else:
                    self.mat_list.append(
                        cls_mat_list[(self.split_offset + self.train_len) % ori_len - ori_len: self.split_offset % ori_len]
                    )

                    mat_list_.append(
                        cls_mat_list[:(self.split_offset + self.train_len) % ori_len - ori_len] +
                        cls_mat_list[self.split_offset % ori_len:]
                    )

        train_list = []
        test_list = []
        if self.sets == 'train':
            for x in range(len(self.mat_list)):
                for name in self.mat_list[x]:
                    tmp = os.path.split(str(name))
                    objID = tmp[-1].split('.')[0]
                    train_list.append(objID)
            for x in range(len(mat_list_)):
                for name in mat_list_[x]:
                    tmp = os.path.split(str(name))
                    objID = tmp[-1].split('.')[0]
                    test_list.append(objID)
        else:
            for x in range(len(self.mat_list)):
                for name in self.mat_list[x]:
                    tmp = os.path.split(str(name))
                    objID = tmp[-1].split('.')[0]
                    test_list.append(objID)
            for x in range(len(mat_list_)):
                for name in mat_list_[x]:
                    tmp = os.path.split(str(name))
                    objID = tmp[-1].split('.')[0]
                    train_list.append(objID)
        str1 = json.dumps(train_list)
        f1 = open(train_file, 'w')
        f1.write(str1)
        f1.close()
        str2 = json.dumps(test_list)
        f2 = open(test_file, 'w')
        f2.write(str2)
        f2.close()

        if not os.path.exists(img_file):
            data_dict = dict()

            for x in range(len(data_list)):
                for name in data_list[x]:
                    tmp = os.path.split(str(name))
                    objID = tmp[-1].split('.')[0]
                    cls = os.path.split(tmp[0])[-1]
                    annotations = self.__get_anno_dict(name, cls)
                    data_dict[objID] = annotations

            data_str = json.dumps(data_dict)
            f3 = open(img_file, 'w')
            f3.write(data_str)
            f3.close()

    def __get_anno_dict(self, mat_file, cls):
        """
        Get an annotation dict from .mat annotation
        """
        assert mat_file.exists(), '{} does not exist.'.format(mat_file)

        img_name = mat_file.stem + '.png'
        img_file = mat_file.parent / img_name

        struct = sio.loadmat(mat_file.open('rb'))
        kpts = struct['pts_coord']

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]

        keypoint_list = []
        for idx, keypoint in enumerate(np.split(kpts, kpts.shape[1], axis=1)):
            attr = {'labels': idx}
            attr['x'] = float(keypoint[0]) * self.obj_resize[0] / w
            attr['y'] = float(keypoint[1]) * self.obj_resize[1] / h
            keypoint_list.append(attr)

        for idx in range(self.rand_outlier):
            attr = {
                'labels': 'outlier',
                'x': random.uniform(0, self.obj_resize[0]),
                'y': random.uniform(0, self.obj_resize[1])
            }
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['path'] = str(img_file)
        anno_dict['kpts'] = keypoint_list
        anno_dict['bounds'] = [xmin, ymin, w, h]
        anno_dict['cls'] = cls
        anno_dict['univ_size'] = 10

        return anno_dict


class SPair71k:
    r"""
    Download and preprocess **SPair71k** dataset.

    :param sets: str, problem set, ``'train'`` for training set and ``'test'`` for testing set
    :param obj_resize: tuple, resized image size
    :param problem: str, problem type, only ``'2GM'`` is supported in SPair71k
    :param ds_dict: settings of dataset, containing at most 5 params(keys) for SPair71k:

        * **ROOT_DIR**: str, directory of data

        * **TRAIN_DIFF_PARAMS**: list of images that should be dumped in train set

        * **EVAL_DIFF_PARAMS**: list of images that should be dumped in testing set

        * **COMB_CLS**: bool, whether to combine images in different classes

        * **SIZE**: str, ``'large'`` for SPair71k-large and ``'small'`` for SPair71k-small
    """
    def __init__(self, sets, obj_resize, problem='2GM', **ds_dict):
        TRAIN_DIFF_PARAMS = dataset_cfg.SPair.TRAIN_DIFF_PARAMS
        EVAL_DIFF_PARAMS = dataset_cfg.SPair.EVAL_DIFF_PARAMS
        COMB_CLS = dataset_cfg.SPair.COMB_CLS
        SIZE = dataset_cfg.SPair.SIZE
        ROOT_DIR = dataset_cfg.SPair.ROOT_DIR
        if len(ds_dict.keys()) > 0:
            if 'TRAIN_DIFF_PARAMS' in ds_dict.keys():
                TRAIN_DIFF_PARAMS = ds_dict['TRAIN_DIFF_PARAMS']
            if 'EVAL_DIFF_PARAMS' in ds_dict.keys():
                EVAL_DIFF_PARAMS = ds_dict['EVAL_DIFF_PARAMS']
            if 'COMB_CLS' in ds_dict.keys():
                COMB_CLS = ds_dict['COMB_CLS']
            if 'SIZE' in ds_dict.keys():
                SIZE = ds_dict['SIZE']
            if 'ROOT_DIR' in ds_dict.keys():
                ROOT_DIR = ds_dict['ROOT_DIR']

        SPair71k_pair_ann_path = ROOT_DIR + "/PairAnnotation"
        SPair71k_image_path = ROOT_DIR + "/JPEGImages"
        SPair71k_image_annotation = ROOT_DIR + "/ImageAnnotation"
        self.SPair71k_layout_path = ROOT_DIR + "/Layout"
        self.SPair71k_dataset_size = SIZE
        self.suffix = 'spair-' + SIZE

        sets_translation_dict = dict(train="trn", test="test")
        difficulty_params_dict = dict(
            trn=TRAIN_DIFF_PARAMS, val=EVAL_DIFF_PARAMS, test=EVAL_DIFF_PARAMS
        )

        assert not problem == 'MGM', 'No match found for problem {} in SPair-71k'.format(problem)
        self.dataset_dir = 'data/SPair-71k'
        if not os.path.exists(SPair71k_image_path):
            assert ROOT_DIR == dataset_cfg.SPair.ROOT_DIR, 'you should not change ROOT_DIR unless the data have been manually downloaded'
            self.download(url='http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz')

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        self.obj_resize = obj_resize
        self.sets = sets_translation_dict[sets]
        self.ann_files = open(os.path.join(self.SPair71k_layout_path, self.SPair71k_dataset_size, self.sets + ".txt"), "r").read().split("\n")
        self.ann_files = self.ann_files[: len(self.ann_files) - 1]
        self.difficulty_params = difficulty_params_dict[self.sets]
        self.pair_ann_path = SPair71k_pair_ann_path
        self.image_path = SPair71k_image_path
        self.image_annoation = Path(SPair71k_image_annotation)
        self.classes = list(map(lambda x: os.path.basename(x), glob.glob("%s/*" % SPair71k_image_path)))
        self.classes.sort()
        self.combine_classes = COMB_CLS
        self.ann_files_filtered, self.ann_files_filtered_cls_dict, _ = self.__filter_annotations(
            self.ann_files, self.difficulty_params
        )
        self.total_size = len(self.ann_files_filtered)
        self.size_by_cls = {cls: len(ann_list) for cls, ann_list in self.ann_files_filtered_cls_dict.items()}

        self.process()

    def download(self, url=None, retries=5):
        r"""
         Automatically download SPair71k dataset.

         :param url: str, web url of SPair71k
         """
        if retries <= 0:
            raise RuntimeError('Max Retries exceeded!')

        dirs = 'data/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        print('Downloading dataset SPair-71k...')
        filename = "data/SPair-71k.tgz"
        download(filename=filename, url=url, to_cache=False)
        try:
            tar = tarfile.open(filename, "r")
        except tarfile.ReadError as err:
            print('Warning: Content error. Retrying...\n', err)
            os.remove(filename)
            return self.download(url, retries - 1)

        file_names = tar.getnames()
        print('Unzipping files...')
        sleep(0.5)
        for file_name in tqdm(file_names):
            tar.extract(file_name, "data/")
        tar.close()
        os.remove(filename)
        return filename

    def process(self):
        r"""
        Process the dataset and generate ``data-(size, size).json`` for preprocessed dataset, ``train.json`` for
        training set, and ``test.json`` for testing set.
        """
        train_file = os.path.join(self.dataset_dir, 'train.json')
        test_file = os.path.join(self.dataset_dir, 'test.json')
        img_file = os.path.join(self.dataset_dir, 'data-' + str(self.obj_resize) + '-' + self.suffix + '.json')
        if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
            train_list = []
            test_list = []
            if self.sets == 'trn':
                for x in self.ann_files_filtered:
                    tmp = x.split('-')
                    tmp2 = tmp[2].split(':')
                    id1 = tmp[1] + '_' + tmp2[1]
                    id2 = tmp2[0] + '_' + tmp2[1]
                    pair_tuple = (id1, id2)
                    train_list.append(pair_tuple)

                str1 = json.dumps(train_list)
                f1 = open(train_file, 'w')
                f1.write(str1)
                f1.close()
            else:
                for x in self.ann_files_filtered:
                    tmp = x.split('-')
                    tmp2 = tmp[2].split(':')
                    id1 = tmp[1] + '_' + tmp2[1]
                    id2 = tmp2[0] + '_' + tmp2[1]
                    pair_tuple = (id1, id2)
                    test_list.append(pair_tuple)

                str2 = json.dumps(test_list)
                f2 = open(test_file, 'w')
                f2.write(str2)
                f2.close()

            data_list = []
            data_dict = dict()
            for cls_name in self.classes:
                cls_json_list = [p for p in (self.image_annoation / cls_name).glob('*.json')]
                ori_len = len(cls_json_list)
                assert ori_len > 0, 'No data found for SPair-71k. Is the dataset installed correctly?'
                data_list.append(cls_json_list)

            for x in range(len(data_list)):
                for name in data_list[x]:
                    tmp = os.path.split(str(name))
                    objID = tmp[-1].split('.')[0]
                    cls = os.path.split(tmp[0])[-1]
                    annotations = self.__get_anno_dict(name, cls)
                    ID = objID + '_' + cls
                    data_dict[ID] = annotations

            data_str = json.dumps(data_dict)
            f3 = open(img_file, 'w')
            f3.write(data_str)
            f3.close()

    def __get_anno_dict(self, anno_file, cls):
        assert anno_file.exists(), '{} does not exist.'.format(anno_file)

        img_file = self.image_path + '/' + cls + '/' + anno_file.stem + '.jpg'

        with open(anno_file) as f:
            annotations = json.load(f)

        h = float(annotations['image_height'])
        w = float(annotations['image_width'])
        boundbox = annotations['bndbox']

        keypoint_list = []
        for key, value in annotations['kps'].items():
            if not value == None:
                x = (value[0] - boundbox[0]) * self.obj_resize[0] / (boundbox[2] - boundbox[0])
                y = (value[1] - boundbox[1]) * self.obj_resize[1] / (boundbox[3] - boundbox[1])
                kpts_anno = dict()
                kpts_anno['labels'] = key
                kpts_anno['x'] = x
                kpts_anno['y'] = y
                keypoint_list.append(kpts_anno)

        anno_dict = dict()
        anno_dict['kpts'] = keypoint_list
        anno_dict['path'] = img_file
        anno_dict['cls'] = cls
        anno_dict['bounds'] = boundbox
        anno_dict['univ_size'] = len(VOC2011_KPT_NAMES[cls])

        return anno_dict

    def __filter_annotations(self, ann_files, difficulty_params):
        if len(difficulty_params) > 0:
            basepath = os.path.join(self.pair_ann_path, "pickled", self.sets)
            if not os.path.exists(basepath):
                os.makedirs(basepath)
            difficulty_paramas_str = self.__diff_dict_to_str(difficulty_params)
            try:
                filepath = os.path.join(basepath, difficulty_paramas_str + ".pickle")
                ann_files_filtered = pickle.load(open(filepath, "rb"))
                print(
                    f"Found filtered annotations for difficulty parameters {difficulty_params} and {self.sets}-set at {filepath}"
                )
            except (OSError, IOError) as e:
                print(
                    f"No pickled annotations found for difficulty parameters {difficulty_params} and {self.sets}-set. Filtering..."
                )
                ann_files_filtered_dict = {}

                for ann_file in ann_files:
                    with open(os.path.join(self.pair_ann_path, self.sets, ann_file + ".json")) as f:
                        annotation = json.load(f)
                    diff = {key: annotation[key] for key in self.difficulty_params.keys()}
                    diff_str = self.__diff_dict_to_str(diff)
                    if diff_str in ann_files_filtered_dict:
                        ann_files_filtered_dict[diff_str].append(ann_file)
                    else:
                        ann_files_filtered_dict[diff_str] = [ann_file]
                total_l = 0
                for diff_str, file_list in ann_files_filtered_dict.items():
                    total_l += len(file_list)
                    filepath = os.path.join(basepath, diff_str + ".pickle")
                    pickle.dump(file_list, open(filepath, "wb"))
                assert total_l == len(ann_files)
                print(f"Done filtering. Saved filtered annotations to {basepath}.")
                ann_files_filtered = ann_files_filtered_dict[difficulty_paramas_str]
        else:
            print(f"No difficulty parameters for {self.sets}-set. Using all available data.")
            ann_files_filtered = ann_files

        ann_files_filtered_cls_dict = {
            cls: list(filter(lambda x: cls in x, ann_files_filtered)) for cls in self.classes
        }
        class_len = {cls: len(ann_list) for cls, ann_list in ann_files_filtered_cls_dict.items()}
        print(f"Number of annotation pairs matching the difficulty params in {self.sets}-set: {class_len}")
        if self.combine_classes:
            cls_name = "combined"
            ann_files_filtered_cls_dict = {cls_name: ann_files_filtered}
            filtered_classes = [cls_name]
            print(f"Combining {self.sets}-set classes. Total of {len(ann_files_filtered)} image pairs used.")
        else:
            filtered_classes = []
            for cls, ann_f in ann_files_filtered_cls_dict.items():
                if len(ann_f) > 0:
                    filtered_classes.append(cls)
                else:
                    print(f"Excluding class {cls} from {self.sets}-set.")
        return ann_files_filtered, ann_files_filtered_cls_dict, filtered_classes

    def __diff_dict_to_str(self, diff):
        diff_str = ""
        keys = ["mirror", "viewpoint_variation", "scale_variation", "truncation", "occlusion"]
        for key in keys:
            if key in diff.keys():
                diff_str += key
                diff_str += str(diff[key])
        return diff_str


class IMC_PT_SparseGM:
    r"""
    Download and preprocess **IMC_PT_SparseGM** dataset.

    :param sets: str, problem set, ``'train'`` for training set and ``'test'`` for testing set
    :param obj_resize: tuple, resized image size
    :param ds_dict: settings of dataset, containing at most 4 params(keys) for IMC_PT_SparseGM:

        * **ROOT_DIR_IMG**: str, directory of images

        * **ROOT_DIR_NPZ**: str, directory of annotations

        * **CLASSES**: dict, classes of training and test data, keys: ``'train'`` for training and ``'test'`` for test

        * **MAX_KPT_NUM**: int, maximum kpt_num in an image
    """
    def __init__(self, sets, obj_resize, **ds_dict):
        assert sets in ('train', 'test'), 'No match found for dataset {}'.format(sets)
        MAX_KPT_NUM = dataset_cfg.IMC_PT_SparseGM.MAX_KPT_NUM
        CLASSES = dataset_cfg.IMC_PT_SparseGM.CLASSES
        ROOT_DIR_NPZ = dataset_cfg.IMC_PT_SparseGM.ROOT_DIR_NPZ
        ROOT_DIR_IMG = dataset_cfg.IMC_PT_SparseGM.ROOT_DIR_IMG
        URL = 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1Po9pRMWXTqKK2ABPpVmkcsOq-6K_2v-B'
        if len(ds_dict.keys()) > 0:
            if 'MAX_KPT_NUM' in ds_dict.keys():
                MAX_KPT_NUM = ds_dict['MAX_KPT_NUM']
            if 'CLASSES' in ds_dict.keys():
                CLASSES = ds_dict['CLASSES']
            if 'ROOT_DIR_NPZ' in ds_dict.keys():
                ROOT_DIR_NPZ = ds_dict['ROOT_DIR_NPZ']
            if 'ROOT_DIR_IMG' in ds_dict.keys():
                ROOT_DIR_IMG = ds_dict['ROOT_DIR_IMG']
            if 'URL' in ds_dict.keys():
                URL = ds_dict['URL']

        self.dataset_dir = 'data/IMC-PT-SparseGM'
        if not os.path.exists(ROOT_DIR_IMG):
            assert ROOT_DIR_IMG == dataset_cfg.IMC_PT_SparseGM.ROOT_DIR_IMG, 'you should not change ROOT_DIR_IMG or ROOT_DIR_NPZ unless the data have been manually downloaded'
            assert ROOT_DIR_NPZ == dataset_cfg.IMC_PT_SparseGM.ROOT_DIR_NPZ, 'you should not change ROOT_DIR_IMG or ROOT_DIR_NPZ unless the data have been manually downloaded'
            self.download(url=URL)

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        self.sets = sets
        self.classes = CLASSES[sets]
        self.class_dict = CLASSES
        self.max_kpt_num = MAX_KPT_NUM
        self.suffix = 'imcpt-' + str(MAX_KPT_NUM)

        self.root_path_npz = Path(ROOT_DIR_NPZ)
        self.root_path_img = Path(ROOT_DIR_IMG)
        self.obj_resize = obj_resize

        self.img_lists = [np.load(self.root_path_npz / cls / 'img_info.npz')['img_name'].tolist()
                          for cls in self.classes]

        self.process()

    def download(self, url=None, retries=15):
        r"""
         Automatically download IMC_PT_SparseGM dataset.

         :param url: str, web url of IMC_PT_SparseGM
         """
        if retries <= 0:
            raise RuntimeError('Max Retries exceeded!')

        dirs = 'data/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        print('Downloading dataset IMC-PT-SparseGM...')
        filename = 'data/IMC-PT-SparseGM.tar.gz'
        download(filename=filename, url=url, to_cache=False)
        try:
            tar = tarfile.open(filename, "r")
        except tarfile.ReadError as err:
            print('Warning: Content error. Retrying...\n', err)
            os.remove(filename)
            return self.download(url, retries - 1)

        file_names = tar.getnames()
        print('Unzipping files...')
        sleep(0.5)
        for file_name in tqdm(file_names):
            tar.extract(file_name, "data/")
        tar.close()
        os.remove(filename)
        return filename

    def process(self):
        r"""
        Process the dataset and generate ``data-(size, size).json`` for preprocessed dataset, ``train.json`` for
        training set, and ``test.json`` for testing set.
        """
        set_file = os.path.join(self.dataset_dir, self.sets + '.json')
        img_file = os.path.join(self.dataset_dir, 'data-' + str(self.obj_resize) + '-' + self.suffix + '.json')

        if not os.path.exists(set_file):
            set_list = []
            for _list in self.img_lists:
                for img_name in _list:
                    set_list.append(img_name.split('.')[0])
            str1 = json.dumps(set_list)
            f1 = open(set_file, 'w')
            f1.write(str1)
            f1.close()

        if not os.path.exists(img_file):
            total_cls = []
            for cls in self.class_dict['train']:
                total_cls.append(cls)
            for cls in self.class_dict['test']:
                total_cls.append(cls)

            total_img_lists = [np.load(self.root_path_npz / cls / 'img_info.npz')['img_name'].tolist()
                               for cls in total_cls]
            data_dict = dict()
            for i, _list in enumerate(total_img_lists):
                cls = total_cls[i]
                for img_name in _list:
                    img_id = img_name.split('.')[0]
                    anno_dict = self.__get_anno_dict(img_name, cls)
                    data_dict[img_id] = anno_dict

            str2 = json.dumps(data_dict)
            f2 = open(img_file, 'w')
            f2.write(str2)
            f2.close()

    def __get_anno_dict(self, img_name, cls):
        """
        Get an annotation dict from ``.npz`` annotation
        """
        img_file = self.root_path_img / cls / img_name
        npz_file = self.root_path_npz / cls / (img_name.split('.')[0] + '.npz')

        assert img_file.exists(), '{} does not exist.'.format(img_file)
        assert npz_file.exists(), '{} does not exist.'.format(npz_file)

        with Image.open(str(img_file)) as img:
            ori_sizes = img.size
            xmin = 0
            ymin = 0
            w = ori_sizes[0]
            h = ori_sizes[1]

        with np.load(str(npz_file)) as npz_anno:
            kpts = npz_anno['points']
            if len(kpts.shape) != 2:
                ValueError('{} contains no keypoints.'.format(img_file))

        keypoint_list = []
        for i in range(kpts.shape[1]):
            kpt_index = int(kpts[0, i])
            assert kpt_index < self.max_kpt_num
            attr = {
                'labels': kpt_index,
                'x': kpts[1, i] * self.obj_resize[0] / w,
                'y': kpts[2, i] * self.obj_resize[1] / h
            }
            keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['path'] = str(img_file)
        anno_dict['kpts'] = keypoint_list
        anno_dict['bounds'] = [xmin, ymin, w, h]
        anno_dict['cls'] = cls
        anno_dict['univ_size'] = self.max_kpt_num

        return anno_dict

    def __len(self, cls):
        if type(cls) == int:
            cls = self.classes[cls]
        assert cls in self.classes
        return len(self.img_lists[self.classes.index(cls)])


class CUB2011:
    r"""
    Download and preprocess **CUB2011** dataset.

    :param sets: str, problem set, ``'train'`` for training set and ``'test'`` for testing set
    :param obj_resize: tuple, resized image size
    :param ds_dict: settings of dataset, containing at most 1 params(key) for CUB2011:

        * **ROOT_DIR**: str, directory of data
    """
    def __init__(self, sets, obj_resize, **ds_dict):
        CLS_SPLIT = dataset_cfg.CUB2011.CLASS_SPLIT
        ROOT_DIR = dataset_cfg.CUB2011.ROOT_DIR
        URL = 'https://drive.google.com/u/0/uc?export=download&confirm=B8eu&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
        if len(ds_dict.keys()) > 0:
            if 'ROOT_DIR' in ds_dict.keys():
                ROOT_DIR = ds_dict['ROOT_DIR']
            if 'URL' in ds_dict.keys():
                URL = ds_dict['URL']

        self.set_data = {'train': [], 'test': []}
        self.classes = []

        self._set_pairs = {}
        self._set_mask = {}
        self.cls_split = CLS_SPLIT
        self.suffix = 'cub2011'

        self.rootpath = ROOT_DIR

        self.dataset_dir = 'data/CUB_200_2011'
        if not os.path.exists(ROOT_DIR):
            assert ROOT_DIR == dataset_cfg.CUB2011.ROOT_DIR, 'you should not change ROOT_DIR unless the data have been manually downloaded'
            self.download(url=URL)

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        with open(os.path.join(self.rootpath, 'images.txt')) as f:
            self.im2fn = dict(l.rstrip('\n').split() for l in f.readlines())
        with open(os.path.join(self.rootpath, 'train_test_split.txt')) as f:
            train_split = dict(l.rstrip('\n').split() for l in f.readlines())
        with open(os.path.join(self.rootpath, 'classes.txt')) as f:
            classes = dict(l.rstrip('\n').split() for l in f.readlines())
        with open(os.path.join(self.rootpath, 'image_class_labels.txt')) as f:
            img2class = [l.rstrip('\n').split() for l in f.readlines()]
            img_idxs, class_idxs = map(list, zip(*img2class))
            class2img = self.__lists2dict_for_cub(class_idxs, img_idxs)
        with open(os.path.join(self.rootpath, 'parts', 'part_locs.txt')) as f:
            part_locs = [l.rstrip('\n').split() for l in f.readlines()]
            fi, pi, x, y, v = map(list, zip(*part_locs))
            self.im2kpts = self.__lists2dict_for_cub(fi, zip(pi, x, y, v))
        with open(os.path.join(self.rootpath, 'bounding_boxes.txt')) as f:
            bboxes = [l.rstrip('\n').split() for l in f.readlines()]
            ii, x, y, w, h = map(list, zip(*bboxes))
            self.im2bbox = dict(zip(ii, zip(x, y, w, h)))
        if self.cls_split == 'ori':
            for class_idx in sorted(classes):
                self.classes.append(classes[class_idx])
                train_set = []
                test_set = []
                for img_idx in class2img[class_idx]:
                    if train_split[img_idx] == '1':
                        train_set.append(img_idx)
                    else:
                        test_set.append(img_idx)
                self.set_data['train'].append(train_set)
                self.set_data['test'].append(test_set)
        self.sets = sets
        self.obj_resize = obj_resize

        self.process()

    def download(self, url=None, retries=50):
        r"""
         Automatically download CUB2011 dataset.

         :param url: str, web url of CUB2011
         """
        if retries <= 0:
            raise RuntimeError('Max Retries exceeded!')

        dirs = 'data/'
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        print('Downloading dataset CUB2011...')
        filename = 'data/CUB_200_2011.tgz'
        download(filename=filename, url=url, to_cache=False)
        try:
            tar = tarfile.open(filename, "r")
        except tarfile.ReadError as err:
            print('Warning: Content error. Retrying...\n', err)
            os.remove(filename)
            return self.download(url, retries - 1)

        file_names = tar.getnames()
        print('Unzipping files...')
        sleep(0.5)
        for file_name in tqdm(file_names):
            tar.extract(file_name, "data/")
        tar.close()
        os.remove(filename)
        return filename

    def process(self):
        r"""
        Process the dataset and generate ``data-(size, size).json`` for preprocessed dataset, ``train.json`` for
        training set, and ``test.json`` for testing set.
        """
        set_file = os.path.join(self.dataset_dir, self.sets + '.json')
        img_file = os.path.join(self.dataset_dir, 'data-' + str(self.obj_resize) + '-' + self.suffix + '.json')

        if not os.path.exists(set_file):
            set_list = []
            set_img_idx_list = self.set_data[self.sets]
            for cls_img_idx_list in set_img_idx_list:
                for img_idx in cls_img_idx_list:
                    img_name = self.im2fn[img_idx].split('/')[-1].split('.')[0]
                    set_list.append(img_name)

            str1 = json.dumps(set_list)
            f1 = open(set_file, 'w')
            f1.write(str1)
            f1.close()

        if not os.path.exists(img_file):
            data_dict = dict()
            for img_idx, img_name in self.im2fn.items():
                cls = img_name.split('/')[0]
                obj_id = img_name.split('/')[-1].split('.')[0]
                obj_dict = self.__get_anno_dict(img_idx, cls)
                data_dict[obj_id] = obj_dict

            str2 = json.dumps(data_dict)
            f2 = open(img_file, 'w')
            f2.write(str2)
            f2.close()

    def __get_imgname(self, data):
        return os.path.join(self.rootpath, 'images', self.im2fn[data])

    def __get_meta(self, data):
        pi, x, y, v = map(list, zip(*self.im2kpts[data]))
        order = np.argsort(np.array(pi).astype(int))
        keypts = np.array([np.array(x).astype('float')[order],
                           np.array(y).astype('float')[order]])
        visible = np.array(v).astype('uint8')[order]
        bbox = np.array(self.im2bbox[data]).astype(float)
        return keypts, visible, bbox

    def __get_anno_dict(self, img_name, cls):
        keypts, visible, bbox = self.__get_meta(img_name)
        xmin, ymin, w, h = bbox
        img_file = self.__get_imgname(img_name)
        with Image.open(str(img_file)) as img:
            xmin, xmax = np.clip((xmin, xmin + w), 0, img.size[0])
            ymin, ymax = np.clip((ymin, ymin + h), 0, img.size[1])

        keypoint_list = []
        for keypt_idx in range(keypts.shape[1]):
            if visible[keypt_idx]:
                attr = dict()
                attr['labels'] = keypt_idx
                attr['x'] = (keypts[0, keypt_idx] - xmin) * self.obj_resize[0] / w
                attr['y'] = (keypts[1, keypt_idx] - ymin) * self.obj_resize[1] / h
                keypoint_list.append(attr)

        anno_dict = dict()
        anno_dict['path'] = img_file
        anno_dict['kpts'] = keypoint_list
        anno_dict['bounds'] = [xmin, ymin, xmax, ymax]
        anno_dict['cls'] = cls
        anno_dict['univ_size'] = 15

        return anno_dict

    @staticmethod
    def __lists2dict_for_cub(keys, vals):
        ans = {}
        for idx, val_i in enumerate(vals):
            if keys[idx] in ans:
                ans[keys[idx]].append(val_i)
            else:
                ans[keys[idx]] = [val_i]
        return ans
