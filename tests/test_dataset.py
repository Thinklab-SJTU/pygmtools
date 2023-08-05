# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import pygmtools as pygm
from pygmtools.dataset_config import dataset_cfg
from random import choice
import os

# Test dataset download and preprocess, and data fetch and evaluation
def _test_benchmark(name, sets, problem, filter, **ds_dict):
    benchmark = pygm.benchmark.Benchmark(name=name, sets=sets, problem=problem, filter=filter, **ds_dict)
    if sets == 'test':
        num = 2 if benchmark.problem == '2GM' else 3
        _test_get_data(benchmark, num)
    os.remove(benchmark.data_list_path)
    os.remove(benchmark.data_path)


# Test data fetch and evaluation
def _test_get_data(benchmark, num):
    data_list, perm_dict, ids = benchmark.rand_get_data(cls=benchmark.classes[0], num=num)
    rand_data = benchmark.rand_get_data(num=num)
    assert rand_data is not None

    if num == 2:
        data_length = benchmark.compute_length(num=num)
        assert data_length is not None
        cls_data_length = benchmark.compute_length(cls=benchmark.classes[0], num=num)
        assert cls_data_length is not None
        img_num = benchmark.compute_img_num(classes=benchmark.classes)
        assert img_num is not None
        data_id_comb = benchmark.get_id_combination(num=num)
        assert data_id_comb is not None
        cls_data_id_comb = benchmark.get_id_combination(cls=benchmark.classes[0], num=num)
        assert cls_data_id_comb is not None

        perm_mat = perm_dict[(0, 1)].toarray()
        cls = benchmark.classes[0]
        pred = []
        pred_dict = dict()
        pred_dict['ids'] = ids
        pred_dict['cls'] = cls
        pred_dict['perm_mat'] = perm_mat
        pred.append(pred_dict)
        result_cls = benchmark.eval_cls(prediction=pred, cls=benchmark.classes[0], verbose=True)
        assert result_cls['f1'] == 1, f'Accuracy should be 1, something wrong in {benchmark.name} dataset test.'

        result = benchmark.eval(prediction=pred, classes=[benchmark.classes[0]], verbose=True)
        assert result['mean']['f1'] == 1, f'Accuracy should be 1, something wrong in {benchmark.name} dataset test.'

# Entry function
def test_dataset_and_benchmark():
    dataset_name_list = ['PascalVOC', 'WillowObject', 'SPair71k', 'IMC_PT_SparseGM', 'CUB2011']
    problem_type_list = ['2GM', 'MGM']
    set_list = ['train', 'test']
    filter_list = ['intersection', 'inclusion', 'unfiltered']
    dict_list = []
    voc_cfg_dict = dict()
    voc_cfg_dict['KPT_ANNO_DIR'] = dataset_cfg.PascalVOC.KPT_ANNO_DIR
    voc_cfg_dict['ROOT_DIR'] = dataset_cfg.PascalVOC.ROOT_DIR
    voc_cfg_dict['SET_SPLIT'] = dataset_cfg.PascalVOC.SET_SPLIT
    voc_cfg_dict['CLASSES'] = dataset_cfg.PascalVOC.CLASSES
    voc_cfg_dict['CACHE_PATH'] = dataset_cfg.CACHE_PATH
    dict_list.append(voc_cfg_dict)

    willow_cfg_dict = dict()
    willow_cfg_dict['CLASSES'] = dataset_cfg.WillowObject.CLASSES
    willow_cfg_dict['KPT_LEN'] = dataset_cfg.WillowObject.KPT_LEN
    willow_cfg_dict['ROOT_DIR'] = dataset_cfg.WillowObject.ROOT_DIR
    willow_cfg_dict['TRAIN_NUM'] = dataset_cfg.WillowObject.TRAIN_NUM
    willow_cfg_dict['SPLIT_OFFSET'] = dataset_cfg.WillowObject.SPLIT_OFFSET
    willow_cfg_dict['TRAIN_SAME_AS_TEST'] = dataset_cfg.WillowObject.TRAIN_SAME_AS_TEST
    willow_cfg_dict['RAND_OUTLIER'] = dataset_cfg.WillowObject.RAND_OUTLIER
    willow_cfg_dict['URL'] = 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=18AvGwkuhnih5bFDjfJK5NYM16LvDfwW_'
    dict_list.append(willow_cfg_dict)

    spair_cfg_dict = dict()
    spair_cfg_dict['TRAIN_DIFF_PARAMS'] = {'mirror': 0}
    spair_cfg_dict['EVAL_DIFF_PARAMS'] = dataset_cfg.SPair.EVAL_DIFF_PARAMS
    spair_cfg_dict['COMB_CLS'] = True
    spair_cfg_dict['SIZE'] = 'small'
    spair_cfg_dict['ROOT_DIR'] = dataset_cfg.SPair.ROOT_DIR
    dict_list.append(spair_cfg_dict)

    imcpt_cfg_dict = dict()
    imcpt_cfg_dict['MAX_KPT_NUM'] = dataset_cfg.IMC_PT_SparseGM.MAX_KPT_NUM
    imcpt_cfg_dict['CLASSES'] = {'train': ['brandenburg_gate'],
                            'test': ['reichstag']}
    imcpt_cfg_dict['ROOT_DIR_NPZ'] = dataset_cfg.IMC_PT_SparseGM.ROOT_DIR_NPZ
    imcpt_cfg_dict['ROOT_DIR_IMG'] = dataset_cfg.IMC_PT_SparseGM.ROOT_DIR_IMG
    imcpt_cfg_dict['URL'] = 'https://drive.google.com/u/0/uc?id=1bisri2Ip1Of3RsUA8OBrdH5oa6HlH3k-&export=download'
    dict_list.append(imcpt_cfg_dict)

    cub_cfg_dict = dict()
    cub_cfg_dict['ROOT_DIR'] = dataset_cfg.CUB2011.ROOT_DIR
    cub_cfg_dict['URL'] = 'https://drive.google.com/u/0/uc?id=1fcN3m2PmQF7rMQGPxldEICU8CtJ0-F-z&export=download'
    dict_list.append(cub_cfg_dict)

    for i, dataset_name in enumerate(dataset_name_list):
        for set in set_list:
            for problem_type in problem_type_list:
                filter = choice(filter_list)
                if dataset_name == 'SPair71k' and problem_type == 'MGM':
                    continue
                if filter == 'inclusion' and problem_type == 'MGM':
                    continue
                _test_benchmark(dataset_name, set, problem_type, filter, **dict_list[i])


if __name__ == '__main__':
    test_dataset_and_benchmark()
