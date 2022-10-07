"""
The Benchmark module with a unified data interface to evaluate graph matching methods.

If you are interested in the performance and the deep learning framework, please refer to our `ThinkMatch project <https://github.com/Thinklab-SJTU/ThinkMatch>`_.
"""

import tempfile
import shutil
import itertools
from scipy.sparse import coo_matrix
from pygmtools.dataset import *


class Benchmark:
    r"""
    The `Benchmark` module provides a unified data interface and an evaluating platform for different datasets.

    :param name: str, dataset name, currently support ``'PascalVOC'``, ``'WillowObject'``, ``'IMC_PT_SparseGM'``, ``'CUB2011'``, ``'SPair71k'``
    :param sets: str, problem set, ``'train'`` for training set and ``'test'`` for test set
    :param obj_resize: tuple, (default: ``(256, 256)``) resized object size
    :param problem: str, (default: ``'2GM'``) problem type, ``'2GM'`` for 2-graph matching and ``'MGM'`` for multi-graph matching
    :param filter: str, (default: ``'intersection'``) filter of nodes, ``'intersection'`` refers to retaining only common nodes;
       ``'inclusion'`` is only for 2GM and refers to filtering only one graph to make its nodes a subset of the other graph,
       and ``'unfiltered'`` refers to retaining all nodes in all graphs
    :param args: keyword settings for specific dataset
    """

    def __init__(self, name, sets, obj_resize=(256, 256), problem='2GM', filter='intersection', **args):
        assert name == 'PascalVOC' or name == 'SPair71k' or name == 'WillowObject' or name == 'IMC_PT_SparseGM' or name == 'CUB2011', 'No match found for dataset {}'.format(
            name)
        assert problem == '2GM' or problem == 'MGM' or problem == 'MGM3', 'No match found for problem {}'.format(
            problem)
        assert filter == 'intersection' or filter == 'inclusion' or filter == 'unfiltered', 'No match found for filter {}'.format(
            filter)
        assert not ((
                            problem == 'MGM' or problem == 'MGM3') and filter == 'inclusion'), 'The filter inclusion only matches 2GM'

        self.name = name
        self.problem = problem
        self.filter = filter
        self.sets = sets
        self.obj_resize = obj_resize

        data_set = eval(self.name)(self.sets, self.obj_resize, **args)
        self.data_path = os.path.join(data_set.dataset_dir, 'data-' + str(self.obj_resize) + '.json')
        self.data_list_path = os.path.join(data_set.dataset_dir, sets + '.json')
        self.classes = data_set.classes

        with open(self.data_path) as f:
            self.data_dict = json.load(f)

        if self.sets == 'test':
            tmpfile = tempfile.gettempdir()
            pid_num = os.getpid()
            cache_dir = str(pid_num) + '_gt_cache'
            self.gt_cache_path = os.path.join(tmpfile, cache_dir)

            if not os.path.exists(self.gt_cache_path):
                os.mkdir(self.gt_cache_path)
                print('gt perm mat cache built')

    def get_data(self, ids, test=False, shuffle=True):
        r"""
        Fetch a data pair or pairs of data by image ID for training or test.

        :param ids: list of image ID, usually in ``train.json`` or ``test.json``
        :param test: bool, whether the fetched data is used for test; if true, this function will not return ground truth
        :param shuffle: bool, whether to shuffle the order of keypoints
        :return:
                    **data_list**: list of data, like ``[{'img': np.array, 'kpts': coordinates of kpts}, ...]``

                    **perm_mat_dict**: ground truth, like ``{(0,1):scipy.sparse, (0,2):scipy.sparse, ...}``, ``(0,1)`` refers to data pair ``(ids[0],ids[1])``

                    **ids**: list of image ID
        """
        assert (self.problem == '2GM' and len(ids) == 2) or ((self.problem == 'MGM' or self.problem == 'MGM3') and len(
            ids) > 2), '{} problem cannot get {} data'.format(self.problem, len(ids))

        ids.sort()
        data_list = []
        for keys in ids:
            obj_dict = dict()
            boundbox = self.data_dict[keys]['bounds']
            img_file = self.data_dict[keys]['path']
            with Image.open(str(img_file)) as img:
                obj = img.resize(self.obj_resize, resample=Image.BICUBIC,
                                 box=(boundbox[0], boundbox[1], boundbox[2], boundbox[3]))
                if self.name == 'CUB2011':
                    if not obj.mode == 'RGB':
                        obj = obj.convert('RGB')
            obj_dict['img'] = np.array(obj)
            obj_dict['kpts'] = self.data_dict[keys]['kpts']
            obj_dict['cls'] = self.data_dict[keys]['cls']
            obj_dict['univ_size'] = self.data_dict[keys]['univ_size']
            if shuffle:
                random.shuffle(obj_dict['kpts'])
            data_list.append(obj_dict)

        perm_mat_dict = dict()
        id_combination = list(itertools.combinations(list(range(len(ids))), 2))
        for id_tuple in id_combination:
            perm_mat = np.zeros([len(data_list[_]['kpts']) for _ in id_tuple], dtype=np.float32)
            row_list = []
            col_list = []

            for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        if keypoint['labels'] != 'outlier':
                            perm_mat[i, j] = 1
            for i, keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        row_list.append(i)
                        break
            for i, keypoint in enumerate(data_list[id_tuple[1]]['kpts']):
                for j, _keypoint in enumerate(data_list[id_tuple[0]]['kpts']):
                    if keypoint['labels'] == _keypoint['labels']:
                        col_list.append(i)
                        break
            row_list.sort()
            col_list.sort()
            if self.filter == 'intersection':
                perm_mat = perm_mat[row_list, :]
                perm_mat = perm_mat[:, col_list]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
                data_list[id_tuple[1]]['kpts'] = [data_list[id_tuple[1]]['kpts'][i] for i in col_list]
            elif self.filter == 'inclusion':
                perm_mat = perm_mat[row_list, :]
                data_list[id_tuple[0]]['kpts'] = [data_list[id_tuple[0]]['kpts'][i] for i in row_list]
            if not (len(ids) > 2 and self.filter == 'intersection'):
                sparse_perm_mat = coo_matrix(perm_mat)
                perm_mat_dict[id_tuple] = sparse_perm_mat

        if len(ids) > 2 and self.filter == 'intersection':
            for p in range(len(ids) - 1):
                perm_mat_list = [np.zeros([len(data_list[p]['kpts']), len(x['kpts'])], dtype=np.float32) for x in
                                 data_list[p + 1: len(ids)]]
                row_list = []
                col_lists = []
                for i in range(len(ids) - p - 1):
                    col_lists.append([])

                for i, keypoint in enumerate(data_list[p]['kpts']):
                    kpt_idx = []
                    for anno_dict in data_list[p + 1: len(ids)]:
                        kpt_name_list = [x['labels'] for x in anno_dict['kpts']]
                        if keypoint['labels'] in kpt_name_list:
                            kpt_idx.append(kpt_name_list.index(keypoint['labels']))
                        else:
                            kpt_idx.append(-1)
                    row_list.append(i)
                    for k in range(len(ids) - p - 1):
                        j = kpt_idx[k]
                        if j != -1:
                            col_lists[k].append(j)
                            if keypoint['labels'] != 'outlier':
                                perm_mat_list[k][i, j] = 1

                row_list.sort()
                for col_list in col_lists:
                    col_list.sort()

                for k in range(len(ids) - p - 1):
                    perm_mat_list[k] = perm_mat_list[k][row_list, :]
                    perm_mat_list[k] = perm_mat_list[k][:, col_lists[k]]
                    id_tuple = (p, k + p + 1)
                    perm_mat_dict[id_tuple] = coo_matrix(perm_mat_list[k])

        if self.sets == 'test':
            for pair in id_combination:
                id_pair = (ids[pair[0]], ids[pair[1]])
                gt_path = os.path.join(self.gt_cache_path, str(id_pair) + '.npy')
                if not os.path.exists(gt_path):
                    np.save(gt_path, perm_mat_dict[pair])

        if not test:
            return data_list, perm_mat_dict, ids
        else:
            return data_list, ids

    def rand_get_data(self, cls=None, num=2, test=False, shuffle=True):
        r"""
        Randomly fetch data for training or test. Implemented by calling ``get_data`` function.

        :param cls: int or str, class of expected data. None for random class
        :param num: int, number of images; for example, 2 for 2GM
        :param test: bool, whether the fetched data is used for test; if true, this function will not return ground truth
        :param shuffle: bool, whether to shuffle the order of keypoints
        :return:
                    **data_list**: list of data, like ``[{'img': np.array, 'kpts': coordinates of kpts}, ...]``

                    **perm_mat_dict**: ground truth, like ``{(0,1):scipy.sparse, (0,2):scipy.sparse, ...}``, ``(0,1)`` refers to data pair ``(ids[0],ids[1])``

                    **ids**: list of image ID
        """
        if cls == None:
            cls = random.randrange(0, len(self.classes))
            clss = self.classes[cls]
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        data_list = []
        ids = []
        if self.name != 'SPair71k':
            for id in data_id:
                if self.data_dict[id]['cls'] == clss:
                    data_list.append(id)

            for objID in random.sample(data_list, num):
                ids.append(objID)
        else:
            for id in data_id:
                if self.data_dict[id[0]]['cls'] == clss:
                    data_list.append(id)
            ids = random.sample(data_list, 1)[0]

        return self.get_data(ids, test, shuffle)

    def get_id_combination(self, cls=None, num=2):
        r"""
        Get the combination of images and length of combinations in specified class.

        :param cls: int or str, class of expected data. None for all classes
        :param num: int, number of images in each image ID list; for example, 2 for 2GM
        :return:
                **id_combination_list**: list of combinations of image ids

                **length**: length of combinations
        """
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls
        else:
            raise ValueError(f'Expect cls argument to be NoneType or str, got {type(cls)}!')

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0
        id_combination_list = []
        if clss != None:
            data_list = []
            if self.name != 'SPair71k':
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length += len(id_combination)
                id_combination_list.append(id_combination)
            else:
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        data_list.append(id_pair)
                length += len(data_list)
                id_combination_list.append(data_list)
        else:
            for clss in self.classes:
                data_list = []
                if self.name != 'SPair71k':
                    for id in data_id:
                        if self.data_dict[id]['cls'] == clss:
                            data_list.append(id)
                    id_combination = list(itertools.combinations(data_list, num))
                    length += len(id_combination)
                    id_combination_list.append(id_combination)
                else:
                    for id_pair in data_id:
                        if self.data_dict[id_pair[0]]['cls'] == clss:
                            data_list.append(id_pair)
                    length += len(data_list)
                    id_combination_list.append(data_list)

        return id_combination_list, length

    def compute_length(self, cls=None, num=2):
        r"""
        Compute the length of image combinations in specified class.

        :param cls: int or str, class of expected data. None for all classes
        :param num: int, number of images in each image ID list; for example, 2 for two-graph matching problem
        :return: length of combinations
        """
        if cls == None:
            clss = None
        elif type(cls) == str:
            clss = cls

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        length = 0

        if clss != None:
            if self.name != 'SPair71k':
                data_list = []
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        data_list.append(id)
                id_combination = list(itertools.combinations(data_list, num))
                length += len(id_combination)
            else:
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        length += 1

        else:
            for clss in self.classes:
                if self.name != 'SPair71k':
                    data_list = []
                    for id in data_id:
                        if self.data_dict[id]['cls'] == clss:
                            data_list.append(id)
                    id_combination = list(itertools.combinations(data_list, num))
                    length += len(id_combination)
                else:
                    for id_pair in data_id:
                        if self.data_dict[id_pair[0]]['cls'] == clss:
                            length += 1
        return length

    def compute_img_num(self, classes):
        r"""
        Compute number of images in specified classes.

        :param classes: list of dataset classes
        :return: list of numbers of images in each class
        """
        with open(self.data_list_path) as f1:
            data_id = json.load(f1)
        num_list = []
        for clss in classes:
            cls_img_num = 0
            if self.name != 'SPair71k':
                for id in data_id:
                    if self.data_dict[id]['cls'] == clss:
                        cls_img_num += 1
                num_list.append(cls_img_num)
            else:
                img_cache = []
                for id_pair in data_id:
                    if self.data_dict[id_pair[0]]['cls'] == clss:
                        if id_pair[0] not in img_cache:
                            img_cache.append(id_pair[0])
                            cls_img_num += 1
                        if id_pair[1] not in img_cache:
                            img_cache.append(id_pair[1])
                            cls_img_num += 1
                num_list.append(cls_img_num)

        return num_list

    def eval(self, prediction, classes, verbose=False):
        r"""
        Evaluate test results and compute matching accuracy and coverage.

        :param prediction: list, prediction result, like ``[{'ids': (id1, id2), 'cls': cls, 'permmat': np.array or scipy.sparse}, ...]``
        :param classes: list of evaluated classes
        :param verbose: bool, whether to print the result
        :return: evaluation result in each class and their averages, including p, r, f1 and their standard deviation and coverage
        """

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        cls_dict = dict()
        pred_cls_dict = dict()
        result = dict()
        id_cache = []
        cls_precision = dict()
        cls_recall = dict()
        cls_f1 = dict()

        for cls in classes:
            cls_dict[cls] = 0
            pred_cls_dict[cls] = 0
            result[cls] = dict()
            cls_precision[cls] = []
            cls_recall[cls] = []
            cls_f1[cls] = []

        if self.name != 'SPair71k':
            for key, obj in self.data_dict.items():
                if (key in data_id) and (obj['cls'] in classes):
                    cls_dict[obj['cls']] += 1
        else:
            for cls in classes:
                cls_dict[cls] = self.compute_img_num([cls])[0]

        for pair_dict in prediction:
            ids = (pair_dict['ids'][0], pair_dict['ids'][1])
            if ids not in id_cache:
                id_cache.append(ids)
                pred_cls_dict[pair_dict['cls']] += 1
                perm_mat = pair_dict['perm_mat']
                gt_path = os.path.join(self.gt_cache_path, str(ids) + '.npy')
                gt = np.load(gt_path, allow_pickle=True).item()
                gt_array = gt.toarray()
                assert type(perm_mat) == type(gt_array)

                if perm_mat.sum() == 0 or gt_array.sum() == 0:
                    precision = 1
                    recall = 1
                else:
                    precision = (perm_mat * gt_array).sum() / perm_mat.sum()
                    recall = (perm_mat * gt_array).sum() / gt_array.sum()
                if precision == 0 or recall == 0:
                    f1_score = 0
                else:
                    f1_score = (2 * precision * recall) / (precision + recall)

                cls_precision[pair_dict['cls']].append(precision)
                cls_recall[pair_dict['cls']].append(recall)
                cls_f1[pair_dict['cls']].append(f1_score)

        p_sum = 0
        r_sum = 0
        f1_sum = 0
        p_std_sum = 0
        r_std_sum = 0
        f1_std_sum = 0

        for cls in classes:
            result[cls]['precision'] = np.mean(cls_precision[cls])
            result[cls]['recall'] = np.mean(cls_recall[cls])
            result[cls]['f1'] = np.mean(cls_f1[cls])
            result[cls]['precision_std'] = np.std(cls_precision[cls])
            result[cls]['recall_std'] = np.std(cls_recall[cls])
            result[cls]['f1_std'] = np.std(cls_f1[cls])
            result[cls]['coverage'] = 2 * pred_cls_dict[cls] / (cls_dict[cls] * (cls_dict[cls] - 1))
            p_sum += result[cls]['precision']
            r_sum += result[cls]['recall']
            f1_sum += result[cls]['f1']
            p_std_sum += result[cls]['precision_std']
            r_std_sum += result[cls]['recall_std']
            f1_std_sum += result[cls]['f1_std']

        result['mean'] = dict()
        result['mean']['precision'] = p_sum / len(classes)
        result['mean']['recall'] = r_sum / len(classes)
        result['mean']['f1'] = f1_sum / len(classes)
        result['mean']['precision_std'] = p_std_sum / len(classes)
        result['mean']['recall_std'] = r_std_sum / len(classes)
        result['mean']['f1_std'] = f1_std_sum / len(classes)

        if verbose:
            print('Matching accuracy')
            for cls in classes:
                print('{}: {}'.format(cls, 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}, cvg = {:.4f}' \
                                      .format(result[cls]['precision'], result[cls]['precision_std'],
                                              result[cls]['recall'], result[cls]['recall_std'], result[cls]['f1'],
                                              result[cls]['f1_std'], result[cls]['coverage']
                                              )))
            print('average accuracy: {}'.format('p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}' \
                                                .format(result['mean']['precision'], result['mean']['precision_std'],
                                                        result['mean']['recall'], result['mean']['recall_std'],
                                                        result['mean']['f1'], result['mean']['f1_std']
                                                        )))
        return result

    def eval_cls(self, prediction, cls, verbose=False):
        r"""
        Evaluate test results and compute matching accuracy and coverage on one specified class.

        :param prediction: list, prediction result on one class, like ``[{'ids': (id1, id2), 'cls': cls, 'permmat': np.array or scipy.sparse}, ...]``
        :param cls: str, evaluated class
        :param verbose: bool, whether to print the result
        :return: evaluation result on the specified class, including p, r, f1 and their standard deviation and coverage
        """

        with open(self.data_list_path) as f1:
            data_id = json.load(f1)

        result = dict()
        id_cache = []
        cls_precision = []
        cls_recall = []
        cls_f1 = []

        cls_dict = 0
        pred_cls_dict = 0

        if self.name != 'SPair71k':
            for key, obj in self.data_dict.items():
                if (key in data_id) and (obj['cls'] == cls):
                    cls_dict += 1
        else:
            cls_dict = self.compute_img_num([cls])[0]

        for pair_dict in prediction:
            ids = (pair_dict['ids'][0], pair_dict['ids'][1])
            if ids not in id_cache:
                id_cache.append(ids)
                pred_cls_dict += 1
                perm_mat = pair_dict['perm_mat']
                gt_path = os.path.join(self.gt_cache_path, str(ids) + '.npy')
                gt = np.load(gt_path, allow_pickle=True).item()
                gt_array = gt.toarray()
                assert type(perm_mat) == type(gt_array)

                if perm_mat.sum() == 0 or gt_array.sum() == 0:
                    precision = 1
                    recall = 1
                else:
                    precision = (perm_mat * gt_array).sum() / perm_mat.sum()
                    recall = (perm_mat * gt_array).sum() / gt_array.sum()
                if precision == 0 or recall == 0:
                    f1_score = 0
                else:
                    f1_score = (2 * precision * recall) / (precision + recall)

                cls_precision.append(precision)
                cls_recall.append(recall)
                cls_f1.append(f1_score)

        result['precision'] = np.mean(cls_precision)
        result['recall'] = np.mean(cls_recall)
        result['f1'] = np.mean(cls_f1)
        result['precision_std'] = np.std(cls_precision)
        result['recall_std'] = np.std(cls_recall)
        result['f1_std'] = np.std(cls_f1)
        result['coverage'] = 2 * pred_cls_dict / (cls_dict * (cls_dict - 1))

        if verbose:
            print('Class {}: {}'.format(cls, 'p = {:.4f}±{:.4f}, r = {:.4f}±{:.4f}, f1 = {:.4f}±{:.4f}, cvg = {:.4f}' \
                                        .format(result['precision'], result['precision_std'], result['recall'],
                                                result['recall_std'], result['f1'], result['f1_std'], result['coverage']
                                                )))
        return result

    def rm_gt_cache(self, last_epoch=False):
        r"""
        Remove ground truth cache.

        :param last_epoch: Boolean variable, whether this epoch is last epoch; if true, the directory of cache will also be removed.
        """
        if os.path.exists(self.gt_cache_path):
            shutil.rmtree(self.gt_cache_path)
            print('gt perm mat cache deleted')

            if not last_epoch:
                os.mkdir(self.gt_cache_path)
