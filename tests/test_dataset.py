import pygmtools as pygm


# Test dataset download and preprocess, and data fetch and evaluation
def _test_benchmark(name, sets, problem, filter):
    benchmark = pygm.benchmark.Benchmark(name=name, sets=sets, problem=problem, filter=filter)
    if sets == 'test':
        num = 2 if benchmark.problem == '2GM' else 3
        _test_get_data(benchmark, num)


# Test data fetch and evaluation
def _test_get_data(benchmark, num):
    rand_data = benchmark.rand_get_data(num=num)
    assert rand_data is not None
    data_list, perm_dict, ids = benchmark.rand_get_data(cls=benchmark.classes[0], num=num)

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
        result = benchmark.eval(prediction=pred, classes=[benchmark.classes[0]], verbose=True)
        assert result['mean']['f1'] == 1, f'Accuracy should be 1, something wrong in {benchmark.name} dataset test.'
        benchmark.eval_cls(prediction=pred, cls=benchmark.classes[0], verbose=True)
        assert result['f1'] == 1, f'Accuracy should be 1, something wrong in {benchmark.name} dataset test.'


# Entry function
def test_dataset_and_benchmark():
    dataset_name_list = ['PascalVOC', 'WillowObject', 'SPair71k', 'CUB2011', 'IMC_PT_SparseGM']
    problem_type_list = ['2GM', 'MGM']
    set_list = ['train', 'test']
    filter_list = ['intersection', 'inclusion', 'unfiltered']
    for dataset_name in dataset_name_list:
        for set in set_list:
            for problem_type in problem_type_list:
                for filter in filter_list:
                    if dataset_name == 'SPair71k' and problem_type == 'MGM':
                        continue
                    if filter == 'inclusion' and problem_type == 'MGM':
                        continue
                    _test_benchmark(dataset_name, set, problem_type, filter)


if __name__ == '__main__':
    test_dataset_and_benchmark()
