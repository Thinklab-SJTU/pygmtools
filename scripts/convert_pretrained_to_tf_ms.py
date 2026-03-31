import argparse
import os
import sys

sys.path.insert(0, '.')

import numpy as np
from appdirs import user_cache_dir

import pygmtools as pygm
import pygmtools.tensorflow_backend as tensorflow_backend
import pygmtools.mindspore_backend as mindspore_backend


def _tf_imports():
    import tensorflow as tf
    return tf


def _ms_imports():
    import mindspore
    return mindspore


def _torch_imports():
    import torch
    return torch


def _default_source_dir():
    return user_cache_dir("pygmtools")


def _find_source_checkpoint(source_dir, model_name, tag):
    filename = f'{model_name}_{tag}_pytorch.pt'
    candidates = [
        os.path.join(source_dir, filename),
        os.path.join('pretrained', filename),
        os.path.join('pygmtools', 'temp', filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'Cannot find source checkpoint {filename} in {candidates}')


def _build_tf_pca(network, in_channel=1024, cross_iter=-1, sk_max_iter=20, sk_tau=0.05):
    tf = _tf_imports()
    n = 4
    feat1 = tf.zeros((1, n, in_channel), dtype=tf.float32)
    feat2 = tf.zeros((1, n, in_channel), dtype=tf.float32)
    A1 = tf.eye(n, batch_shape=[1], dtype=tf.float32)
    A2 = tf.eye(n, batch_shape=[1], dtype=tf.float32)
    n1 = tf.constant([n], dtype=tf.int32)
    n2 = tf.constant([n], dtype=tf.int32)
    network(feat1, feat2, A1, A2, n1, n2, cross_iter, sk_max_iter, sk_tau)


def _build_tf_cie(network, in_node_channel=1024, in_edge_channel=1, sk_max_iter=20, sk_tau=0.05):
    tf = _tf_imports()
    n = 4
    feat_node1 = tf.zeros((1, n, in_node_channel), dtype=tf.float32)
    feat_node2 = tf.zeros((1, n, in_node_channel), dtype=tf.float32)
    A1 = tf.eye(n, batch_shape=[1], dtype=tf.float32)
    A2 = tf.eye(n, batch_shape=[1], dtype=tf.float32)
    feat_edge1 = tf.zeros((1, n, n, in_edge_channel), dtype=tf.float32)
    feat_edge2 = tf.zeros((1, n, n, in_edge_channel), dtype=tf.float32)
    n1 = tf.constant([n], dtype=tf.int32)
    n2 = tf.constant([n], dtype=tf.int32)
    network(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau)


def _build_tf_ngm(network, sk_max_iter=20, sk_tau=0.05):
    tf = _tf_imports()
    n = 4
    K = tf.zeros((1, n * n, n * n), dtype=tf.float32)
    n1 = tf.constant([n], dtype=tf.int32)
    n2 = tf.constant([n], dtype=tf.int32)
    v0 = tf.ones((1, n * n, 1), dtype=tf.float32) / float(n * n)
    network(K, n1, n2, n, n, v0, sk_max_iter, sk_tau)


def _assign_tf_dense(layer, state_dict, prefix):
    weight = state_dict[f'{prefix}.weight'].detach().cpu().numpy().T
    bias = state_dict[f'{prefix}.bias'].detach().cpu().numpy()
    layer.kernel.assign(weight)
    layer.bias.assign(bias)


def _assign_ms_dense(layer, state_dict, prefix):
    mindspore = _ms_imports()
    weight = state_dict[f'{prefix}.weight'].detach().cpu().numpy()
    bias = state_dict[f'{prefix}.bias'].detach().cpu().numpy()
    layer.weight.set_data(mindspore.Tensor(weight, dtype=layer.weight.dtype))
    layer.bias.set_data(mindspore.Tensor(bias, dtype=layer.bias.dtype))


def _assign_tf_affinity(layer, state_dict, prefix):
    layer.A.assign(state_dict[prefix].detach().cpu().numpy())


def _assign_ms_affinity(layer, state_dict, prefix):
    mindspore = _ms_imports()
    value = state_dict[prefix].detach().cpu().numpy()
    layer.A.set_data(mindspore.Tensor(value, dtype=layer.A.dtype))


def _map_pca_like_to_tf(network, state_dict):
    for i, gnn_layer in enumerate(network.gnn_layer_list):
        _assign_tf_dense(gnn_layer.gconv.a_fc, state_dict, f'gnn_layer_{i}.gconv.a_fc')
        _assign_tf_dense(gnn_layer.gconv.u_fc, state_dict, f'gnn_layer_{i}.gconv.u_fc')
        if i in network.cross_graph_list and f'cross_graph_{i}.weight' in state_dict:
            _assign_tf_dense(network.cross_graph_list[i], state_dict, f'cross_graph_{i}')
        if i in network.affinity_list and f'affinity_{i}.A' in state_dict:
            _assign_tf_affinity(network.affinity_list[i], state_dict, f'affinity_{i}.A')


def _map_pca_like_to_ms(network, state_dict):
    for i, gnn_layer in enumerate(network.gnn_layer_list):
        _assign_ms_dense(gnn_layer.gconv.a_fc, state_dict, f'gnn_layer_{i}.gconv.a_fc')
        _assign_ms_dense(gnn_layer.gconv.u_fc, state_dict, f'gnn_layer_{i}.gconv.u_fc')
        cross_graph = network.cross_graph_list[i]
        if hasattr(cross_graph, 'weight') and f'cross_graph_{i}.weight' in state_dict:
            _assign_ms_dense(cross_graph, state_dict, f'cross_graph_{i}')
        affinity = network.affinity_list[i]
        if hasattr(affinity, 'A') and f'affinity_{i}.A' in state_dict:
            _assign_ms_affinity(affinity, state_dict, f'affinity_{i}.A')


def _map_cie_to_tf(network, state_dict):
    for i, gnn_layer in enumerate(network.gnn_layer_list):
        _assign_tf_dense(gnn_layer.gconv.node_fc, state_dict, f'gnn_layer_{i}.gconv.node_fc')
        _assign_tf_dense(gnn_layer.gconv.node_sfc, state_dict, f'gnn_layer_{i}.gconv.node_sfc')
        _assign_tf_dense(gnn_layer.gconv.edge_fc, state_dict, f'gnn_layer_{i}.gconv.edge_fc')
        if i in network.cross_graph_list and f'cross_graph_{i}.weight' in state_dict:
            _assign_tf_dense(network.cross_graph_list[i], state_dict, f'cross_graph_{i}')
        if i in network.affinity_list and f'affinity_{i}.A' in state_dict:
            _assign_tf_affinity(network.affinity_list[i], state_dict, f'affinity_{i}.A')


def _map_cie_to_ms(network, state_dict):
    for i, gnn_layer in enumerate(network.gnn_layer_list):
        _assign_ms_dense(gnn_layer.gconv.node_fc, state_dict, f'gnn_layer_{i}.gconv.node_fc')
        _assign_ms_dense(gnn_layer.gconv.node_sfc, state_dict, f'gnn_layer_{i}.gconv.node_sfc')
        _assign_ms_dense(gnn_layer.gconv.edge_fc, state_dict, f'gnn_layer_{i}.gconv.edge_fc')
        cross_graph = network.cross_graph_list[i]
        if hasattr(cross_graph, 'weight') and f'cross_graph_{i}.weight' in state_dict:
            _assign_ms_dense(cross_graph, state_dict, f'cross_graph_{i}')
        affinity = network.affinity_list[i]
        if hasattr(affinity, 'A') and f'affinity_{i}.A' in state_dict:
            _assign_ms_affinity(affinity, state_dict, f'affinity_{i}.A')


def _map_ngm_to_tf(network, state_dict):
    for i, gnn_layer in enumerate(network.gnn_layer_list):
        _assign_tf_dense(gnn_layer.classifier, state_dict, f'gnn_layer_{i}.classifier')
        _assign_tf_dense(gnn_layer.n_func.layers[0], state_dict, f'gnn_layer_{i}.n_func.0')
        _assign_tf_dense(gnn_layer.n_func.layers[2], state_dict, f'gnn_layer_{i}.n_func.2')
        _assign_tf_dense(gnn_layer.n_self_func.layers[0], state_dict, f'gnn_layer_{i}.n_self_func.0')
        _assign_tf_dense(gnn_layer.n_self_func.layers[2], state_dict, f'gnn_layer_{i}.n_self_func.2')
    _assign_tf_dense(network.classifier, state_dict, 'classifier')


def _map_ngm_to_ms(network, state_dict):
    for i, gnn_layer in enumerate(network.gnn_layer_list):
        _assign_ms_dense(gnn_layer.classifier, state_dict, f'gnn_layer_{i}.classifier')
        _assign_ms_dense(gnn_layer.n_func[0], state_dict, f'gnn_layer_{i}.n_func.0')
        _assign_ms_dense(gnn_layer.n_func[2], state_dict, f'gnn_layer_{i}.n_func.2')
        _assign_ms_dense(gnn_layer.n_self_func[0], state_dict, f'gnn_layer_{i}.n_self_func.0')
        _assign_ms_dense(gnn_layer.n_self_func[2], state_dict, f'gnn_layer_{i}.n_self_func.2')
    _assign_ms_dense(network.classifier, state_dict, 'classifier')


def _solver_specs():
    return {
        'pca_gm': {
            'tags': ['voc', 'willow', 'voc-all'],
            'tf_ctor': lambda: tensorflow_backend.PCA_GM_Net(1024, 2048, 2048, 2),
            'ms_ctor': lambda: mindspore_backend.PCA_GM_Net(1024, 2048, 2048, 2),
            'tf_build': lambda net: _build_tf_pca(net, cross_iter=-1),
            'tf_map': _map_pca_like_to_tf,
            'ms_map': _map_pca_like_to_ms,
            'tf_filename': lambda tag: f'pca_gm_{tag}_tensorflow.weights.h5',
            'ms_filename': lambda tag: f'pca_gm_{tag}_mindspore.ckpt',
        },
        'ipca_gm': {
            'tags': ['voc', 'willow'],
            'tf_ctor': lambda: tensorflow_backend.PCA_GM_Net(1024, 2048, 2048, 2, 3),
            'ms_ctor': lambda: mindspore_backend.PCA_GM_Net(1024, 2048, 2048, 2, 3),
            'tf_build': lambda net: _build_tf_pca(net, cross_iter=3),
            'tf_map': _map_pca_like_to_tf,
            'ms_map': _map_pca_like_to_ms,
            'tf_filename': lambda tag: f'ipca_gm_{tag}_tensorflow.weights.h5',
            'ms_filename': lambda tag: f'ipca_gm_{tag}_mindspore.ckpt',
        },
        'cie': {
            'tags': ['voc', 'willow'],
            'tf_ctor': lambda: tensorflow_backend.CIE_Net(1024, 1, 2048, 2048, 2),
            'ms_ctor': lambda: mindspore_backend.CIE_Net(1024, 1, 2048, 2048, 2),
            'tf_build': _build_tf_cie,
            'tf_map': _map_cie_to_tf,
            'ms_map': _map_cie_to_ms,
            'tf_filename': lambda tag: f'cie_{tag}_tensorflow.weights.h5',
            'ms_filename': lambda tag: f'cie_{tag}_mindspore.ckpt',
        },
        'ngm': {
            'tags': ['voc', 'willow'],
            'tf_ctor': lambda: tensorflow_backend.NGM_Net((16, 16, 16), 1),
            'ms_ctor': lambda: mindspore_backend.NGM_Net((16, 16, 16), 1),
            'tf_build': _build_tf_ngm,
            'tf_map': _map_ngm_to_tf,
            'ms_map': _map_ngm_to_ms,
            'tf_filename': lambda tag: f'ngm_{tag}_tensorflow.weights.h5',
            'ms_filename': lambda tag: f'ngm_{tag}_mindspore.ckpt',
        },
    }


def _convert_one(model_name, tag, backend, source_dir, target_dir):
    torch = _torch_imports()
    spec = _solver_specs()[model_name]
    state_dict = torch.load(_find_source_checkpoint(source_dir, model_name, tag), map_location='cpu')
    os.makedirs(target_dir, exist_ok=True)

    if backend == 'tensorflow':
        network = spec['tf_ctor']()
        spec['tf_build'](network)
        spec['tf_map'](network, state_dict)
        target_path = os.path.join(target_dir, spec['tf_filename'](tag))
        tensorflow_backend._save_model(network, target_path)
    elif backend == 'mindspore':
        network = spec['ms_ctor']()
        spec['ms_map'](network, state_dict)
        target_path = os.path.join(target_dir, spec['ms_filename'](tag))
        mindspore_backend._save_model(network, target_path)
    else:
        raise ValueError(f'Unknown backend {backend}')

    print(f'Converted {model_name}:{tag} -> {target_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert canonical PyTorch pygmtools pretrained weights to TensorFlow and MindSpore formats.')
    parser.add_argument('--backend', choices=['tensorflow', 'mindspore', 'all'], default='all')
    parser.add_argument('--models', nargs='*', choices=['pca_gm', 'ipca_gm', 'cie', 'ngm'], default=['pca_gm', 'ipca_gm', 'cie', 'ngm'])
    parser.add_argument('--source-dir', default=_default_source_dir())
    parser.add_argument('--target-dir', default=_default_source_dir())
    args = parser.parse_args()

    backends = ['tensorflow', 'mindspore'] if args.backend == 'all' else [args.backend]
    specs = _solver_specs()
    for model_name in args.models:
        for tag in specs[model_name]['tags']:
            for backend in backends:
                _convert_one(model_name, tag, backend, args.source_dir, args.target_dir)


if __name__ == '__main__':
    main()
