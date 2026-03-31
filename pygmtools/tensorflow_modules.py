# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import tensorflow as tf


def _l1_normalize(input_tensor, axis=-1, eps=1e-12):
    denom = tf.reduce_sum(tf.abs(input_tensor), axis=axis, keepdims=True)
    return input_tensor / tf.maximum(denom, eps)


class WeightedInnerProdAffinity(tf.keras.layers.Layer):
    """
    Weighted inner product affinity layer to compute the affinity matrix from feature space.
    """

    def __init__(self, d):
        super().__init__()
        self.d = d

    def build(self, input_shape):
        stdv = 1.0 / (self.d ** 0.5)
        init = tf.random.uniform((self.d, self.d), minval=-stdv, maxval=stdv, dtype=self.dtype or tf.float32)
        init = init + tf.eye(self.d, dtype=init.dtype)
        self.A = self.add_weight(
            name='A',
            shape=(self.d, self.d),
            initializer=tf.constant_initializer(init.numpy()),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, X, Y):
        M = tf.matmul(X, self.A)
        M = tf.matmul(M, tf.transpose(Y, perm=[0, 2, 1]))
        return M


class Gconv(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = tf.keras.layers.Dense(self.num_outputs)
        self.u_fc = tf.keras.layers.Dense(self.num_outputs)

    def call(self, A, x, norm=True):
        if norm:
            A = _l1_normalize(A, axis=-2)
        ax = self.a_fc(x)
        ux = self.u_fc(x)
        return tf.matmul(A, tf.nn.relu(ax)) + tf.nn.relu(ux)


class ChannelIndependentConv(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        super().__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        self.node_fc = tf.keras.layers.Dense(out_features)
        self.node_sfc = tf.keras.layers.Dense(out_features)
        self.edge_fc = tf.keras.layers.Dense(self.out_edges)

    def call(self, A, emb_node, emb_edge, mode=1):
        if mode != 1:
            raise ValueError(f'Unknown mode {mode}. Possible options: 1 or 2')

        node_x = self.node_fc(emb_node)
        node_sx = self.node_sfc(emb_node)
        edge_x = self.edge_fc(emb_edge)

        A = tf.expand_dims(A, axis=-1)
        A = tf.multiply(tf.broadcast_to(A, tf.shape(edge_x)), edge_x)

        node_x = tf.matmul(
            tf.transpose(A, perm=[0, 3, 1, 2]),
            tf.transpose(tf.expand_dims(node_x, axis=2), perm=[0, 3, 1, 2]),
        )
        node_x = tf.transpose(tf.squeeze(node_x, axis=-1), perm=[0, 2, 1])
        node_x = tf.nn.relu(node_x) + tf.nn.relu(node_sx)
        edge_x = tf.nn.relu(edge_x)

        return node_x, edge_x


class Siamese_Gconv(tf.keras.layers.Layer):
    def __init__(self, in_features, num_features):
        super().__init__()
        self.gconv = Gconv(in_features, num_features)

    def call(self, g1, *args):
        emb1 = self.gconv(*g1)
        if len(args) == 0:
            return emb1
        returns = [emb1]
        for g in args:
            returns.append(self.gconv(*g))
        return returns


class Siamese_ChannelIndependentConv(tf.keras.layers.Layer):
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super().__init__()
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def call(self, g1, *args):
        emb1, emb_edge1 = self.gconv(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv(*g)
            embs.append(emb2)
            emb_edges.append(emb_edge2)
        return embs + emb_edges


class NGMConvLayer(tf.keras.layers.Layer):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, sk_channel=0):
        super().__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.classifier = tf.keras.layers.Dense(self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.classifier = None

        self.n_func = tf.keras.Sequential([
            tf.keras.layers.Dense(self.out_nfeat),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.out_nfeat),
            tf.keras.layers.ReLU(),
        ])
        self.n_self_func = tf.keras.Sequential([
            tf.keras.layers.Dense(self.out_nfeat),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.out_nfeat),
            tf.keras.layers.ReLU(),
        ])

    def call(self, A, W, x, n1=None, n2=None, norm=True, sk_func=None):
        W_new = W

        if norm:
            A = _l1_normalize(A, axis=2)

        x1 = self.n_func(x)
        x2 = tf.matmul(
            tf.transpose(tf.expand_dims(A, axis=-1) * W_new, perm=[0, 3, 1, 2]),
            tf.transpose(tf.expand_dims(x1, axis=2), perm=[0, 3, 1, 2]),
        )
        x2 = tf.transpose(tf.squeeze(x2, axis=-1), perm=[0, 2, 1])
        x2 = x2 + self.n_self_func(x)

        if self.classifier is not None:
            assert sk_func is not None
            x3 = self.classifier(x2)
            n1max = tf.cast(tf.reduce_max(n1), tf.int32)
            n2max = tf.cast(tf.reduce_max(n2), tf.int32)
            n1_rep = tf.repeat(tf.cast(n1, tf.int32), repeats=self.sk_channel)
            n2_rep = tf.repeat(tf.cast(n2, tf.int32), repeats=self.sk_channel)
            x4 = tf.reshape(
                tf.transpose(x3, perm=[0, 2, 1]),
                [tf.shape(x)[0] * self.sk_channel, n2max, n1max],
            )
            x4 = tf.transpose(x4, perm=[0, 2, 1])
            x5 = tf.transpose(sk_func(x4, n1_rep, n2_rep, dummy_row=True), perm=[0, 2, 1])
            x6 = tf.reshape(x5, [tf.shape(x)[0], self.sk_channel, n1max * n2max])
            x6 = tf.transpose(x6, perm=[0, 2, 1])
            x_new = tf.concat((x2, x6), axis=-1)
        else:
            x_new = x2

        return W_new, x_new
