# -*- coding:utf-8 -*-
"""
Implementations of our field-aware interaction layer and baseline layers,
including `FilLayer (our), AutoIntLayer, CIN layer, CrossNet and FM layer.
Reference code: https://github.com/shenweichen/DeepCTR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.keras.backend import zeros


class FilLayer(K.layers.Layer):
    """
    Filed-aware interaction learning layer in FILM model.

    Attributes:
        layer_idx (int): Index of this layer, used for debugging.
        feature_num (int): Num of feature fields.
        feature_dim (int): Dimension of feature representation.
        num_heads (int): Num of channels in each semantic sub-space.
        multi_space (bool): If use multiple semantic spaces.
        residual: (bool): If use residual connection in this layer.
        adaptive_weight (bool): If use adaptive weights for feature interactions.
    """

    def __init__(self,
                 layer_idx,
                 feature_num,
                 feature_dim,
                 num_heads=2,
                 reduce_head='mean',
                 multi_space=True,
                 residual=True,
                 adaptive_weight=True,
                 regularizer="none",
                 num_bases=-1,
                 l1_reg=0.0,
                 **kwargs):
        super(FilLayer, self).__init__(**kwargs)
        self._layer_idx = layer_idx
        self._feature_num = feature_num
        self._feature_dim = feature_dim
        self._num_heads = num_heads
        self._reduce_head = reduce_head
        self._residual = residual
        self._multi_space = multi_space
        self._adaptive_weight = adaptive_weight
        self._regularizer = regularizer
        self._num_bases = num_bases
        self._l1_reg = l1_reg

    def build(self, input_shape):
        """
        Create weight parameters used in this layer.

        Args:
            input_shape (Tuple): Shape of the input of the `call` function.
        """
        kernel_num_in_one_head = self._feature_num if self._multi_space else 1
        sum_feature_dim = self._feature_num * self._feature_dim

        if self._regularizer == "none":
            self.fil_kernels = self.add_weight(
                name=f"fil_kernels_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head,
                       sum_feature_dim, self._feature_dim],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l1(self._l1_reg),
                trainable=True)

        elif self._regularizer == "quat":
            if sum_feature_dim % self._num_bases != 0 or self._feature_dim % self._num_bases != 0:
                raise ValueError(
                    "Feature size must be a multiplier of num_bases")
            self.field_A_matrices, self.field_S_matrices = [], []
            for field in range(kernel_num_in_one_head):
                A_matrices, S_matrices = [], []
                for space in range(self._num_bases):
                    A_matrix = self.add_weight(
                        name=f"fil_a_matrix_{field}_{space}_{self._layer_idx}",
                        shape=[self._num_bases, self._num_bases],
                        dtype=tf.float32,
                        initializer=K.initializers.glorot_normal(),
                        trainable=True
                    )
                    S_matrix = self.add_weight(
                        name=f"fil_s_matrix_{field}_{space}_{self._layer_idx}",
                        shape=[sum_feature_dim // self._num_bases, self._feature_dim // self._num_bases],
                        dtype=tf.float32,
                        initializer=K.initializers.glorot_normal(),
                        trainable=True
                    )
                    A_matrices.append(A_matrix)
                    S_matrices.append(S_matrix)

                self.field_A_matrices.append(A_matrices)
                self.field_S_matrices.append(S_matrices)

        elif self._regularizer == "rank":
            self.fil_kernels_u = self.add_weight(
                name=f"fil_kernels_u_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head,
                       sum_feature_dim, self._num_bases],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l1(self._l1_reg),
                trainable=True
            )

            self.fil_kernel_t = self.add_weight(
                name=f"fil_kernel_t_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head,
                       self._num_bases, self._num_bases],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l1(self._l1_reg),
                trainable=True
            )

            self.fil_kernels_v = self.add_weight(
                name=f"fil_kernels_v_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head,
                       self._num_bases, self._feature_dim],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l1(self._l1_reg),
                trainable=True
            )

        elif self._regularizer == "bdd":
            if sum_feature_dim % self._num_bases != 0 or self._feature_dim % self._num_bases != 0:
                raise ValueError(
                    "Feature size must be a multiplier of num_bases")
            self.submat_in = sum_feature_dim // self._num_bases
            self.submat_out = self._feature_dim // self._num_bases
            self.fil_kernels = self.add_weight(
                name=f"fil_kernels_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head,
                       self._num_bases, self.submat_in, self.submat_out],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l1(self._l1_reg),
                trainable=True
            )

        elif self._regularizer == "basis":
            if self._num_bases > kernel_num_in_one_head:
                raise ValueError("num_bases must be smaller than kernel num")
            self.fil_kernels = self.add_weight(
                name=f"fil_kernel_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head, self._num_bases, self._feature_dim, self._feature_dim],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l1(self._l1_reg),
                trainable=True
            )
            self.w_comp = self.add_weight(
                name=f"fil_kernel_comp_{self._layer_idx}",
                shape=[self._num_heads, kernel_num_in_one_head, self._feature_num, self._num_bases],
                dtype=tf.float32,
                initializer=K.initializers.glorot_normal(),
                regularizer=K.regularizers.l2(self._l1_reg),
                trainable=True
            )

        else:
            raise NotImplementedError(
                f"Not supported regularizer type {self._regularizer}")

        self.fil_weights = self.add_weight(
            name=f"fil_weights_{self._layer_idx}",
            shape=[self._num_heads, self._feature_num, self._feature_num],
            dtype=tf.float32,
            initializer=K.initializers.Ones(),
            trainable=self._adaptive_weight)

        super(FilLayer, self).build(input_shape)

    def get_projected_feature(self, weighted_vector):
        """weighted_vector with shape of (h, f, b, f*d)"""
        kernel_num_in_one_head = self._feature_num if self._multi_space else 1
        sum_feature_dim = self._feature_num * self._feature_dim

        if self._regularizer == "none":
            tiled_kernel = K.backend.repeat_elements(
                self.fil_kernels,
                rep=1 if self._multi_space else self._feature_num,
                axis=1)  # (h, f, f*d, d)
            projection = tf.matmul(weighted_vector, tiled_kernel)  # (h, f, b, d)
            projection = tf.nn.leaky_relu(projection)

        elif self._regularizer == "quat":
            field_matrices = []
            for field in range(kernel_num_in_one_head):
                A_matrices, S_matrices = self.field_A_matrices[field], self.field_S_matrices[field]
                kronecker_products = []
                for space in range(self._num_bases):
                    A_matrix, S_matrix = A_matrices[space], S_matrices[space]
                    operator_1 = tf.linalg.LinearOperatorFullMatrix(A_matrix)
                    operator_2 = tf.linalg.LinearOperatorFullMatrix(S_matrix)
                    operator = tf.linalg.LinearOperatorKronecker([operator_1, operator_2])
                    kronecker_product = operator.to_dense()
                    kronecker_products.append(kronecker_product)
                field_matrices.append(tf.add_n(kronecker_products))
            kernel = K.backend.stack(field_matrices, axis=0)
            tiled_kernel = K.backend.repeat_elements(
                kernel,
                rep=1 if self._multi_space else self._feature_num,
                axis=0
            )
            tiled_kernel = K.backend.repeat_elements(
                tiled_kernel,
                rep=self._num_heads,
                axis=0
            )
            projection = tf.matmul(weighted_vector, tiled_kernel)  # (h, f, b, d)
            projection = tf.nn.leaky_relu(projection)

        elif self._regularizer == "rank":
            tiled_kernel_u = K.backend.repeat_elements(
                self.fil_kernels_u,
                rep=1 if self._multi_space else self._feature_num,
                axis=1)  # (h, f f*d, n)
            tiled_kernel_t = K.backend.repeat_elements(
                self.fil_kernel_t,
                rep=1 if self._multi_space else self._feature_num,
                axis=1)  # (h, f, n, n)
            tiled_kernel_v = K.backend.repeat_elements(
                self.fil_kernels_v,
                rep=1 if self._multi_space else self._feature_num,
                axis=1)  # (h, f, n, d)

            projection = tf.matmul(weighted_vector, tiled_kernel_u)  # (h, f, b, n)
            projection = tf.nn.leaky_relu(projection)
            projection = tf.matmul(projection, tiled_kernel_t)  # (h, f, b, n)
            projection = tf.nn.leaky_relu(projection)
            projection = tf.matmul(projection, tiled_kernel_v)  # (h, f, b, d)

        elif self._regularizer == "bdd":
            weighted_vector = tf.reshape(weighted_vector, shape=[
                                         self._num_heads, self._feature_num, -1, self._num_bases, self.submat_in])
            tiled_kernel = K.backend.repeat_elements(
                self.fil_kernels,
                rep=1 if self._multi_space else self._feature_num,
                axis=1)  # (h, f, n, submat_in, submat_out)
            projection = tf.einsum("abcde,abdef->abcdf", weighted_vector, tiled_kernel)  # (h, f, b, n, o)
            projection = tf.nn.leaky_relu(projection)
            projection = tf.reshape(projection, shape=[self._num_heads, self._feature_num, -1, self._feature_dim])

        elif self._regularizer == "basis":
            fil_kernels = tf.reshape(self.fil_kernels, shape=[self._num_heads, kernel_num_in_one_head, self._num_bases, -1])  # (h, f, n, d*d)
            fil_kernels = tf.matmul(self.w_comp, fil_kernels)  # (h, f, f, d*d)
            fil_kernels = tf.nn.leaky_relu(fil_kernels)
            fil_kernels = tf.reshape(
                fil_kernels,
                shape=[self._num_heads, kernel_num_in_one_head, sum_feature_dim, self._feature_dim])  # (h, f, f*d, d)
            tiled_kernel = K.backend.repeat_elements(
                fil_kernels,
                rep=1 if self._multi_space else self._feature_num,
                axis=1)  # (h, f, f*d, d)
            projection = tf.matmul(weighted_vector, tiled_kernel)  # (h, f, b, d)
            projection = tf.nn.leaky_relu(projection)

        return projection

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._feature_num, self._feature_dim)

    def call(self, inputs):
        """
        Invoke this field-aware feature interaction learning layer.

        Args:
            inputs (tf.Tensor): Input tensor with the shape of (batch_size, feature_num, feature_dim).

        Raises:
            NotImplementedError: Raised if the type of `reduce_head` is not supported.

        Returns:
            tf.Tensor: The interacted combinatorial feature with the same shape of `inputs`. 
        """

        transpose_inputs = K.backend.permute_dimensions(
            inputs, pattern=(1, 0, 2))  # (f, b, d)
        transpose_inputs = K.backend.reshape(
            transpose_inputs, shape=(self._feature_num, -1))  # (f, b*d)
        transpose_inputs = K.backend.expand_dims(
            K.backend.expand_dims(transpose_inputs, axis=0),
            axis=0)  # (1, 1, f, b*d)

        weights = K.backend.expand_dims(
            self.fil_weights, axis=-1)  # (h, f, f, 1)
        weighted_inputs = tf.multiply(
            weights, transpose_inputs)  # (h, f, f, b*d)
        weighted_vector = K.backend.reshape(
            weighted_inputs,
            shape=(self._num_heads, self._feature_num,
                   self._feature_num, -1, self._feature_dim))  # (h, f, f, b, d)
        weighted_vector = K.backend.permute_dimensions(
            weighted_vector, pattern=(0, 1, 3, 2, 4))  # (h, f, b, f, d)
        weighted_vector = K.backend.reshape(
            weighted_vector,
            shape=(self._num_heads, self._feature_num, -1,
                   self._feature_num * self._feature_dim))  # (h, f, b, f*d)

        projection = self.get_projected_feature(weighted_vector)  # (h, f, b, d)

        projection = K.backend.permute_dimensions(
            projection, pattern=(2, 0, 1, 3))  # (b, h, f, d)

        fil_out = K.backend.expand_dims(
            inputs, axis=1)  # (b, 1, f, d)
        fil_out = tf.multiply(projection, fil_out)  # (b, h, f, d)

        # To avoid to many dimension
        if self._reduce_head == 'sum':
            fil_out = K.backend.sum(fil_out, axis=1)  # (b, f, d)
        elif self._reduce_head == 'mean':
            fil_out = K.backend.mean(fil_out, axis=1)  # (b, f, d)
        else:
            raise NotImplementedError('Invalid reduction head type')

        if self._residual:
            fil_out = fil_out + inputs
        return fil_out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'feature_num': self._feature_num,
            'feature_dim': self._feature_dim,
            'num_heads': self._num_heads,
            'reduce_head': self._reduce_head,
            'residual': self._residual,
            'adaptive_weight': self._adaptive_weight,
        })
        return config


class AutoIntLayer(K.layers.Layer):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,att_embedding_size * head_num)``.
      Arguments
            - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
            - **head_num**: int.The head number in multi-head  self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, att_embedding_size=16, head_num=2, use_res=True, use_relu=True, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res

        self.Q_dense = K.layers.Dense(
            units=att_embedding_size * head_num, activation='relu')
        self.K_dense = K.layers.Dense(
            units=att_embedding_size * head_num, activation='relu')
        self.V_dense = K.layers.Dense(
            units=att_embedding_size * head_num, activation='relu')
        self.R_dense = K.layers.Dense(
            units=att_embedding_size * head_num, activation='relu')

        super(AutoIntLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(AutoIntLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.backend.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.backend.ndim(inputs)))

        querys = self.Q_dense(inputs)
        keys = self.K_dense(inputs)
        values = self.V_dense(inputs)

        # head_num None F D
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(
            querys, keys, transpose_b=True)  # head_num None F F
        self.normalized_att_scores = K.backend.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores,
                           values)  # head_num None F D
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0)  # None F D*head_num

        if self.use_res:
            result += self.R_dense(inputs)
        result = tf.nn.relu(result)

        return result

    def compute_output_shape(self, input_shape):

        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num, 'use_res': self.use_res,
                  'seed': self.seed}
        base_config = super(AutoIntLayer, self).get_config()
        base_config.update(config)
        return base_config


class CINLayer(K.layers.Layer):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size=(100, 100, 50), activation='relu', split_half=True, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.split_half = split_half
        self.activation = activation
        super(CINLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [int(input_shape[1])]
        self.filters = []
        self.bias = []
        for i, size in enumerate(self.layer_size):

            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1]
                                                       * self.field_nums[0], size],
                                                dtype=tf.float32, initializer=K.initializers.glorot_uniform()))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.activation_layers = [K.layers.Activation(
            self.activation) for _ in self.layer_size]

        # Be sure to call this somewhere!
        super(CINLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.backend.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.backend.ndim(inputs)))

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)

            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])

            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')

            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = K.backend.sum(result, axis=-1, keepdims=False)

        return result

    def compute_output_shape(self, input_shape):
        if self.split_half:
            featuremap_num = sum(
                self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            featuremap_num = sum(self.layer_size)
        return (None, featuremap_num)

    def get_config(self, ):

        config = {'layer_size': self.layer_size, 'split_half': self.split_half, 'activation': self.activation,
                  'seed': self.seed}
        base_config = super(CINLayer, self).get_config()
        base_config.update(config)
        return base_config


class CrossNetLayer(K.layers.Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, parameterization='vector', **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNetLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer=K.initializers.glorot_normal(),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.parameterization == 'matrix':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, dim),
                                            initializer=K.initializers.glorot_normal(),
                                            trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=K.initializers.Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNetLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.backend.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.backend.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.parameterization == 'matrix':
                # W * xi  (bs, dim, 1)
                dot_ = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                print("parameterization should be 'vector' or 'matrix'")
            x_l = dot_ + x_l
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'layer_num': self.layer_num, 'parameterization': self.parameterization,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNetLayer, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class CrossNetMixLayer(K.layers.Layer):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, low_rank=32, num_experts=4, layer_num=2, **kwargs):
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        super(CrossNetMixLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])

        # U: (dim, low_rank)
        self.U_list = [self.add_weight(name='U_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=K.initializers.glorot_normal(),
                                       trainable=True) for i in range(self.layer_num)]
        # V: (dim, low_rank)
        self.V_list = [self.add_weight(name='V_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=K.initializers.glorot_normal(),
                                       trainable=True) for i in range(self.layer_num)]
        # C: (low_rank, low_rank)
        self.C_list = [self.add_weight(name='C_list' + str(i),
                                       shape=(self.num_experts, self.low_rank, self.low_rank),
                                       initializer=K.initializers.glorot_normal(),
                                       trainable=True) for i in range(self.layer_num)]

        self.gating = [tf.keras.layers.Dense(1, use_bias=False) for i in range(self.num_experts)]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=K.initializers.zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNetMixLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.backend.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](tf.squeeze(x_l, axis=2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][expert_id]), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = tf.nn.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.C_list[i][expert_id], v_x)  # (bs, low_rank, 1)
                v_x = tf.nn.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][expert_id], v_x)  # (bs, dim, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(tf.squeeze(dot_, axis=2))

            # (3) mixture of low-rank experts
            output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
            gating_score_of_experts = tf.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score_of_experts, 1))
            x_l = moe_out + x_l  # (bs, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'low_rank': self.low_rank, 'num_experts': self.num_experts, 'layer_num': self.layer_num,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNetMixLayer, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class FMLayer(K.layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        # Be sure to call this somewhere!
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.backend.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.backend.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


class FwFMLayer(K.layers.Layer):
    """Field-weighted Factorization Machines
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      Arguments
        - **num_fields** : integer for number of fields
        - **regularizer** : L2 regularizer weight for the field strength parameters of FwFM
      References
        - [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising]
        https://arxiv.org/pdf/1806.03514.pdf
    """

    def __init__(self, num_fields=4, **kwargs):
        self.num_fields = num_fields
        super(FwFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        if input_shape[1] != self.num_fields:
            raise ValueError("Mismatch in number of fields {} and \
                 concatenated embeddings dims {}".format(self.num_fields, input_shape[1]))

        self.field_strengths = self.add_weight(name='field_pair_strengths',
                                               shape=(self.num_fields, self.num_fields),
                                               initializer=K.initializers.TruncatedNormal(),
                                               trainable=True)

        super(FwFMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.backend.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.backend.ndim(inputs)))

        if inputs.shape[1] != self.num_fields:
            raise ValueError("Mismatch in number of fields {} and \
                 concatenated embeddings dims {}".format(self.num_fields, inputs.shape[1]))

        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            # get field strength for pair fi and fj
            r_ij = self.field_strengths[fi, fj]

            # get embeddings for the features of both the fields
            feat_embed_i = tf.squeeze(inputs[0:, fi:fi + 1, 0:], axis=1)
            feat_embed_j = tf.squeeze(inputs[0:, fj:fj + 1, 0:], axis=1)

            f = tf.scalar_mul(r_ij, K.backend.batch_dot(feat_embed_i, feat_embed_j, axes=1))
            pairwise_inner_prods.append(f)

        sum_ = tf.add_n(pairwise_inner_prods)
        return sum_

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = super(FwFMLayer, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields
        })
        return config


class FFMLayer(K.layers.Layer):
    """Field-aware Factorization Machines for CTR Prediction.
    """

    def __init__(self, num_fields, **kwargs):
        self.num_fields = num_fields
        super(FFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            v_i = inputs[:, fi, :, fj]
            v_j = inputs[:, fj, :, fi]

            f = tf.reduce_sum(tf.multiply(v_i, v_j), axis=1)
            pairwise_inner_prods.append(f)

        sum_ = tf.add_n(pairwise_inner_prods)
        return K.backend.reshape(sum_, shape=(-1, 1))

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = super(FFMLayer, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields
        })
        return config


class FmFMLayer(K.layers.Layer):

    def __init__(self, num_fields, feature_dim, **kwargs):
        self.num_fields = num_fields
        self.feature_dim = feature_dim
        super(FmFMLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.field_matrices = self.add_weight(
            name='field_matrices',
            shape=(self.num_fields, self.num_fields, self.feature_dim, self.feature_dim),
            initializer=K.initializers.TruncatedNormal(),
            trainable=True)

        super(FmFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        pairwise_inner_prods = []
        for fi, fj in itertools.combinations(range(self.num_fields), 2):
            v_i = inputs[:, fi, :]
            v_j = inputs[:, fj, :]

            v_i = tf.matmul(v_i, self.field_matrices[fi, fj])

            f = tf.reduce_sum(tf.multiply(v_i, v_j), axis=1)
            pairwise_inner_prods.append(f)

        sum_ = tf.add_n(pairwise_inner_prods)
        return K.backend.reshape(sum_, shape=(-1, 1))

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self):
        config = super(FwFMLayer, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields,
            'feature_dim': self.feature_dim
        })
        return config


class VectorDense_Layer(tf.keras.layers.Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        kernel_regularizer=None,
        dropout=None
    ):
        super(VectorDense_Layer, self).__init__()
        self.units = units
        self.dropout = dropout

        self.permute_layer = tf.keras.layers.Permute(
            dims=(2, 1)
        )

        if self.dropout is not None and self.dropout > 0:
            self.dropout_layer = tf.keras.layers.Dropout(
                rate=float(self.dropout)
            )

        self.dense_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs, training):
        net = self.permute_layer(inputs)
        if self.dropout is not None and self.dropout > 0:
            net = self.dropout_layer(net, training=training)
        net = self.dense_layer(net)
        outputs = self.permute_layer(net)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        output_shape = copy.copy(input_shape)
        output_shape[1] = self.units
        return tf.TensorShape(output_shape)


class Polynomial_Block(tf.keras.layers.Layer):

    def __init__(
        self,
        num_interaction_layer,
        num_sub_spaces,
        activation,
        dropout,
        residual,
        initializer,
        regularizer,
        field_size,
        embedding_size,
    ):
        super(Polynomial_Block, self).__init__()
        self.num_interaction_layer = num_interaction_layer
        self.num_sub_spaces = num_sub_spaces
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer
        self.field_size = field_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        field_size = input_shape[1]

        self.vector_dense_layers = [
            VectorDense_Layer(
                units=int(field_size * self.num_sub_spaces),
                activation=self.activation,
                use_bias=False,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                dropout=self.dropout
            )
            for i in range(self.num_interaction_layer)
        ]

    def call(
        self,
        inputs,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, self.field_size, self.embedding_size)
        )

        # Split
        inputs = tf.concat(
            tf.split(inputs, self.num_sub_spaces, axis=2),
            axis=1
        )

        # Interaction
        interaction = inputs
        if not self.residual:
            interaction_list = []
            interaction_list.append(interaction)

        for layer_id in range(0, self.num_interaction_layer, +1):

            weighted_inputs = self.vector_dense_layers[layer_id](
                inputs,
                training=training
            )

            if self.residual:
                interaction = tf.keras.layers.multiply(
                    [interaction, (1.0 + weighted_inputs)]
                )
            else:
                interaction = tf.keras.layers.multiply(
                    [interaction, weighted_inputs]
                )
                interaction_list.append(interaction)

        # Output
        if self.residual:
            interaction_outputs = interaction
        else:
            interaction_outputs = tf.keras.backend.concatenate(
                interaction_list, axis=1
            )

        # Combine
        interaction_outputs = tf.concat(
            tf.split(interaction_outputs, self.num_sub_spaces, axis=1),
            axis=2
        )

        return interaction_outputs


class MultiHead_Polynomial_Block(tf.keras.layers.Layer):

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_interaction_layer,
        num_sub_spaces,
        activation,
        dropout,
        residual,
        initializer,
        regularizer,
        field_size,
        embedding_size
    ):
        super(MultiHead_Polynomial_Block, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_interaction_layer = num_interaction_layer
        self.num_sub_spaces = num_sub_spaces
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.initializer = initializer
        self.regularizer = regularizer
        self.field_size = field_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.projection_layer = tf.keras.layers.Dense(
            units=self.hidden_size,
            activation=tf.keras.activations.linear,
            use_bias=False,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer
        )
        self.polynomial_block_list = [
            Polynomial_Block(
                num_interaction_layer=self.num_interaction_layer,
                num_sub_spaces=self.num_sub_spaces,
                activation=self.activation,
                dropout=self.dropout,
                residual=self.residual,
                initializer=self.initializer,
                regularizer=self.regularizer,
                field_size=self.field_size,
                embedding_size=int(self.hidden_size / self.num_heads)
            )
            for i in range(self.num_heads)
        ]

    def call(
        self,
        inputs,
        training
    ):
        # Input
        inputs = tf.keras.backend.reshape(
            inputs,
            shape=(-1, self.field_size, self.embedding_size)
        )

        # Linear Projection
        inputs = self.projection_layer(inputs)

        # Split
        inputs_heads = tf.split(inputs, self.num_heads, axis=2)

        # Polynomial Interaction
        outputs_heads = [
            self.polynomial_block_list[i](
                inputs_heads[i],
                training=training
            )
            for i in range(self.num_heads)
        ]

        # Combine
        outputs = tf.concat(outputs_heads, axis=1)

        # Return
        return outputs

