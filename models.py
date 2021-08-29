# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as K

from layers import FilLayer, AutoIntLayer, CINLayer, CrossNetLayer, CrossNetMixLayer
from layers import FMLayer, FwFMLayer, FFMLayer, FmFMLayer, MultiHead_Polynomial_Block, VectorDense_Layer


class Model(K.Model):

    def __init__(self, args, **kwargs):
        super(Model, self).__init__(args, **kwargs)
        self._sparse_num = args.sparse_field_num
        self._dense_num = args.dense_field_num
        self._sparse_feature_size = args.sparse_feature_size
        self._with_deep = False
        self._feature_dim = args.feature_dim

        self._sparse_embed_layer = K.layers.Embedding(
            input_dim=self._sparse_feature_size,
            output_dim=self._feature_dim,
            embeddings_initializer=K.initializers.glorot_normal(),
            embeddings_regularizer=K.regularizers.l2(args.embedding_l2),
            input_length=self._sparse_num,
            name='Sparse_Embedding')

        if self._dense_num > 0:
            self._dense_embed_layer = K.layers.Embedding(
                input_dim=self._dense_num,
                output_dim=self._feature_dim,
                input_length=self._dense_num,
                name='Dense_Embedding')

        # Construct deep module
        if args.with_deep:
            self._with_deep = True
            self._mlp = K.models.Sequential([
                K.layers.Flatten(),
                K.layers.Dense(400, activation='relu'),
                K.layers.Dense(400, activation='relu'),
                K.layers.Dense(1, activation=None)
            ], name='MLP')

    def build(self, args):
        input_shape = (args.batch_size, args.sparse_field_num +
                       args.dense_field_num,)
        super().build(input_shape)

    def build_embedding(self, inputs):
        sparse_feats = inputs[:, :self._sparse_num]
        sparse_feats = K.backend.cast(sparse_feats, dtype='int32')
        sparse_embeddings = self._sparse_embed_layer(sparse_feats)

        if self._dense_num > 0:
            dense_feats = inputs[:, self._sparse_num:]
            dense_index = K.backend.cumsum(
                K.backend.ones_like(dense_feats, dtype=tf.int32), axis=1) - 1
            dense_embeddings = self._dense_embed_layer(dense_index)
            dense_embeddings = K.layers.multiply(
                [dense_embeddings, K.backend.expand_dims(dense_feats)])
            embeddings = K.backend.concatenate(
                [sparse_embeddings, dense_embeddings], axis=1)
        else:
            embeddings = sparse_embeddings

        return embeddings


class FILMModel(Model):

    def __init__(self, args, **kwargs):
        super(FILMModel, self).__init__(args, **kwargs)

        # Construct `num_layers` sequential fil module
        feature_num = self._sparse_num + self._dense_num
        self._fil = K.models.Sequential(name='FILM')
        for idx in range(args.num_layers):
            self._fil.add(FilLayer(layer_idx=idx,
                                   feature_num=feature_num,
                                   feature_dim=self._feature_dim,
                                   num_heads=args.num_heads,
                                   reduce_head=args.reduce_head,
                                   multi_space=args.multi_space,
                                   residual=args.residual,
                                   adaptive_weight=args.adaptive_weight,
                                   regularizer=args.reguralizer,
                                   num_bases=args.num_bases,
                                   l1_reg=args.weight_l1))
            if idx != args.num_layers - 1 or args.logits_type == "dense":
                self._fil.add(K.layers.Activation(K.activations.relu))
        self._fil.add(K.layers.Flatten())
        if args.logits_type == "dense":
            self._fil.add(K.layers.Dense(1, activation=None))
        elif args.logits_type == "sum":
            self._fil.add(K.layers.Lambda(
                lambda x: K.backend.sum(x, axis=1, keepdims=True)))
        else:
            raise ValueError("Unspported logits type")

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._fil(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class AutoIntModel(Model):

    def __init__(self, args, **kwargs):
        super(AutoIntModel, self).__init__(args, **kwargs)

        self._autoint = K.models.Sequential(name="AutoInt")
        for idx in range(args.num_layers):
            use_relu = (idx != args.num_layers -
                        1) or (args.logits_type == "dense")
            self._autoint.add(AutoIntLayer(
                att_embedding_size=self._feature_dim * 2, use_relu=use_relu))
        self._autoint.add(K.layers.Flatten())
        if args.logits_type == "dense":
            self._autoint.add(K.layers.Dense(1, activation=None))
        elif args.logits_type == "sum":
            self._autoint.add(K.layers.Lambda(
                lambda x: K.backend.sum(x, axis=1, keepdims=True)))
        else:
            raise ValueError("Unspported logits type")

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._autoint(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FILAutoIntModel(AutoIntModel):

    def __init__(self, args, **kwargs):
        super(FILAutoIntModel, self).__init__(args, **kwargs)

        # Construct `num_layers` sequential fil module
        feature_num = self._sparse_num + self._dense_num
        self._fil = K.models.Sequential([
            FilLayer(0, feature_num, self._feature_dim, args.num_heads,
                     args.reduce_head, args.multi_space, args.residual, args.adaptive_weight),
            K.layers.Activation('relu')
        ], name='FIL')

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)
        embeddings = self._fil(embeddings)

        logit = self._autoint(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class DCNv2Model(Model):

    def __init__(self, args, **kwargs):
        super(DCNv2Model, self).__init__(args, **kwargs)

        self._dcnv2 = K.models.Sequential([
            K.layers.Flatten(),
            CrossNetMixLayer(low_rank=32, num_experts=4, layer_num=args.num_layers),
        ], name='DCN-v2')

        if args.logits_type == "dense":
            self._dcnv2.add(K.layers.Dense(1, activation=None))
        elif args.logits_type == "sum":
            self._dcnv2.add(K.layers.Lambda(
                lambda x: K.backend.sum(x, axis=1, keepdims=True)))
        else:
            raise ValueError("Unspported logits type")

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._dcnv2(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class CINModel(Model):

    def __init__(self, args, **kwargs):
        super(CINModel, self).__init__(args, **kwargs)

        self._cin = K.models.Sequential([
            CINLayer(layer_size=(400,)),
            K.layers.Flatten()
        ], name='CIN')

        if args.logits_type == "dense":
            self._cin.add(K.layers.Dense(1, activation=None))
        elif args.logits_type == "sum":
            self._cin.add(K.layers.Lambda(
                lambda x: K.backend.sum(x, axis=1, keepdims=True)))
        else:
            raise ValueError("Unspported logits type")

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._cin(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FILCINModel(Model):

    def __init__(self, args, **kwargs):
        super(FILCINModel, self).__init__(args, **kwargs)

        # Construct `num_layers` sequential fil module
        feature_num = self._sparse_num + self._dense_num
        self._fil = K.models.Sequential([
            FilLayer(0, feature_num, self._feature_dim, args.num_heads,
                     args.reduce_head, args.multi_space, args.residual, args.adaptive_weight),
            K.layers.Activation('relu')
        ], name='FIL')

        self._cin = K.models.Sequential([
            CINLayer(),
            K.layers.Flatten(),
            K.layers.Dense(1, activation=None)
        ], name='CIN')

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)
        embeddings = self._fil(embeddings)

        logit = self._cin(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FILCrossNetModel(Model):

    def __init__(self, args, **kwargs):
        super(FILCrossNetModel, self).__init__(args, **kwargs)

        # Construct `num_layers` sequential fil module
        feature_num = self._sparse_num + self._dense_num
        self._fil = K.models.Sequential([
            FilLayer(0, feature_num, self._feature_dim, args.num_heads,
                     args.reduce_head, args.multi_space, args.residual, args.adaptive_weight),
            K.layers.Activation('relu')
        ], name='FIL')

        self._crossnet = K.models.Sequential([
            K.layers.Flatten(),
            CrossNetLayer(layer_num=3),
            K.layers.Dense(1, activation=None)
        ], name='CrossNet')

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)
        embeddings = self._fil(embeddings)

        logit = self._crossnet(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FMModel(Model):

    def __init__(self, args, **kwargs):
        super(FMModel, self).__init__(args, **kwargs)

        self._fm = K.models.Sequential([
            FMLayer()
        ], name='FM')

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._fm(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FwFMModel(Model):

    def __init__(self, args, **kwargs):
        super(FwFMModel, self).__init__(args, **kwargs)

        self._fwfm = K.models.Sequential([
            FwFMLayer(num_fields=self._sparse_num + self._dense_num)
        ], name="FwFM")

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._fwfm(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FmFMModel(Model):

    def __init__(self, args, **kwargs):
        super(FmFMModel, self).__init__(args, **kwargs)

        self._fmfm = K.models.Sequential([
            FmFMLayer(num_fields=self._sparse_num + self._dense_num,
                      feature_dim=self._feature_dim)
        ], name="FmFM")

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._fmfm(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class FFMModel(K.Model):

    def __init__(self, args, **kwargs):
        super(FFMModel, self).__init__(args, **kwargs)

        self._sparse_num = args.sparse_field_num
        self._dense_num = args.dense_field_num
        self._sparse_feature_size = args.sparse_feature_size
        self._feature_dim = args.feature_dim

        self._num_fields = self._sparse_num + self._dense_num

        self._sparse_embed_layers = []
        self._dense_embed_layers = []

        for field in range(self._num_fields):
            embedding = K.layers.Embedding(
                input_dim=self._sparse_feature_size,
                output_dim=self._feature_dim,
                embeddings_initializer='glorot_normal',
                input_length=self._sparse_num,
                name=f'Sparse_Embedding_{field}')
            self._sparse_embed_layers.append(embedding)

        if self._dense_num > 0:
            for field in range(self._num_fields):
                embedding = K.layers.Embedding(
                    input_dim=self._dense_num,
                    output_dim=self._feature_dim,
                    input_length=self._dense_num,
                    name=f'Dense_Embedding_{field}')
                self._dense_embed_layers.append(embedding)

        # Construct deep module
        if args.with_deep:
            print("FFM model has no deep module")

        self._ffm = K.models.Sequential([
            FFMLayer(num_fields=self._num_fields)
        ], name="FFM")

    def build(self, args):
        input_shape = (args.batch_size, args.sparse_field_num +
                       args.dense_field_num,)
        super().build(input_shape)

    def build_embedding(self, inputs):
        sparse_feats = inputs[:, :self._sparse_num]
        sparse_feats = K.backend.cast(sparse_feats, dtype='int32')
        sparse_embeddings = [self._sparse_embed_layers[i](
            sparse_feats) for i in range(self._num_fields)]

        if self._dense_num > 0:
            dense_feats = inputs[:, self._sparse_num:]
            dense_index = K.backend.cumsum(
                K.backend.ones_like(dense_feats, dtype=tf.int32), axis=1) - 1
            dense_embeddings = [self._dense_embed_layers[i](
                dense_index) for i in range(self._num_fields)]
            dense_embeddings = [K.layers.multiply([embed, K.backend.expand_dims(dense_feats)])
                                for embed in dense_embeddings]
            embeddings = [K.backend.concatenate([sparse_embed, dense_embed], axis=1)
                          for sparse_embed, dense_embed in zip(sparse_embeddings, dense_embeddings)]
        else:
            embeddings = sparse_embeddings

        embeddings = K.backend.stack(embeddings, axis=-1)
        return embeddings

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        logit = self._ffm(embeddings, training)
        outputs = K.backend.sigmoid(logit)
        return outputs


class FILFMModel(Model):

    def __init__(self, args, **kwargs):
        super(FILFMModel, self).__init__(args, **kwargs)

        # Construct `num_layers` sequential fil module
        feature_num = self._sparse_num + self._dense_num
        self._fil = K.models.Sequential([
            FilLayer(0, feature_num, self._feature_dim, args.num_heads,
                     args.reduce_head, args.multi_space, args.residual, args.adaptive_weight),
            K.layers.Activation('relu')
        ], name='FIL')

        self._fm = K.models.Sequential([
            FMLayer()
        ], name='FM')

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)
        embeddings = self._fil(embeddings)

        logit = self._fm(embeddings, training)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs


class LRModel(K.Model):

    def __init__(self, args, **kwargs):
        super(LRModel, self).__init__(args, **kwargs)
        self._lr = K.models.Sequential([
            K.layers.Dense(
                units=1, kernel_initializer=K.initializers.glorot_normal())
        ], name="LR")

    def build(self, args):
        input_shape = (args.batch_size, args.sparse_field_num +
                       args.dense_field_num,)
        super().build(input_shape)

    def call(self, inputs, training):
        logit = self._lr(inputs, training)
        outputs = K.backend.sigmoid(logit)
        return outputs


class xDeepIntModel(Model):

    def __init__(self, args, **kwargs):
        super(xDeepIntModel, self).__init__(args, **kwargs)

        feature_num = self._sparse_num + self._dense_num

        self._poly_block = K.models.Sequential([
            MultiHead_Polynomial_Block(
                hidden_size=args.feature_dim * args.num_heads,
                num_heads=args.num_heads,
                num_interaction_layer=args.num_layers,
                num_sub_spaces=16,
                activation=tf.keras.activations.relu,
                dropout=None,
                residual=True,
                initializer=tf.keras.initializers.glorot_uniform(),
                regularizer=None,
                field_size=feature_num,
                embedding_size=args.feature_dim),
        ], name="MultiHead_Polynomial_Block")

        self._vector_linear_block = VectorDense_Layer(
            units=1,
            activation=tf.keras.activations.linear,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            kernel_regularizer=None,
            dropout=None
        )

    def call(self, inputs, training):
        embeddings = self.build_embedding(inputs)

        output = self._poly_block(embeddings, training)
        vector_logits = self._vector_linear_block(
            inputs=output, training=training)

        logit = tf.reduce_sum(tf.keras.backend.squeeze(
            vector_logits, axis=1), axis=1, keepdims=True)
        if self._with_deep:
            dnn_logit = self._mlp(embeddings, training)
            logit = dnn_logit + logit
        outputs = K.backend.sigmoid(logit)
        return outputs
