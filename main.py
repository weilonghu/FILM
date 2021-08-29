#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import random
import os

import tensorflow as tf

from data_utils import make_parse_example_fn
from models import DCNv2Model as Model


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str,
                        default='data/movielens-1m', help='Base directory of datasets')
    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords', help='Train tfrecord file')
    parser.add_argument('--valid_file', type=str,
                        default='valid.tfrecords', help='Validation tfrecord file')
    parser.add_argument('--test_file', type=str,
                        default='test.tfrecords', help='test tfrecord file')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--lr_scale', type=float, default=1.0,
                        help='Scaling factor of learning rate')
    parser.add_argument('--min_lr', type=float,
                        default=5e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=2048,
                        help='Random seed used in numpy and TensorFlow')
    parser.add_argument('--epochs', type=int,
                        default=20, help='Epochs for training')
    parser.add_argument('--cache_in_memory', action='store_true',
                        help='Whether cache all data in memory')
    parser.add_argument('--shuffle_buffer_size', type=int,
                        default=591360, help='Shuffle buffer size')
    parser.add_argument('--prefetch_buffer_size', type=int,
                        default=591360, help='Prefetch buffer size')
    parser.add_argument('--num_parallel_calls', type=int, default=4,
                        help='The number elements to process asynchronously in parallel')
    parser.add_argument('--sparse_field_num', type=int, default=11,
                        help='The number of sparse features')
    parser.add_argument('--sparse_feature_size', type=int, default=3569,
                        help='The number of sparse feature keys')
    parser.add_argument('--dense_field_num', type=int, default=1,
                        help='The number of dense features')
    parser.add_argument('--feature_dim', type=int, default=16,
                        help='The number of feature dimension')
    parser.add_argument('--model_path', type=str, default='logs/model.h5',
                        help='The path of best model')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The number of weighted feature interaction layers')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='The number of semantic spaces')
    parser.add_argument("--embedding_l2", type=float, default=0.0,
                        help="L2 reguralizer factor for embedding")
    parser.add_argument("--weight_l1", type=float, default=0.0,
                        help="L2 reguralizer factor for weight")
    parser.add_argument("--logits_type", type=str, choices=["dense", "sum"], default="sum",
                        help="How to calculate logits for prediction")
    parser.add_argument('--reduce_head', type=str, choices=['mean', 'sum'], default='sum',
                        help='Reduction manner used to reduce between multiple heads')
    parser.add_argument('--adaptive_weight', action='store_false',
                        help='Whether use adaptive weights. Default is True')
    parser.add_argument('--with_deep', action='store_true',
                        help="Whether use deep component with our model")
    parser.add_argument('--multi_space', action='store_false',
                        help="Whether each field has its own projection kernel")
    parser.add_argument('--residual', action='store_false',
                        help='Whether use residual connection')
    parser.add_argument("--reguralizer", type=str, default="none",
                        help="Reguralizer type for FIL kernel")
    parser.add_argument("--num_bases", type=int, default=2,
                        help="Rank or block num for the reguralizer")
    parser.add_argument('--do_train', action='store_true',
                        help='Train and evaluate the model')
    parser.add_argument('--do_eval', action='store_true',
                        help='Evaluate the model')
    args = parser.parse_args()

    return args


def _load_dataset(data_file, args):
    """Load TFRecord dataset from `data_file` given arguments"""
    parse_example_fn = make_parse_example_fn(
        args.sparse_field_num + args.dense_field_num)
    filenames = [os.path.join(args.base_dir, data_file)]
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.shuffle(buffer_size=args.shuffle_buffer_size)
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.map(
        parse_example_fn, num_parallel_calls=args.num_parallel_calls)
    if args.cache_in_memory:
        dataset = dataset.cache()
    dataset = dataset.prefetch(args.prefetch_buffer_size)
    return dataset


def _set_random_seed(seed):
    """Set random seed for `numpy` and `TensorFlow` to get reproducible results.
    """
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)


def _set_gpu_option():
    """Make TensorFlow dynamiclly allocate the GPU memory"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def _train_and_eval(train_set, eval_set, model, args):
    """Train and evaluate the given model on train and test dataset"""
    callback_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', min_delta=0.0001, patience=3, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path, monitor='val_auc', save_best_only=True, mode='max'),
        tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=0),
        tf.keras.callbacks.LearningRateScheduler(
            lambda e, lr: max(lr * (args.lr_scale ** e), args.min_lr))
    ]
    model.fit(train_set, epochs=args.epochs,
              callbacks=callback_list, validation_data=eval_set)


def _evaluate(dataset, model, args):
    """Evalute the model on the given `dataset`, e.g. test dataset"""
    model.evaluate(dataset)


def main():
    args = _parse_args()
    print(args)
    _set_random_seed(args.random_seed)
    _set_gpu_option()

    model = Model(args)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.AUC()])
                #   run_eagerly=True)
    model.build(args)
    model.summary()

    if args.do_train:
        train_dataset = _load_dataset(args.train_file, args)
        eval_dataset = _load_dataset(args.valid_file, args)
        _train_and_eval(train_dataset, eval_dataset, model, args)
    elif args.do_eval:
        eval_dataset = _load_dataset(args.test_file, args)
        model.load_weights(args.model_path)
        _evaluate(eval_dataset, model, args)


if __name__ == '__main__':
    main()
