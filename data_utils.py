#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

from typing import List, Dict, Iterator


def _BKDRHash(string):
    """Hash a string to int64"""
    seed = 131
    hash = 0
    for ch in string:
        hash = hash * seed + ord(ch)
    return hash & 0x7FFFFFFF


def _make_example(data) -> tf.train.Example:
    """Generate tf.train.Example instance given dictionary data and feature names.

    Args:
        data (dict): One example containing all feature values.

    Returns:
        tf.train.Example: Ready for TFRecord.
    """
    features = {
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=data['x'])),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=data['y']))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def _make_feature_description(featuer_cols) -> Dict[str, tf.io.FixedLenFeature]:
    """Create feature description for tfrecord parsing"""
    feature_columns = {
        'x': tf.io.FixedLenFeature(
            shape=[featuer_cols],
            dtype=tf.float32,
            default_value=[0.0 for _ in range(featuer_cols)]),
        'y': tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
    }
    return feature_columns


def make_parse_example_fn(featuer_cols):
    """Create a function for parsing tf.train.Example"""
    feature_columns = _make_feature_description(featuer_cols)

    def _parse_example(example_proto):
        example = tf.io.parse_example(example_proto, feature_columns)
        return (example['x'], example['y'])

    return _parse_example


def _write_tfrecord(filename: str,
                    data_iterator: Iterator):
    """Write DataFrame data to TFRecord"""
    writer = tf.io.TFRecordWriter(filename)
    for line in data_iterator:
        ex = _make_example(line)
        writer.write(ex.SerializeToString())
    writer.close()


def _make_criteo_iterator(file: str):
    """Create a generator for parsing criteo dataset
    """
    with open(file, 'r') as data_file:
        line = data_file.readline()
        while line:
            one_example = dict()
            contents = line.split('\t')
            one_example['y'] = [float(contents[0])]

            dense_features = [0 if len(contents[index]) == 0 else float(contents[index])
                              for index in range(1, 14)]
            sparse_features = [float(contents[index])
                               for index in range(14, 40)]
            one_example['x'] = sparse_features + dense_features
            yield one_example
            line = data_file.readline()


def _make_avazu_iterator(file: str):
    """Create a generator for parsing avazu dataset
    """
    with open(file, 'r') as data_file:
        line = data_file.readline()
        while line:
            one_example = dict()
            contents = line.split('\t')
            one_example['y'] = [float(contents[1])]

            sparse_features = [float(contents[index])
                               for index in range(2, 24)]
            one_example['x'] = sparse_features
            yield one_example
            line = data_file.readline()


def _make_movielens_1m_iterator(file: str):
    """Create a generator for parsing avazu dataset
    """
    with open(file, 'r') as data_file:
        line = data_file.readline()
        while line:
            one_example = dict()
            contents = line.split('\t')
            one_example['y'] = [float(contents[0])]

            sparse_features = [float(contents[index])
                               for index in range(1, 13)]
            one_example['x'] = sparse_features
            yield one_example
            line = data_file.readline()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_dir", type=str, default='data', help="Directory to store target files")
    argparser.add_argument("--dataset", type=str, default='movielens-1m')
    args = argparser.parse_args()

    for datatype in ['train', 'valid', 'test']:
        data_iter = None
        data_file = os.path.join(args.base_dir, args.dataset, '{}.csv'.format(datatype))
        tfrecord_file = os.path.join(args.base_dir, args.dataset, '{}.tfrecords'.format(datatype))
        if args.dataset == 'criteo':
            data_iter = _make_criteo_iterator(data_file)
        elif args.dataset == 'avazu':
            data_iter = _make_avazu_iterator(data_file)
        elif args.dataset == 'movielens-1m':
            data_iter = _make_movielens_1m_iterator(data_file)

        assert data_iter is not None
        _write_tfrecord(tfrecord_file, data_iter)


if __name__ == '__main__':
    main()
