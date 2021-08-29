#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_criteo(source_csv, target_dir):
    label_col = ['label']
    dense_cols = ['c_{}'.format(i) for i in range(1, 14)]
    sparse_cols = ['c_{}'.format(i) for i in range(14, 40)]
    col_names = label_col + dense_cols + sparse_cols
    dtypes = {x: np.float32 for x in label_col + dense_cols}
    dtypes.update({x: str for x in sparse_cols})

    # Read data
    df = pd.read_csv(source_csv, sep='\t',
                     header=None, names=col_names, dtype=dtypes)
    print(df.shape)

    # Fill NA
    df = df.apply(lambda x: x.fillna(0) if x.name in dense_cols else x)
    df = df.apply(lambda x: x.fillna('0') if x.name in sparse_cols else x)

    for col in sparse_cols:
        feature_count = df[col].nunique()
        print('column {} has {} unique key before filtering'.format(col, feature_count))
    print('')

    # infrequent feature filtering
    df = df.apply(lambda x: x.apply(lambda y: int(math.log(y)**2) if y > 2 else y) if x.name in dense_cols else x)
    df = df.apply(lambda x: x.mask(x.map(x.value_counts()) <= 10,
                                   'unkown') if x.name in sparse_cols else x)

    # Total feature keys， Encode feature index
    offset = 0
    for col in sparse_cols:
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        unique_keys = len(label_encoder.classes_)
        print('column {} has {} unique key after filtering'.format(col, unique_keys))
        df[col] = label_encoder.transform(df[col]) + offset
        offset += unique_keys
    print('total sparse feature: {}'.format(offset))

    # Split to train, valid and test set
    train, valid, test = np.split(
        df.sample(frac=1, random_state=999).reset_index(drop=True),
        [int(.8 * len(df)), int(.9 * len(df))])

    print('train_size: {}, valid_size: {}, test_size: {}'.format(train.shape, valid.shape, test.shape))

    train.to_csv(os.path.join(target_dir, 'train.csv'), sep='\t', header=0, index=False)
    valid.to_csv(os.path.join(target_dir, 'valid.csv'), sep='\t', header=0, index=False)
    test.to_csv(os.path.join(target_dir, 'test.csv'), sep='\t', header=0, index=False)


def preprocess_avazu(source_csv, target_dir):
    column_names = [
        'id',  # ad identifier
        'click',  # 0/1 for non-click/click
        'hour',  # format is YYMMDDHH
        'C1',  # anonymized categorical variable
        'banner_pos',  # (0/1) position of banner
        'site_id',  # (alphanumeric)
        'site_domain',  # categoryof site's domain (alphanumeric)
        'site_category',  # category of site (alphanumeric)
        'app_id',  # (alphanumeric)
        'app_domain',  # categorization of application domain (alphanumeric)
        'app_category',  # categorization of application (alphanumeric)
        'device_id',  # (alphanumeric)
        'device_ip',  # ip address of the device
        'device_model',  # model of the device
        'device_type',  # type of device (1/0)
        'device_conn_type',  # device connection type (0/2/5)
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'  # anonymized categorical variables
    ]
    dtypes = {x: str for x in column_names}
    feature_cols = set(column_names) - set(['id', 'click'])

    # Read data
    df = pd.read_csv(source_csv, sep=',',
                     header=0, names=column_names, dtype=dtypes)
    print(df.shape)

    # Fill NA
    df = df.apply(lambda x: x.fillna('0') if x.name in feature_cols else x)

    # Count feature nums in each column
    for col in feature_cols:
        feature_count = df[col].nunique()
        print('column {} has {} unique key before filtering'.format(col, feature_count))
    print('')

    # infrequent feature filtering
    df = df.apply(lambda x: x.mask(x.map(x.value_counts()) < 5,
                                   'unkown') if x.name in feature_cols else x)

    # Total feature keys， Encode feature index
    offset = 0
    for col in feature_cols:
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        unique_keys = len(label_encoder.classes_)
        print('column {} has {} unique key after filtering'.format(col, unique_keys))
        df[col] = label_encoder.transform(df[col]) + offset
        offset += unique_keys
    print('total sparse feature: {}'.format(offset))

    # Split to train, valid and test set
    train, valid, test = np.split(
        df.sample(frac=1, random_state=999).reset_index(drop=True),
        [int(.8 * len(df)), int(.9 * len(df))])

    print('train_size: {}, valid_size: {}, test_size: {}'.format(train.shape, valid.shape, test.shape))

    train.to_csv(os.path.join(target_dir, 'train.csv'), sep='\t', header=0, index=False)
    valid.to_csv(os.path.join(target_dir, 'valid.csv'), sep='\t', header=0, index=False)
    test.to_csv(os.path.join(target_dir, 'test.csv'), sep='\t', header=0, index=False)


def preprocess_movielens_1m(source_csv, target_dir):
    from pathlib import Path
    csv_path = Path(source_csv)
    raw_dir = csv_path.parent  # We suppose user.dat and item.dat are in the same directory of source_csv

    rating_file = source_csv
    user_file = os.path.join(raw_dir, 'users.dat')
    movie_file = os.path.join(raw_dir, 'movies.dat')

    # Read user file
    uname = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(user_file, sep='::', header=None,
                          names=uname, engine='python')

    # Read movie file
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(movie_file, header=None,
                           sep='::', names=mnames, engine='python')

    # Read rating file
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(rating_file, header=None,
                            sep='::', names=rnames, engine='python')

    # Merge user, movie properties and rating
    df = pd.merge(pd.merge(ratings, users), movies)

    # Filter sample with rating of 3, and change rating to 1 and 0
    df = df[df.rating != 3]
    df['rating'][df['rating'] < 3] = 0
    df['rating'][df['rating'] > 3] = 1

    # Min-Max normalize timestamp
    max_v, min_v = df['timestamp'].max(), df['timestamp'].min()
    df['timestamp'] = (df['timestamp'] - min_v) / (max_v - min_v)

    # Process title
    df['title'] = df['title'].map(lambda x: int(x[-5: -1]))

    # Split genre attribute
    df['genres_num'] = df['genres'].map(lambda x: len(x.split('|')))
    max_genres = df['genres_num'].max()
    for i in range(max_genres):
        df['genres_{}'.format(i)] = df['genres'].map(
            lambda x: x.split('|')[i] if len(x.split('|')) >= i + 1 else 'none')

    # Total feature keys， Encode feature index
    offset = 0
    field_names = ['gender', 'age', 'occupation', 'zip', 'title']
    for col in field_names:
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        unique_keys = len(label_encoder.classes_)
        df[col] = label_encoder.transform(df[col]) + offset
        offset += unique_keys
    genre_field_names = ['genres_{}'.format(i) for i in range(max_genres)]
    label_encoder = LabelEncoder()
    genres_series = pd.concat([df[col] for col in genre_field_names])
    label_encoder.fit(genres_series)
    for col in genre_field_names:
        df[col] = label_encoder.transform(df[col]) + offset
    unique_keys = len(label_encoder.classes_)
    offset += unique_keys
    print('total sparse feature: {}'.format(offset))
    print('max_genre_num: {}'.format(max_genres))

    # Select columns
    df = df[['rating'] + field_names + genre_field_names + ['timestamp']]

    # Split to train, valid and test set
    train, valid, test = np.split(
        df.sample(frac=1, random_state=999).reset_index(drop=True),
        [int(.8 * len(df)), int(.9 * len(df))])

    print('train_size: {}, valid_size: {}, test_size: {}'.format(train.shape, valid.shape, test.shape))

    train.to_csv(os.path.join(target_dir, 'train.csv'), sep='\t', header=0, index=False)
    valid.to_csv(os.path.join(target_dir, 'valid.csv'), sep='\t', header=0, index=False)
    test.to_csv(os.path.join(target_dir, 'test.csv'), sep='\t', header=0, index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="movielens-1m", help="Dataset name")
    argparser.add_argument("--raw_file", type=str, default='data/movielens-1m/raw/ratings.dat', help="File to preprocess")
    argparser.add_argument("--target_dir", type=str, default='data/movielens-1m', help="Directory to store target files")
    args = argparser.parse_args()

    if args.dataset == 'criteo':
        preprocess_criteo(args.raw_file, args.target_dir)
    elif args.dataset == 'avazu':
        preprocess_avazu(args.raw_file, args.target_dir)
    elif args.dataset == 'movielens-1m':
        preprocess_movielens_1m(args.raw_file, args.target_dir)
    else:
        raise ValueError("type {} not supported".format(args.dataset))
