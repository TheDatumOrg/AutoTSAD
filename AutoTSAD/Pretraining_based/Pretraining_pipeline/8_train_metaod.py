#  This function is adapted from [MetaOD] by [yzhao062]
#  Original source: [https://github.com/yzhao062/MetaOD]

import argparse
import os
import re
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from utils.config import *
from utils.Metaod_core import MetaODClass
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

def train_feature_based(data_path, path_save=None, test_dataset=None):
    
    # Read Train/Val/Test splits read from file
    train_set, val_set, test_set = [], [], []
    if test_dataset == 'ID':
        train_list = pd.read_csv(f'/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/pretraining_based/data/split_list/ind/train_list.csv').values.tolist()
        val_list = pd.read_csv(f'/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/pretraining_based/data/split_list/ind/val_list.csv').values.tolist()        
        test_list = pd.read_csv('file_list/eva/eva_file_list.csv').values.tolist()
    else:
        train_list = pd.read_csv(f'/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/pretraining_based/data/split_list/ood/{test_dataset}/train_list.csv').values.tolist()
        val_list = pd.read_csv(f'/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/pretraining_based/data/split_list/ood/{test_dataset}/val_list.csv').values.tolist()
        test_list_all = pd.read_csv('file_list/eva/eva_file_list.csv')
        test_list = test_list_all[test_list_all['Dataset']==test_dataset].values.tolist()
    train_set.extend(os.path.join(file[0]+'/'+file[1]+'.csv') for file in train_list)
    val_set.extend(os.path.join(file[0]+'/'+file[1]+'.csv') for file in val_list)
    test_set.extend(os.path.join(file[0]+'/'+file[1]+'.csv') for file in test_list)

    train_indexes = [x[:-4] for x in train_set]
    val_indexes = [x[:-4] for x in val_set]
    test_indexes = [x[:-4] for x in test_set]

    # Read tabular data
    data = pd.read_csv(data_path, index_col=0)

    # Reindex them
    data_index = list(data.index)
    new_index = [tuple(x.rsplit('.', 1)) for x in data_index]
    new_index = pd.MultiIndex.from_tuples(new_index, names=["name", "n_window"])
    data.index = new_index
    
    # Create subsets
    training_data = data.loc[data.index.get_level_values("name").isin(train_indexes)]
    val_data = data.loc[data.index.get_level_values("name").isin(val_indexes)]
    test_data = data.loc[data.index.get_level_values("name").isin(test_indexes)]

    print('test_data:', test_data)
    
    # Split data from labels
    y_train, X_train = training_data.iloc[:, :23], training_data.iloc[:, 23:]
    y_val, X_val = val_data.iloc[:, :23], val_data.iloc[:, 23:]
    y_test, X_test = test_data.iloc[:, :23], test_data.iloc[:, 23:]

    # print('y_test:', y_test.to_numpy())

    train_set = y_train.to_numpy().astype('float64')
    valid_set = y_val.to_numpy().astype('float64')

    train_meta = X_train.to_numpy().astype('float64')
    valid_meta = X_val.to_numpy().astype('float64')

    meta_scalar = MinMaxScaler()
    train_meta = meta_scalar.fit_transform(train_meta)
    valid_meta = meta_scalar.fit_transform(valid_meta)

    train_meta[np.isinf(train_meta)] = np.nan
    train_meta[np.isnan(train_meta)] = np.nanmean(train_meta)
    valid_meta[np.isinf(valid_meta)] = np.nan
    valid_meta[np.isnan(valid_meta)] = np.nanmean(valid_meta)

    # train_set = train_set[:50, :]
    # valid_set = valid_set[:50, :]
    # train_meta = train_meta[:50, :]
    # valid_meta = valid_meta[:50, :]

    clf = MetaODClass(train_set, valid_performance=valid_set, n_factors=30,
                    learning='sgd')
    clf.train(meta_features=train_meta, valid_meta=valid_meta, n_iter=10, 
            learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1, n_steps=8)
    dump(clf, f'{path_save}/MetaOD_{test_dataset}.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/pretraining_based/data/TSB_1024_ranking_value/Metaod_TSB_1024_ranking_value.csv')
    parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/pretraining_based/results/weights_ranking/metaod")
    parser.add_argument('-t', '--test_dataset', type=str, default='ID')

    args = parser.parse_args()
    train_feature_based(
        data_path=args.path,
        path_save=args.path_save,
        test_dataset=args.test_dataset)
