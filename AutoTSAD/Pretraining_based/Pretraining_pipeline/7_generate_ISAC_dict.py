import argparse
import os
from time import perf_counter
import re
from collections import Counter
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from eval_feature_based import eval_feature_based
from utils.evaluator import save_classifier
from utils.config import *

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.gmeans import gmeans
from pyclustering.utils import read_sample
from joblib import dump, load

Det_Pool = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5',
            'NORMA_1_hierarchical', 'NORMA_3_kshape', 'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
            'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

def train_feature_based(data_path, path_save=None, test_dataset=None):
    # Read Train/Val/Test splits read from file
    train_set, val_set, test_set = [], [], []
    if test_dataset == 'ID':
        train_list = pd.read_csv(f'data/split_list/ind/train_list.csv').values.tolist()
        val_list = pd.read_csv(f'data/split_list/ind/val_list.csv').values.tolist()        
        test_list = pd.read_csv('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/file_list/eva/eva_file_list.csv').values.tolist()
    else:
        train_list = pd.read_csv(f'data/split_list/ood/{test_dataset}/train_list.csv').values.tolist()
        val_list = pd.read_csv(f'data/split_list/ood/{test_dataset}/val_list.csv').values.tolist()
        test_list_all = pd.read_csv('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/file_list/eva/eva_file_list.csv')
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

    print(test_data)
    
    # Split data from labels
    y_train, X_train = training_data['label'], training_data.drop('label', axis=1)
    y_val, X_val = val_data['label'], val_data.drop('label', axis=1)

    train_meta = X_train.to_numpy().astype('float64')
    valid_meta = X_val.to_numpy().astype('float64')
    X_tr_val = np.concatenate((train_meta, valid_meta), axis=0)
    X_tr_val[np.isinf(X_tr_val)] = np.nan
    X_tr_val[np.isnan(X_tr_val)] = np.nanmean(X_tr_val)

    # X_tr_val = X_tr_val[:100,]

    gmeans_instance = gmeans(X_tr_val).process()

    clf_name = 'gmeans'
    os.makedirs(f'{path_save}/{clf_name}', exist_ok=True)
    dump(gmeans_instance, f'{path_save}/{clf_name}/{test_dataset}.joblib')

    clusters = gmeans_instance.get_clusters()
    cluster_dict = {i: 0 for i in range(len(clusters))}
    for i, cluster in enumerate(clusters):
        det_performance_dict = {det: 0 for det in Det_Pool}
        for id in cluster:
            file_name_list = []
            file_name = data_index[id].rsplit('.', 1)[0]
            file_name_list.append(file_name)
        for file_name in set(file_name_list):
            for det in Det_Pool:
                df = pd.read_csv(f"/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/AutoAD/model_performance/{file_name.split('/')[0]}/{det}.csv")
                det_result = df.loc[df['file']==file_name.split('/')[1]].drop(columns=['file']).to_dict('list')
                for key, value in det_result.items(): det_result[key] = value[0]  
                det_performance_dict[det] += det_result['AUC_PR']    
        det_cluster = max(det_performance_dict, key=det_performance_dict.get)
        cluster_dict[i] = det_cluster
    np.save(f'{path_save}/{clf_name}/{test_dataset}_cluster_dict.npy', cluster_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024/Catch22_TSB_1024.csv')
    parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights")
    parser.add_argument('-t', '--test_dataset', type=str, default='ID')

    args = parser.parse_args()
    train_feature_based(
        data_path=args.path,
        path_save=args.path_save,
        test_dataset=args.test_dataset
    )
