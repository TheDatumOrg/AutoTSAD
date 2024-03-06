#  This function is adapted from [ADRecommender] by [Jose Manuel Navarro et al.]
#  Original source: [https://figshare.com/articles/code/Meta-Learning_for_Fast_Model_Recommendation_in_Unsupervised_Multivariate_Time_Series_Anomaly_Detection/22320367]

from umap import UMAP
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import argparse
import os
import re
import pickle

def fit_u_regression(train_performance, train_metafeatures, n_factors=5):
    # Adjust n_factors if necessary
    n_factors = min(n_factors, train_performance.shape[1])
    
    # SVD Decomposition
    svd = TruncatedSVD(n_components=n_factors)
    U = svd.fit_transform(train_performance)
    D = np.diag(svd.singular_values_)
    Vt = svd.components_
    DVt = D.dot(Vt)
    
    # Train a Random Forest for each factor
    models = []
    for i in range(n_factors):
        y = U[:, i]
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, verbose=True)
        rf.fit(train_metafeatures, y)
        models.append(rf)
    
    result = {
        'models': models,
        'DVt': DVt,
        'configurations': train_performance.columns,
        'recommender_type': 'URegression (RF)'
    }    
    return result

def fit_cfact(train_performance, train_metafeatures, n_factors=5):
    # UMAP projection
    umap = UMAP(n_components=10, metric='manhattan')
    algo_proj = umap.fit_transform(train_performance.T)
    
    # Clustering
    algo_clust = DBSCAN(eps=1).fit(algo_proj).labels_

    # Fit UReg model for each cluster
    models = []
    unique_clusters = np.unique(algo_clust)
    for clust_num in unique_clusters:
        indices = np.where(algo_clust == clust_num)[0]

        subset_performance = train_performance.iloc[:, indices]
        # subset_metafeatures = train_metafeatures[:, indices]

        print('subset_performance: {}'.format(subset_performance.shape))

        try:
            u_reg = fit_u_regression(subset_performance, train_metafeatures, n_factors)
        except Exception as e:
            print('Detected error with the internal SVD computations. Trying again.')
            continue
        models.append(u_reg)

    result = {
        'models': models,
        'cluster_indices': algo_clust,
        'configurations': train_performance.columns,
        'recommender_type': 'cFact'
    }
    
    return result

def train_cfact(data_path, test_dataset=None):
    # Set up
    window_size = int(re.search(r'\d+', data_path).group())
    training_stats = {}
    original_dataset = data_path.split('/')[:-1]
    original_dataset = '/'.join(original_dataset)
    
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

    ### 
    # train_indexes = train_indexes[:50]

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

    y_train, X_train = training_data.iloc[:, :23], training_data.iloc[:, 23:]
    y_val, X_val = val_data.iloc[:, :23], val_data.iloc[:, 23:]
    y_test, X_test = test_data.iloc[:, :23], test_data.iloc[:, 23:]

    X_train = X_train.replace([np.nan, np.inf, -np.inf], 0)
    X_val = X_val.replace([np.nan, np.inf, -np.inf], 0)

    result = fit_cfact(y_train, X_train)

    with open(f'results/weights_ranking/cfact/{test_dataset}.pkl', 'wb') as file:
        pickle.dump(result, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024_ranking_value/Catch22_TSB_1024_ranking_value.csv')
    parser.add_argument('-t', '--test_dataset', type=str, default='NAB')

    args = parser.parse_args()

    train_cfact(
        data_path=args.path,
        test_dataset=args.test_dataset
    )