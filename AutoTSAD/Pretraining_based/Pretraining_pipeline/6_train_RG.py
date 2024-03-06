import argparse
import os
from time import perf_counter
import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from utils.evaluator import save_classifier
from utils.config import *

classifiers = {
        "knn": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, verbose=True)
}

def train_feature_based(data_path, classifier_name, eval_model=False, path_save=None, test_dataset=None):
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
        test_list = pd.read_csv('file_list/eva/eva_file_list.csv').values.tolist()
    else:
        train_list = pd.read_csv(f'data/split_list/ood/{test_dataset}/train_list.csv').values.tolist()
        val_list = pd.read_csv(f'data/split_list/ood/{test_dataset}/val_list.csv').values.tolist()
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

    print(test_data)

    y_train, X_train = training_data.iloc[:, :23], training_data.iloc[:, 23:]
    y_val, X_val = val_data.iloc[:, :23], val_data.iloc[:, 23:]
    y_test, X_test = test_data.iloc[:, :23], test_data.iloc[:, 23:]

    X_train = X_train.replace([np.nan, np.inf, -np.inf], 0)
    X_val = X_val.replace([np.nan, np.inf, -np.inf], 0)

    # X_train = X_train.to_numpy().astype('float64')
    # X_val = X_val.to_numpy().astype('float64')
    # meta_scalar = MinMaxScaler()
    # X_train = meta_scalar.fit_transform(X_train)
    # X_val = meta_scalar.fit_transform(X_val)
    # X_train[np.isinf(X_train)] = np.nan
    # X_train[np.isnan(X_train)] = np.nanmean(X_train)
    # X_val[np.isinf(X_val)] = np.nan
    # X_val[np.isnan(X_val)] = np.nanmean(X_val)

    # Select the classifier
    classifier = classifiers[classifier_name]
    clf_name = classifier_name

    # Fit the classifier
    tic = perf_counter()
    classifier.fit(X_train, y_train)
    toc = perf_counter()

    # Print training time
    training_stats["training_time"] = toc - tic
    print(f"training time: {training_stats['training_time']:.3f} secs")
    
    # Print valid accuracy and inference time
    tic = perf_counter()
    classifier_score = classifier.score(X_val, y_val)
    toc = perf_counter()
    training_stats["val_acc"] = classifier_score
    training_stats["avg_inf_time"] = ((toc-tic)/X_val.shape[0]) * 1000
    print(f"valid accuracy: {training_stats['val_acc']:.3%}")
    print(f"inference time: {training_stats['avg_inf_time']:.3} ms")

    # Save training stats
    save_done_training_ml = save_done_training+'/'+classifier_name+'_ranking'
    os.makedirs(save_done_training_ml, exist_ok=True)
    classifier_name = f"{clf_name}_{window_size}"
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    df = pd.DataFrame.from_dict(training_stats, columns=["training_stats"], orient="index")
    df.to_csv(os.path.join(save_done_training_ml, f"{classifier_name}_{test_dataset}.csv"))

    # Save pipeline
    saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
    saved_model_path = save_classifier(classifier, saving_dir, fname=test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024_ranking_value/Catch22_TSB_1024_ranking_value.csv')
    parser.add_argument('-c', '--classifier', type=str, help='classifier to run', default='random_forest')
    parser.add_argument('-e', '--eval-true', help='whether to evaluate the model on test data after training', default=False)
    parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights_ranking")
    parser.add_argument('-t', '--test_dataset', type=str, default='NAB')

    args = parser.parse_args()

    train_feature_based(
        data_path=args.path,
        classifier_name=args.classifier,
        eval_model=args.eval_true,
        path_save=args.path_save,
        test_dataset=args.test_dataset
    )
