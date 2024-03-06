#  This function is adapted from [MSAD] by [boniolp]
#  Original source: [https://github.com/boniolp/MSAD]

import argparse
import os
from time import perf_counter
import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC

from utils.evaluator import save_classifier
from utils.config import *

names = {
        "knn": "Nearest Neighbors",
        "svc_linear": "Linear SVM",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "mlp": "Neural Net",
        "ada_boost": "AdaBoost",
        "bayes": "Naive Bayes",
        "qda": "QDA",
}

classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "svc_linear": LinearSVC(C=0.025, verbose=True),
        "decision_tree": DecisionTreeClassifier(max_depth=5),
        "random_forest": RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=4, verbose=True),
        "mlp": MLPClassifier(alpha=1, max_iter=1000, verbose=True),
        "ada_boost": AdaBoostClassifier(),
        "bayes": GaussianNB(),
        "qda": QuadraticDiscriminantAnalysis(),
}


def train_CLF(data_path, classifier_name, split_per=0.7, seed=None, read_from_file=None, eval_model=False, path_save=None, test_dataset=None):
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
    
    # Split data from labels
    y_train, X_train = training_data['label'], training_data.drop('label', axis=1)
    y_val, X_val = val_data['label'], val_data.drop('label', axis=1)
    y_test, X_test = test_data['label'], test_data.drop('label', axis=1)

    X_train = X_train.replace([np.nan, np.inf, -np.inf], 0)
    X_val = X_val.replace([np.nan, np.inf, -np.inf], 0)

    # Select the classifier
    classifier = classifiers[classifier_name]
    clf_name = classifier_name

    # For svc_linear use only a random subset of the dataset to train
    if 'svc' in classifier_name and len(y_train) > 200000:
        rand_ind = np.random.randint(low=0, high=len(y_train), size=200000)
        X_train = X_train.iloc[rand_ind]
        y_train = y_train.iloc[rand_ind]

    # Fit the classifier
    print(f'----------------------------------\nTraining {names[classifier_name]}...')
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
    save_done_training_ml = save_done_training+'/'+classifier_name
    os.makedirs(save_done_training_ml, exist_ok=True)
    classifier_name = f"{clf_name}_{window_size}_Metaod"
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    df = pd.DataFrame.from_dict(training_stats, columns=["training_stats"], orient="index")
    df.to_csv(os.path.join(save_done_training_ml, f"{classifier_name}_{test_dataset}.csv"))

    # Save pipeline
    saving_dir = os.path.join(path_save, classifier_name) if classifier_name.lower() not in path_save.lower() else path_save
    saved_model_path = save_classifier(classifier, saving_dir, fname=test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024/Catch22_TSB_1024.csv')
    # parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024/TSFRESH_TSB_1024.csv')
    # parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024/Metaod_TSB_1024.csv')
    parser.add_argument('-c', '--classifier', type=str, help='classifier to run', default='random_forest')
    parser.add_argument('-e', '--eval-true', help='whether to evaluate the model on test data after training', default=False)
    parser.add_argument('-ps', '--path_save', type=str, help='path to save the trained classifier', default="results/weights")
    parser.add_argument('-t', '--test_dataset', type=str, default='ID')

    args = parser.parse_args()

    # Option to all classifiers
    if args.classifier == 'all':
        clf_list = list(classifiers.keys())
    else:
        clf_list = [args.classifier]

    for classifier in clf_list:
        train_CLF(
            data_path=args.path,
            classifier_name=classifier,
            eval_model=args.eval_true,
            path_save=args.path_save,
            test_dataset=args.test_dataset
        )
