import numpy as np
import pandas as pd
import argparse
import re
import os
from utils.data_loader import DataLoader
import pycatch22 as catch22

def generate_features(path, ranking):
    """Given a dataset it computes the TSFresh automatically extracted 
    features and saves the new dataset (which does not anymore contain
    time series but tabular data) into one .csv in the folder of the
    original dataset

    :param path: path to the dataset to be converted
    """
    print('ranking: ', ranking)
    # Create name of new dataset
    if ranking:
        new_name = f"Catch22_TSB_1024_ranking_value.csv"
    else:
        new_name = f"Catch22_TSB_1024.csv"

    # Load datasets
    dataloader = DataLoader(path)
    datasets = dataloader.get_dataset_names()
    df = dataloader.load_df(datasets)
    
    # Divide df
    if ranking:
        labels = df.iloc[:, :23]
        x = df.iloc[:, 23:].to_numpy()    # (264762, 1024)
    else:
        labels = df.pop("label")
        x = df.to_numpy()
    
    index = df.index

    meta_mat = np.zeros([x.shape[0], 24])
    for i in range(x.shape[0]):
        catch24_output = catch22.catch22_all( list(x[i].ravel()), catch24=True)
        # print('catch24_output: ', catch24_output)
        meta_mat[i, :] = catch24_output['values']
        fnames24 = catch24_output['names']

    X_transformed = pd.DataFrame(meta_mat, columns=np.array(fnames24))
    # print('X_transformed: ', X_transformed)     # (264762, 24)
    print('labels: ', labels) 

    # Create new dataframe
    X_transformed.index = index
    X_transformed = pd.merge(labels, X_transformed, left_index=True, right_index=True)
    print('X_transformed: ', X_transformed)     # (264762, 25)

    # Save new features
    X_transformed.to_csv(os.path.join(path, new_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_features',
        description='Transform a dataset of time series (of equal length) to tabular data with catch22'
    )
    parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', default='data/TSB_1024_ranking_value_sub/')
    parser.add_argument('-r', '--ranking', action="store_true")
    
    args = parser.parse_args()
    generate_features(
        path=args.path,
        ranking=args.ranking
    )