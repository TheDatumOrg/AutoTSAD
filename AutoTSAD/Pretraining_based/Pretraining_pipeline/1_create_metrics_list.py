from sklearn import metrics
import numpy as np
import os, sys, argparse
import pandas as pd
sys.path.append('..')
from utils import gen_as_from_det

data_root = '/data/liuqinghua/code/ts/TSB-UAD/data/public/'
Candidate_Model_Set = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 
                       'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5', 'NORMA_1_hierarchical', 'NORMA_3_kshape', 
                       'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
                       'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()

    for det in Candidate_Model_Set:
        print('Processing: ', det)

        eval_list = []
        filesList = pd.read_csv('file_list/all_file_list.csv')
        for index, row in filesList.iterrows():

            filepath = data_root + row['Dataset'] + '/' + row['File_name']
            df = pd.read_csv(filepath, header=None).dropna().to_numpy()
            data = df[:,0].astype(float)
            label = df[:,1].astype(int)
            score = gen_as_from_det(data, det, args)
            AUC_PR = metrics.average_precision_score(label, score)
            eval_list.append([row['Dataset'] + '/' + row['File_name'], AUC_PR])

        eval_csv = pd.DataFrame(eval_list, columns=['file_name', f'{det}'])
        os.makedirs(f'data/metrics/{det}', exist_ok=True)
        eval_csv.to_csv(f'data/metrics/{det}/AUC_PR.csv', index=False)