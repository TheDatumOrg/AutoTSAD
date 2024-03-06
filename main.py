# -*- coding: utf-8 -*-
"""
Automated Anomaly Detection Solution for Time Series
@author: Qinghua Liu (liu.11085@osu.edu)
"""

import numpy as np
import math
import pandas as pd
import argparse, pickle, sys
from pathlib import Path
import pycatch22 as catch22
from collections import Counter

from utils import gen_as_from_det, gen_as_from_set, gen_as_unif_from_set, gen_autood_initial_set, split_ts
from TSB_UAD.utils.slidingWindows import find_length_rank
from TSB_UAD.models.feature import Window
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.metrics import metricor

from AutoTSAD.Ensembling_based.OE import OE_AOM
from AutoTSAD.Ensembling_based.HITS import HITS
from AutoTSAD.Ensembling_based.UE import Unsupervised_ens

from AutoTSAD.Pseudo_label_based.Aug import Label_Aug
from AutoTSAD.Pseudo_label_based.Clean import Label_Clean

from AutoTSAD.Internal_Evaluation.EM_MV import Excess_Mass
from AutoTSAD.Internal_Evaluation.CQ import cluster_metrics_xbs
from AutoTSAD.Internal_Evaluation.MC import Model_Centrality
from AutoTSAD.Internal_Evaluation.Synthetic import simulated_ts_stl, synthetic_anomaly_injection_type

Candidate_Model_Set = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 
                       'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5', 'NORMA_1_hierarchical', 'NORMA_3_kshape', 
                       'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
                       'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']
# NORMA requires license permission

Candidate_Model_Set_Sub = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 
                       'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5', 
                       'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
                       'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']

if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Automated Solutions for TSAD')
    parser.add_argument('-a', '--Automated_Solution', nargs='*', default='OE', required=True)
    parser.add_argument('-d', '--data_direc', type=str, default='sample/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.out')
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_direc, header=None).dropna().to_numpy()
    # Use the initial 2000 time steps for test
    data = df[:2000,0].astype(float)
    label = df[:,1].astype(int)
    slidingWindow = find_length_rank(data, rank=1)

    ### Automated Solutions
    if 'OE_AOM' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set, args)
        Anomaly_score = OE_AOM(det_scores)

    if 'UE' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set, args)
        Anomaly_score = Unsupervised_ens(det_scores)

    if 'HITS' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set, args)
        Anomaly_score = HITS(det_scores)

    if 'Aug' in args.Automated_Solution:
        X = Window(window = slidingWindow).convert(data).to_numpy()
        data_window = np.array([X[0]]*math.ceil((slidingWindow-1)/2) + list(X) + [X[-1]]*((slidingWindow-1)//2))
        pred, all_scores, det_scores, instance_index_ranges, detector_index_ranges = gen_autood_initial_set(data, Candidate_Model_Set, args)
        classifier_result_list, scores_for_training = Label_Aug(pred, all_scores, data_window, label=None, instance_index_ranges=instance_index_ranges, 
                                                                detector_index_ranges=detector_index_ranges, max_iteration=10, n_jobs=args.jobs)
        Anomaly_score = np.mean(scores_for_training, axis=1)
    
    if 'Clean' in args.Automated_Solution:
        X = Window(window = slidingWindow).convert(data).to_numpy()
        data_window = np.array([X[0]]*math.ceil((slidingWindow-1)/2) + list(X) + [X[-1]]*((slidingWindow-1)//2))
        predictions_clean = Label_Clean(data_window, label, pred, det_scores, max_iteration=10, initial_set='majority')      # initial_set=['majority', 'ratio', 'avg', 'individual']
        Anomaly_score = predictions_clean

    if 'EM' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set, args)
        det_unif_scores = gen_as_unif_from_set(data, Candidate_Model_Set, args)
        EM_list = Excess_Mass(data, det_scores, det_unif_scores, Candidate_Model_Set)
        Anomaly_score = det_scores.T[EM_list.index(max(EM_list))]

    if 'CQ' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set, args)
        Cluster_XB = cluster_metrics_xbs(det_scores)
        Anomaly_score = det_scores.T[Cluster_XB.index(min(Cluster_XB))]

    if 'MC' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set, args)
        MC_list = Model_Centrality(det_scores, n_neighbors=[5])
        Anomaly_score = det_scores.T[MC_list.index(min(MC_list))]

    if 'Synthetic' in args.Automated_Solution:
        data_simulated = simulated_ts_stl(data, slidingWindow)
        synthetic_performance_list = synthetic_anomaly_injection_type(data_simulated, Candidate_Model_Set, anomaly_type='cutoff')
        selected_model_id = np.argmax(synthetic_performance_list)
        selected_model = Candidate_Model_Set[selected_model_id]
        Anomaly_score = gen_as_from_det(data, selected_model, args)

    if 'CLF' in args.AutoML_Solution_Pool:
        ts_win = split_ts(data, window_size=1024)
        meta_mat = np.zeros([ts_win.shape[0], 24])
        for i in range(ts_win.shape[0]):
            catch24_output = catch22.catch22_all(list(ts_win[i].ravel()), catch24=True)
            meta_mat[i, :] = catch24_output['values']
        meta_mat = pd.DataFrame(meta_mat)
        meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)
        model_path = 'Pretraining_based/Pretrained_weights/CLF.pkl'        
        filename = Path(model_path)
        with open(f'{filename}', 'rb') as input:
            model = pickle.load(input)
        preds = model.predict(meta_mat)
        counter = Counter(preds)
        most_voted = counter.most_common(1)
        det = Candidate_Model_Set[int(most_voted[0][0])]
        Anomaly_score = gen_as_from_det(data, det)

    if 'UReg' in args.AutoML_Solution_Pool:
        ts_win = split_ts(data, window_size=1024)
        meta_mat = np.zeros([ts_win.shape[0], 24])
        for i in range(ts_win.shape[0]):
            catch24_output = catch22.catch22_all(list(ts_win[i].ravel()), catch24=True)
            meta_mat[i, :] = catch24_output['values']
        meta_mat = pd.DataFrame(meta_mat)
        meta_mat = meta_mat.replace([np.nan, np.inf, -np.inf], 0)
        result_path = f'Pretraining_based/Pretrained_weights/UReg.pkl'
        with open(result_path, 'rb') as file:
            result = pickle.load(file)

        U_pred = np.column_stack([rf.predict(meta_mat) for rf in result['models']])
        prediction_scores = U_pred.dot(result['DVt'])
        preds = np.argmax(prediction_scores, axis=1)
        counter = Counter(preds)
        most_voted = counter.most_common(1)
        det = Candidate_Model_Set[int(most_voted[0][0])]
        Anomaly_score = gen_as_from_det(data, det)

    ### Evaluation
    Evaluation_result = get_metrics(Anomaly_score, label, metric='all', slidingWindow=slidingWindow)
    print('Evaluation_result: ', Evaluation_result)
