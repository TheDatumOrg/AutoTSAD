import numpy as np
import math
import pandas as pd
import sys
from TSB_UAD.TSB_run_det import *
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length,plotFig,find_length_rank
from TSB_UAD.vus.metrics import get_metrics
from TSB_UAD.utils.metrics import metricor
import argparse

## AutoML Solution Lib
from AutoTSAD.Ensembling_based.OE import OE_AOM


Candidate_Model_Set = ['IForest_3_200', 'IForest_1_100', 'IForest_0_200', 'LOF_3_60', 'LOF_1_30', 
                       'MP_2_False', 'MP_1_True', 'PCA_3_None', 'PCA_1_0.5', 'NORMA_1_hierarchical', 'NORMA_3_kshape', 
                       'HBOS_3_20', 'HBOS_1_40', 'POLY_3_5', 'POLY_2_1', 'OCSVM_1_rbf', 'OCSVM_3_poly',
                       'AE_1_1_bn', 'AE_2_0_dropout', 'CNN_2_0_relu', 'CNN_3_1_sigmoid', 'LSTM_1_1_relu', 'LSTM_3_1_sigmoid']


if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Automated Solutions for TSAD')
    parser.add_argument('-a', '--Automated_Solution', nargs='*', default='OE', required=True)
    parser.add_argument('-d', '--data_direc', type=str, default='sample/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.out')
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_direc, header=None).dropna().to_numpy()
    data = df[:,0].astype(float)
    label = df[:,1].astype(int)

    if 'OE (AOM)' in args.Automated_Solution:
        det_scores = gen_as_from_set(data, Candidate_Model_Set)
        Anomaly_score = OE_AOM(det_scores)