#  This function is adapted from [EMMV_benchmarks] by [ngoix]
#  Original source: [https://github.com/ngoix/EMMV_benchmarks]

import numpy as np
from sklearn.metrics import auc
from sklearn.utils import shuffle as sh
import sys
sys.path.append("../..")
from TSB_UAD.TSB_run_det_unif import *

def em(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                          t * (s_unif > u).sum() / n_generated
                          * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        print ('\n failed to achieve t_max \n')
        amax = -1
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


def mv(axis_alpha, volume_support, s_unif, s_X, n_generated):
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        # pdb.set_trace()
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv

def Excess_Mass(data, det_scores, det_unif_scores, Det_Pool, alpha_min=0.9, alpha_max=0.999,
                       n_generated=100000, t_max = 0.9):
    
    em_list = []
    data = data.reshape(-1, 1)
    n_features = data.shape[1]
    
    lim_inf = data.min(axis=0)
    lim_sup = data.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    if volume_support == 0:
        volume_support = ((lim_sup - lim_inf) + 0.001).prod()
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup,
                              size=(n_generated, n_features))

    for i, det in enumerate(Det_Pool):
        s_X_clf = det_scores[i]* -1
        s_unif_clf = det_unif_scores[i]* -1    
        
        em_clf = em(t, t_max, volume_support, s_unif_clf,
                    s_X_clf, n_generated)[0]
        em_list.append(em_clf)
        # print('det: {}, em_clf: {}'.format(det, em_clf))
    
    return em_list   # EM larger better, MV smaller better

def Mass_Volume(data, det_scores, det_unif_scores, Det_Pool, alpha_min=0.9, alpha_max=0.999,
                       n_generated=100000, t_max = 0.9):
    
    mv_list = []
    data = data.reshape(-1, 1)
    n_features = data.shape[1]
    
    lim_inf = data.min(axis=0)
    lim_sup = data.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    if volume_support == 0:
        volume_support = ((lim_sup - lim_inf) + 0.001).prod()
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup,
                              size=(n_generated, n_features))

    for i, det in enumerate(Det_Pool):
        s_X_clf = det_scores[i]* -1
        s_unif_clf = det_unif_scores[i]* -1
    
        mv_clf = mv(axis_alpha, volume_support, s_unif_clf,
                    s_X_clf, n_generated)[0]
        mv_list.append(mv_clf)
        # print('det: {}, mv_clf: {}'.format(det, mv_clf))
    
    return mv_list   # EM larger better, MV smaller better