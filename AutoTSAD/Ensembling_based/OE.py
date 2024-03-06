import numpy as np
from sklearn.preprocessing import StandardScaler

def OE_AVG(det_scores):
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)
    avg_ens_scores = np.mean(standardized_det_scores, axis=1)
    return avg_ens_scores

def OE_MAX(det_scores):
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)
    max_ens_scores = np.max(standardized_det_scores, axis=1)
    return max_ens_scores

def OE_AOM(det_scores):
    scaler = StandardScaler()
    scaler.fit(det_scores)
    standardized_det_scores = scaler.transform(det_scores)

    max_ens_scores_list = []
    for i in range(20):
        indices = np.random.choice(standardized_det_scores.shape[1], 5, replace=False)
        max_ens_scores_list.append(np.max(standardized_det_scores[:,indices], axis=1))
    avg_of_max_ens_scores = np.mean(np.array(max_ens_scores_list), axis=0)
    return avg_of_max_ens_scores