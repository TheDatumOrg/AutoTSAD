#  This function is adapted from [UOMS] by [yzhao062]
#  Original source: [https://github.com/yzhao062/UOMS]

import numpy as np
from sklearn import metrics
from s_dbw import S_Dbw

def xb(x, c, c_normal, c_anomaly):
    return sum((x-c)**2) / (len(x) * ((c_normal - c_anomaly) ** 2))

def hubert(x, c):
    n = len(x)
    x = np.array(x)     # (t, )
    c = np.array(c)
    
    diff = abs(x - x[:,None])   # (t, t)
    diff_c = abs(c - c[:,None])

    mask = np.zeros_like(diff, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True

    return ((mask * diff) * (diff_c) * 2).sum() / (n * (n-1))

def i_index(x, c, c_normal, c_anomaly):
    max_d = abs(c_normal - c_anomaly)
    n = len(x)
    x = np.array(x)
    c = np.array(c)
    numerator = (abs(x-x.mean()).sum())
    denominator = (abs(x-c).sum())
    return ((numerator * max_d) / (denominator * 2)) ** 2

def Dunn(x, c_binary):
    x = np.array(x)
    c_binary = np.array(c_binary)
    normal_mask = np.tile(c_binary, (len(c_binary), 1))
    anomaly_mask = np.tile(1 - c_binary.reshape(-1, 1).T, (len(c_binary), 1))
    distance = abs(x - x[:,None])
    
    diff_inter = distance * normal_mask * (anomaly_mask.T) # distance of samples from different clusters
    diff_intra_1 = distance * normal_mask * (normal_mask.T) # distance of samples in cluster 1
    diff_intra_2 = distance * anomaly_mask * (anomaly_mask.T) # distance of samples in cluster 2

    min_value = np.min(diff_inter[np.nonzero(diff_inter)])
    max_value = max(np.max(diff_intra_1), np.max(diff_intra_2))
    return min_value/max_value

def cluster_metrics_xbs(unique_scores):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_normal = np.mean(x[(x<threshold)])
            c_anomaly = np.mean(x[(x>threshold)])
            c = [c_anomaly if i >= threshold else c_normal for i in x]
            xbs = xb(x, c, c_normal, c_anomaly)
            break
        cluster_list.append(xbs)
    return cluster_list

def cluster_metrics_std(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            n = len(x)
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_normal = np.mean(x[(x<threshold)])
            c_anomaly = np.mean(x[(x>threshold)])
            c = [c_anomaly if i >= threshold else c_normal for i in x]
            std = np.sqrt(metrics.mean_squared_error(y_true=x, y_pred=c) * (n / ((n-2) * slidingWindow)))
            break
        cluster_list.append(std)
    return cluster_list

def cluster_metrics_r2(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            n = len(x)
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_normal = np.mean(x[(x<threshold)])
            c_anomaly = np.mean(x[(x>threshold)])
            c = [c_anomaly if i >= threshold else c_normal for i in x]
            r2 = metrics.r2_score(y_true=x, y_pred=c) 
            break
        cluster_list.append(r2)
    return cluster_list

def cluster_metrics_hubert(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):        
        for anomaly_std in threshold_range:
            x = score_ds[id]
            n = len(x)
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_normal = np.mean(x[(x<threshold)])
            c_anomaly = np.mean(x[(x>threshold)])
            c = [c_anomaly if i >= threshold else c_normal for i in x]
            h = hubert(x, c) 
            break
        cluster_list.append(h)
    return cluster_list

def cluster_metrics_ch(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_binary = [1 if i >= threshold else 0 for i in x]
            if len(np.unique(c_binary)) != 2: continue
            ch = metrics.calinski_harabasz_score(X=x.reshape(len(x), 1), labels=c_binary) 
            break   
        cluster_list.append(ch)
    return cluster_list

def cluster_metrics_silhouette(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            n = len(x)
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_binary = [1 if i >= threshold else 0 for i in x]
            if len(np.unique(c_binary)) != 2: continue
            s = metrics.silhouette_score(X=x.reshape(len(x), 1), labels=c_binary) 
            break
        cluster_list.append(s)
    return cluster_list

def cluster_metrics_i_index(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        
        for anomaly_std in threshold_range:
            x = score_ds[id]
            n = len(x)
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_normal = np.mean(x[(x<threshold)])
            c_anomaly = np.mean(x[(x>threshold)])
            c = [c_anomaly if i >= threshold else c_normal for i in x]
            i = i_index(x, c, c_normal, c_anomaly) 
            break
        cluster_list.append(i)
    return cluster_list

def cluster_metrics_db(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as): 
        for anomaly_std in threshold_range:
            x = score_ds[id]
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_binary = [1 if i >= threshold else 0 for i in x]
            if len(np.unique(c_binary)) != 2: continue
            db = metrics.davies_bouldin_score(X=x.reshape(len(x), 1), labels=c_binary) 
            break
        cluster_list.append(db)
    return cluster_list

def cluster_metrics_sd(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_binary = [1 if i >= threshold else 0 for i in x]
            if len(np.unique(c_binary)) != 2: continue
            sd = S_Dbw(X=x.reshape(len(x), 1), labels=c_binary) # need to double check
            break
        cluster_list.append(sd)
    return cluster_list

def cluster_metrics_dunn(unique_scores, slidingWindow):
    scores = unique_scores.T
    downsampling = 10
    n_as, n_t = scores.shape
    right_padding = downsampling - n_t%downsampling
    scores_pad = np.pad(scores, ((0,0), (right_padding, 0) ))
    score_ds = scores_pad.reshape(n_as, scores_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    cluster_list = []
    threshold_range=[3, 2, 1, 0]
    for id in range(n_as):
        for anomaly_std in threshold_range:
            x = score_ds[id]
            threshold = np.mean(x) + anomaly_std*np.std(x)
            if sum(x>threshold) == 0: continue
            c_binary = [1 if i >= threshold else 0 for i in x]
            if len(np.unique(c_binary)) != 2: continue
            dunn = Dunn(x, c_binary) 
            break
        cluster_list.append(dunn)
    return cluster_list