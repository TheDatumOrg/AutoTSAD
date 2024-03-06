from TSB_UAD.TSB_run_det import *
from TSB_UAD.TSB_run_det_unif import *

def gen_as_from_det(data, det, args):
    det_name = det.split('_')[0]
    det_hp = det.split('_')[1:]
    if det_name == 'IForest':
        score = run_iforest_dev(data, periodicity=int(det_hp[0]), n_estimators=int(det_hp[1]), n_jobs=args.jobs)
    elif det_name == 'LOF':
        score = run_lof_dev(data, periodicity=int(det_hp[0]), n_neighbors=int(det_hp[1]), n_jobs=args.jobs)
    elif det_name == 'MP':
        score = run_matrix_profile_dev(data, periodicity=int(det_hp[0]), cross_correlation=bool(det_hp[1]), n_jobs=args.jobs)
    elif det_name == 'PCA':
        det_hp[1]=None if det_hp[1] == 'None' else float(det_hp[1])
        score = run_pca_dev(data, periodicity=int(det_hp[0]), n_components=det_hp[1], n_jobs=args.jobs)
    elif det_name == 'NORMA':
        score = run_norma_dev(data, periodicity=int(det_hp[0]), clustering=det_hp[1], n_jobs=args.jobs)
    elif det_name == 'HBOS':
        score = run_hbos_dev(data, periodicity=int(det_hp[0]), n_bins=int(det_hp[1]), n_jobs=args.jobs)
    elif det_name == 'POLY':
        score = run_poly_dev(data, periodicity=int(det_hp[0]), power=int(det_hp[1]), n_jobs=args.jobs)
    elif det_name == 'OCSVM':
        score = run_ocsvm_dev(data, periodicity=int(det_hp[0]), kernel=det_hp[1], n_jobs=args.jobs)
    elif det_name == 'AE':
        hidden_neurons_list = [[64, 32, 32, 64], [32, 16, 32]]
        score = run_ae_dev(data, periodicity=int(det_hp[0]), hidden_neurons=hidden_neurons_list[int(det_hp[1])], output_activation='relu', norm=det_hp[2], n_jobs=args.jobs)
    elif det_name == 'CNN':
        num_channel_list = [[32, 32, 40], [32, 64, 64]]
        score = run_cnn_dev(data, periodicity=int(det_hp[0]), num_channel=num_channel_list[int(det_hp[1])], activation=det_hp[2], n_jobs=args.jobs)
    elif det_name == 'LSTM':
        hidden_dim_list = [32, 64]
        score = run_lstm_dev(data, periodicity=int(det_hp[0]), hidden_dim=hidden_dim_list[int(det_hp[1])], activation=det_hp[2], n_jobs=args.jobs)
    return score

def gen_as_unif_from_det(data, det, args):
    n_generated = 100000
    data_ts = data.reshape(-1, 1)
    n_features = data_ts.shape[1]
    lim_inf = data_ts.min(axis=0)
    lim_sup = data_ts.max(axis=0)
    unif = np.random.uniform(lim_inf, lim_sup,
                            size=(n_generated, n_features)).ravel()
    for det in det:
        det_name = det.split('_')[0]
        det_hp = det.split('_')[1:]
        if det_name == 'IForest':
            score = run_iforest_unif(data, unif, periodicity=int(det_hp[0]), n_estimators=int(det_hp[1]))
        elif det_name == 'LOF':
            score = run_lof_unif(data, unif, periodicity=int(det_hp[0]), n_neighbors=int(det_hp[1]))
        elif det_name == 'MP':
            score = run_matrix_profile_unif(data, unif, periodicity=int(det_hp[0]), cross_correlation=bool(det_hp[1]))
        elif det_name == 'PCA':
            det_hp[1]=None if det_hp[1] == 'None' else float(det_hp[1])
            score = run_pca_unif(data, unif, periodicity=int(det_hp[0]), n_components=det_hp[1])
        elif det_name == 'NORMA':
            score = run_norma_unif(data, unif, periodicity=int(det_hp[0]), clustering=det_hp[1])
        elif det_name == 'HBOS':
            score = run_hbos_unif(data, unif, periodicity=int(det_hp[0]), n_bins=int(det_hp[1]))
        elif det_name == 'POLY':
            score = run_poly_unif(data, unif, periodicity=int(det_hp[0]), power=int(det_hp[1]))
        elif det_name == 'OCSVM':
            score = run_ocsvm_unif(data, unif, periodicity=int(det_hp[0]), kernel=det_hp[1])
        elif det_name == 'AE':
            hidden_neurons_list = [[64, 32, 32, 64], [32, 16, 32]]
            score = run_ae_unif(data, unif, periodicity=int(det_hp[0]), hidden_neurons=hidden_neurons_list[int(det_hp[1])], output_activation='relu', norm=det_hp[2])
        elif det_name == 'CNN':
            num_channel_list = [[32, 32, 40], [32, 64, 64]]
            score = run_cnn_unif(data, unif, periodicity=int(det_hp[0]), num_channel=num_channel_list[int(det_hp[1])], activation=det_hp[2])
        elif det_name == 'LSTM':
            hidden_dim_list = [32, 64]
            score = run_lstm_unif(data, unif, periodicity=int(det_hp[0]), hidden_dim=hidden_dim_list[int(det_hp[1])], activation=det_hp[2])
    return score

def gen_as_from_set(data, Candidate_Model_Set, args):
    scores_list = []
    for det in Candidate_Model_Set:
        score = gen_as_from_det(data, det, args)
        scores_list.append(score)
    det_scores = np.array(scores_list).T
    return det_scores

def gen_as_unif_from_set(data, Candidate_Model_Set, args):
    scores_list = []
    for det in Candidate_Model_Set:
        score = gen_as_unif_from_det(data, det, args)
        scores_list.append(score)
    det_scores = np.array(scores_list).T
    return det_scores

def get_preds_ratio(scores, outliers_ratio = 0.05):
    num = int(len(scores)*outliers_ratio)
    threshold = np.sort(scores)[::-1][num]
    predictions = np.array(scores > threshold)
    predictions = np.array([int(i) for i in predictions])      
    return predictions

def gen_autood_initial_set(data, Candidate_Model_Set, args):

    threshold_ratio_range = [0.02, 0.04, 0.08, 0.10, 0.15, 0.20]
    threshold_std_range = [1, 2, 3]

    all_scores = []
    unique_score = []
    all_preds = []

    for det in Candidate_Model_Set:
        score = gen_as_from_det(data, det, args)
        unique_score.append(score)
        for threds in threshold_ratio_range:
            all_preds.append(get_preds_ratio(score, threds))
            all_scores.append(score)

    preds = np.stack(all_preds).T
    scores = np.stack(all_scores).T
    unique_score = np.stack(unique_score).T

    # print('preds shape:', np.shape(preds))
    # print('unique_score shape:', np.shape(unique_score))    # [ts_len, num_det]

    # For Candidate_Model_Set_Full
    # TODO: Needs to adjust when changing candidate model set
    instance_index_ranges = [[0, 18], [18, 30], [30, 42], [42, 54], [54, 66], [66, 78], [78, 90], [90, 102], [102, 114], [114, 126], [126, 138]]
    detector_index_ranges = [[0, 3], [3, 5], [5, 7], [7, 9], [9, 11], [11, 13], [13, 15], [15, 17], [17, 19], [19, 21], [21, 23]]

    return preds, scores, unique_score, instance_index_ranges, detector_index_ranges

def split_ts(data, window_size):
    # Compute the modulo
    modulo = data.shape[0] % window_size

    # Compute the number of windows
    k = data[modulo:].shape[0] / window_size
    assert(math.ceil(k) == k)

    # Split the timeserie
    data_split = np.split(data[modulo:], k)
    if modulo != 0:
        data_split.insert(0, list(data[:window_size]))
    data_split = np.asarray(data_split)

    return data_split