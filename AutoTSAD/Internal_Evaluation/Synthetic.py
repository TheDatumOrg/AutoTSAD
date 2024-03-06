import numpy as np
from .inject_anomalies import InjectAnomalies
from .inject_anomalies import gen_synthetic_performance_list
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.seasonal import STL

def simulated_ts_stl(data, slidingWindow):
    stl = STL(data, period=slidingWindow)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    return seasonal + np.random.normal(np.mean(resid),np.std(resid),len(data))
    # return trend + seasonal


def synthetic_anomaly_injection(data, Det_pool):
    data = data[:50000]
    ANOMALY_TYPES = list(ANOMALY_PARAM_GRID.keys())

    if len(data) > 5000:
        T_original = data.reshape(1, -1)
        downsampling = 10
        n_features, n_t = T_original.shape
        right_padding = downsampling - n_t%downsampling
        T_pad = np.pad(T_original, ((0,0), (right_padding, 0) ))
        T = T_pad.reshape(n_features, T_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    else:
        T = data.reshape(1, -1)

    data_std = max(np.std(T), 0.01)
    synthetic_time_series = []
    synthetic_anomaly_labels = []

    for anomaly_type in ANOMALY_TYPES:
        anomaly_obj = InjectAnomalies(random_state=0,
                                    verbose=False,
                                    max_window_size=128,
                                    min_window_size=8)
        for anomaly_params in list(
                ParameterGrid(ANOMALY_PARAM_GRID[anomaly_type])):
            anomaly_params['T'] = T
            anomaly_params['scale'] = anomaly_params['scale'] * data_std
            anomaly_type = anomaly_params['anomaly_type']
            # print('anomaly_params: ', anomaly_params)

            # Inject synthetic anomalies to the data
            T_a, anomaly_sizes, anomaly_labels = anomaly_obj.inject_anomalies(**anomaly_params)
            ts = T_a.ravel()
            
            synthetic_time_series.append(ts)
            synthetic_anomaly_labels.append(anomaly_labels.ravel())
            # print("Number of abnormal points: ",list(y).count(1))
            # print('y.shape: ', y.shape)
            # print('anomaly_labels.shape: ', anomaly_labels.shape)
            # print("Number of synthetic abnormal points: ",list(anomaly_labels.ravel()).count(1))

    print('len(synthetic_anomaly_labels): ', len(synthetic_anomaly_labels))
    synthetic_performance_list = []
    for i in range(len(synthetic_anomaly_labels)):
        data_i = synthetic_time_series[i]
        label_i = synthetic_anomaly_labels[i]
        synthetic_performance_i = gen_synthetic_performance_list(data_i, label_i, Det_pool)
        synthetic_performance_list.append(synthetic_performance_i)
    
    synthetic_performance = np.array(synthetic_performance_list).astype(np.float32)
    # selected_model_id = np.argmax(np.mean(synthetic_performance, axis=0))
    synthetic_performance_list = np.mean(synthetic_performance, axis=0)
    return synthetic_performance_list



def synthetic_anomaly_injection_type(data, Det_pool, anomaly_type):
    data = data[:50000]

    # if len(data) > 2560:
    #     T_original = data.reshape(1, -1)
    #     downsampling = 10
    #     n_features, n_t = T_original.shape
    #     right_padding = downsampling - n_t%downsampling
    #     T_pad = np.pad(T_original, ((0,0), (right_padding, 0) ))
    #     T = T_pad.reshape(n_features, T_pad.shape[-1]//downsampling, downsampling).max(axis=2)
    # else:
    #     T = data.reshape(1, -1)

    T_original = data.reshape(1, -1)
    downsampling = 10
    n_features, n_t = T_original.shape
    right_padding = downsampling - n_t%downsampling
    T_pad = np.pad(T_original, ((0,0), (right_padding, 0) ))
    T = T_pad.reshape(n_features, T_pad.shape[-1]//downsampling, downsampling).max(axis=2)


    data_std = max(np.std(T), 0.01)
    synthetic_time_series = []
    synthetic_anomaly_labels = []

    anomaly_obj = InjectAnomalies(random_state=0,
                                verbose=False,
                                max_window_size=128,
                                min_window_size=8)
    for anomaly_params in list(
            ParameterGrid(ANOMALY_PARAM_GRID[anomaly_type])):
        anomaly_params['T'] = T
        anomaly_params['scale'] = anomaly_params['scale'] * data_std
        anomaly_params['anomaly_type'] = anomaly_type
        # print('anomaly_params: ', anomaly_params)

        # Inject synthetic anomalies to the data
        inject_label = True
        try:
            T_a, anomaly_sizes, anomaly_labels = anomaly_obj.inject_anomalies(**anomaly_params)
            ts = T_a.ravel()
            
            synthetic_time_series.append(ts)
            synthetic_anomaly_labels.append(anomaly_labels.ravel())
        except:
            inject_label = False
            print('Error while injecting anomaly')

    synthetic_performance_list = []
    if inject_label:
        for i in range(len(synthetic_anomaly_labels)):
            data_i = synthetic_time_series[i]
            label_i = synthetic_anomaly_labels[i]
            try:
                synthetic_performance_i = gen_synthetic_performance_list(data_i, label_i, Det_pool)
            except:
                print('Error while generating synthetic performace list')
                synthetic_performance_i = [0]*len(Det_pool)
            synthetic_performance_list.append(synthetic_performance_i)
    else:
        synthetic_performance_list.append([0]*len(Det_pool))

    synthetic_performance = np.array(synthetic_performance_list).astype(np.float32)
    synthetic_performance_list = np.mean(synthetic_performance, axis=0)
    return synthetic_performance_list



ANOMALY_PARAM_GRID = {
    'spikes': {
        'anomaly_type': ['spikes'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
        'anomaly_propensity': [0.5],
    },
    'contextual': {
        'anomaly_type': ['contextual'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'flip': {
        'anomaly_type': ['flip'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'scale': [2],
    },
    'speedup': {
        'anomaly_type': ['speedup'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        # 'speed': [0.25, 0.5, 2, 4],
        'speed': [0.25],
        'scale': [2],
    },
    'noise': {
        'anomaly_type': ['noise'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'noise_std': [0.05],
        'scale': [2],
    },
    'cutoff': {
        'anomaly_type': ['cutoff'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        # 'constant_type': ['noisy_0', 'noisy_1'],
        'constant_type': ['noisy_0'],
        'constant_quantile': [0.75],
        'scale': [2],
    },
    'scale': {
        'anomaly_type': ['scale'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        # 'amplitude_scaling': [0.25, 0.5, 2, 4],
        'amplitude_scaling': [0.25],
        'scale': [2],
    },
    'wander': {
        'anomaly_type': ['wander'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'baseline': [-0.3, -0.1, 0.1, 0.3],
        'scale': [2],
    },
    'average': {
        'anomaly_type': ['average'],
        'random_parameters': [False],
        'max_anomaly_length': [4],
        'anomaly_size_type': ['mae'],
        'feature_id': [None],
        'correlation_scaling': [5],
        'ma_window': [4, 8],
        'scale': [2],
    }
}