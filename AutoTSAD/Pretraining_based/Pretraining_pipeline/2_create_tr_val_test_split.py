import os
import pandas as pd
import math
import numpy as np
import random
random.seed(2023)

Dataset_Pool = ['NAB', 'YAHOO', 'SensorScope', 'NASA-MSL', 'NASA-SMAP', 'Daphnet', 'SMD', 'KDD21', 'MITDB', 'Genesis', 'IOPS', 'Dodgers',
                'MGAB', 'GHL', 'Occupancy', 'SVDB', 'OPPORTUNITY', 'ECG']
select_file_list = pd.read_csv('file_list/all_file_list.csv')

## In-domain list generation
train_id = []
val_id = []
file_names = select_file_list.values.tolist()
random.shuffle(file_names)
length = len(file_names)
split_1 = int(length * 0.8)
train_id.extend(file_names[:split_1])
val_id.extend(file_names[split_1:])
train_id_pd = pd.DataFrame(train_id, columns=['Dataset', 'File_name'])
val_id_pd = pd.DataFrame(val_id, columns=['Dataset', 'File_name'])
train_id_pd.to_csv('data/split_list/pretrained/train_list.csv', index=False)
val_id_pd.to_csv('data/split_list/pretrained/val_list.csv', index=False)


## Out-of-domain list generation
for dataset in Dataset_Pool:
    train_ood = []
    val_ood = []
    
    data_file_names = select_file_list[select_file_list['Dataset']!=dataset].values.tolist()
    random.shuffle(data_file_names)
    length = len(data_file_names)
    split_1 = int(length * 0.8)
    train_ood.extend(data_file_names[:split_1])
    val_ood.extend(data_file_names[split_1:])
    train_ood_pd = pd.DataFrame(train_ood, columns=['Dataset', 'File_name'])
    val_ood_pd = pd.DataFrame(val_ood, columns=['Dataset', 'File_name'])

    os.makedirs(f'data/split_list/ood/{dataset}', exist_ok=True)
    train_ood_pd.to_csv(f'data/split_list/ood/{dataset}/train_list.csv', index=False)
    val_ood_pd.to_csv(f'data/split_list/ood/{dataset}/val_list.csv', index=False)