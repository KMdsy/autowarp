import os
import numpy as np
import os
from utils import read_dataset_cluster

from kmeans import K_means_learned


# set the parameters of autowarp, which are learned from train dataset
best_alpha = 0.95098734 # config
best_gamma = 0.64394510 # config
best_epsilon = 0.64841467 # config


# dataset config
DATASET_IDX = 1 # the index of a dataset 
IF_NORMALIZE = True


print('autowarp params: \na = {}\ng = {}\ne = {}'.format(best_alpha, best_gamma, best_epsilon))   
    
'''
Load data and build up a complete algorithm
'''
configs = {}

data_path = './../UCR_dataset/UCRArchive_2018' #config
file_list = os.listdir(data_path)
file_list.sort()

train_list = []
test_list = []

idx = 0
for name in file_list:
    train_file = data_path+'/'+name+'/'+name+'_TRAIN.tsv'
    test_file = data_path+'/'+name+'/'+name+'_TEST.tsv'

    train_list.append(train_file)
    test_list.append(test_file)
    
    print('idx: {}, name: {}'.format(idx, name))
    idx += 1  
print('dataset_num: {0}'.format(len(file_list)))

'''
Input the data
'''
dataset_name = file_list[DATASET_IDX]
print(dataset_name)


configs['train_file'] = train_list[DATASET_IDX]
configs['test_file'] = test_list[DATASET_IDX]    

data, label, label_dict = read_dataset_cluster(configs, 'train')
t_data, t_label, _ = read_dataset_cluster(configs, 'test', label_dict) 


print('shape of train', data.shape)
print('shape of test', t_data.shape)
print('label class: ', np.unique(t_label))

if IF_NORMALIZE:
    def normalize(seq):
        return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1

    for i in range(data.shape[0]):
        data[i] = normalize(data[i])

    for i in range(t_data.shape[0]):
        t_data[i] = normalize(t_data[i])    
    

'''
Clustering using kmeans
'''     
kmeans_model = K_means_learned(len(np.unique(label)), random_state = 2020, max_iter = 300)

'''
define a distance metric
'''
kmeans_model.set_my_metric(t_data.shape[1], t_data.shape[1], best_alpha, best_gamma, best_epsilon)

'''
fit
'''
centroids = kmeans_model.fit_and_test_every_epoch(data, t_data, t_label)
print('dataset', configs['train_file'])

print('GT',t_label[l:h])
print('PRED',labels_kmeans[l:h])
print('#######################################')