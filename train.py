import tensorflow as tf
import os
import numpy as np
from utils import read_dataset_cluster
from autowarp_tf import Autowarp

print(tf.__version__)
print('IF GPU AVAILABLE: ', tf.test.is_gpu_available())

'''
hyper parameters
'''
DATASET_INDEX = 0 #the index of a dataset
LATENT_SPACE_DIM = 12

USE_TRAINED_MODEL = False
IF_NORMALIZE = True


'''
Load data and build up a complete algorithm
'''
configs = {}

data_path = 'UCRArchive_2018'
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
dataset_name = file_list[DATASET_INDEX]
print(dataset_name)


configs['train_file'] = train_list[DATASET_INDEX]
configs['test_file'] = test_list[DATASET_INDEX]    

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
Autowarp
Set parameters
'''
train_data = np.expand_dims(data, axis=2)#[b,t,1]
test_data = np.expand_dims(t_data, axis=2)#[b,t,1]


#warning: batchsize should be smaller than the number of train data 
#warning: batchsize should be smaller than the number of item < delta 
#__init__(self, train_data, percentile, lr_rate, batchsize, hidden_units, max_iter)
autowarp = Autowarp(train_data=train_data, percentile=0.2, lr_rate=1e-2, batchsize=32, hidden_units=LATENT_SPACE_DIM, max_iter=50)

'''
Seq2seq model
Use a sequence-to-sequence autoencoder trained to minimize the reconstruction loss of the trajectories to learn a latent representation  ℎ𝑖  for each trajectory  𝐭𝑖 .
'''
#init a model
if USE_TRAINED_MODEL == False:
    recons_model, latent_model = autowarp.init_network()
    trainsed_recons_model, trained_latent_model = autowarp.train_network()
    if os.path.exists('./trained_model/'+dataset_name) == False: os.makedirs('./trained_model/'+dataset_name)    
    trainsed_recons_model.save('./trained_model/'+dataset_name+'/recons_model.h5')
    trained_latent_model.save('./trained_model/'+dataset_name+'/latent_model.h5')
else:
    trained_latent_model = tf.keras.models.load_model('./trained_model/'+dataset_name+'/latent_model.h5')

#get latent representation of test and train data
h_train = autowarp.get_latent_vectors(trained_latent_model, train_data)
h_test = autowarp.get_latent_vectors(trained_latent_model, test_data)


'''
Compute distance matrix
Compute the pairwise Euclidean distance matrix between each pair of latent representations
'''
train_matrix = autowarp.distance_matrix(h_train)


'''
Define a delta
Compute the threshold distance  𝛿  defined as the  𝑝𝑡ℎ  percentile of the distribution of distances
'''
delta = autowarp.init_delta()


'''
Initialize the parameters
Initialize the parameters  𝛼 ,  𝛾 ,  𝜖 , (e.g. randomly between 0 and 1)
'''
#Have initialized
print(autowarp.alpha)
print(autowarp.gamma)
print(autowarp.epsilon)

'''
Start optimize parameters:  𝛼 ,  𝛾  and  𝜖
'''
best_beta, best_alpha, best_gamma, best_epsilon = autowarp.train_warping_family()
print('best_beta = {}\nbest_alpha = {}\nbest_gamma = {}\nbest_epsilon = {}'.format(best_beta, best_alpha, best_gamma, best_epsilon))
