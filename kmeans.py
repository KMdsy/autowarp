import os
import numpy as np
import gc
from sklearn.metrics import normalized_mutual_info_score as nmi
from autowarp_np import Trained_distance
'''
Time series clustering
K-means + learned distance metric
'''
class K_means_learned:
    def __init__(self, num_cluster, random_state=None, max_iter=300):
        self.num_cluster = num_cluster
        self.random_seed = random_state
        self.max_iter = max_iter

    def set_my_metric(self,  len_ts_a, len_ts_b, alpha, gamma, epsilon):
        self.my_distance_metric = Trained_distance(len_ts_a, len_ts_b, alpha, gamma, epsilon)
        
    def fit(self, train_data):
        #set a seed
        if type(train_data) != type([]):
            train_data = list(train_data)
            
        import random
        if self.random_seed is not None:
            random.seed(self.random_seed)
        centroids=random.sample(train_data, self.num_cluster)
        
        for n in range(self.max_iter):
            print('iter num: {} of {}'.format(n, self.max_iter-1))
            assignments={}
            #assign data points to clusters
            for idx, data in enumerate(train_data):
                #print('\ttraining sample: {} of {}'.format(idx, len(train_data)-1))
                min_dist=float('inf')
                closest_clust=None
                #分别计算样本idx距离几个中心的距离
                for c_idx, center in enumerate(centroids):
                    cur_dist = self.my_distance_metric.caclulate_distance(data, center)
                    #cur_dist = edr_dist_py(data, center, epsilon=0.5)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_idx
                #将样本idx分配到某个簇中
                #assignments, key = center_idx, value = a list of sample index
                if closest_clust not in assignments:
                    assignments[closest_clust] = []
                assignments[closest_clust].append(idx)                    

            #至此，样本idx被分配到一个簇中，recalculate centroids of clusters
            for center_idx in assignments:#迭代key
                cluster_sum = 0
                for k in assignments[center_idx]:#迭代簇中的样本
                    cluster_sum += train_data[k]
                centroids[center_idx] = [m / len(assignments[center_idx]) for m in cluster_sum]
        self.centroids = centroids
        print('centroids: ')#shape: [n_cluster, data_dim]
        for c_idx in range(self.num_cluster):
            print('center of {}^th cluster: {}'.format(c_idx, centroids[c_idx]))
        print('shape of centroids: ', np.array(self.centroids).shape)
        print('\n#################################\n\n')
        return np.array(self.centroids)
        
    def predict(self, test_data):
        if type(test_data) != type([]):
            test_data = list(test_data)
        pred_label = []
        for idx, data in enumerate(test_data):
            #print('\ttest sample: {} of {}'.format(idx, len(test_data)))
            min_dist=float('inf')
            closest_clust=None
            for c_idx, center in enumerate(self.centroids):
                cur_dist = self.my_distance_metric.caclulate_distance(data, center)
                #cur_dist = edr_dist_py(data, center, epsilon=0.5)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_idx
            #至此，样本已经被分配到一个簇
            pred_label.append(closest_clust)
        pred_label = np.array(pred_label)
        return pred_label
    
    def fit_and_test_every_epoch(self, train_data, test_data, test_label):
        print('START K-MEANS')
        #性能下降的次数
        decrease_num = 0
        nmi_old = None
        #set a seed
        if type(train_data) != type([]):
            train_data = list(train_data)
            
        import random
        if self.random_seed is not None:
            random.seed(self.random_seed)
        centroids=random.sample(train_data, self.num_cluster)
        
        best_nmi = 0
        best_epoch = None
        for n in range(self.max_iter):
            print('iter num: {} of {}'.format(n, self.max_iter - 1))
            assignments={}
            #assign data points to clusters
            for idx, data in enumerate(train_data):
                #print('\ttraining sample: {} of {}'.format(idx, len(train_data) - 1))
                min_dist=float('inf')
                closest_clust=None
                #分别计算样本idx距离几个中心的距离
                for c_idx, center in enumerate(centroids):
                    cur_dist = self.my_distance_metric.caclulate_distance(data, center)
                    #cur_dist = edr_dist_py(data, center, epsilon=0.5)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_idx
                #将样本idx分配到某个簇中
                #assignments, key = center_idx, value = a list of sample index
                if closest_clust not in assignments:
                    assignments[closest_clust] = []
                assignments[closest_clust].append(idx)                    

            #至此，样本idx被分配到一个簇中，recalculate centroids of clusters
            for center_idx in assignments:#迭代key
                cluster_sum = 0
                for k in assignments[center_idx]:#迭代簇中的样本
                    cluster_sum += train_data[k]
                centroids[center_idx] = [m / len(assignments[center_idx]) for m in cluster_sum]
            self.centroids = centroids   
            
            
            #至此，一轮结束
            labels = self.predict(test_data)
            nmi_ = nmi(test_label, labels, average_method='arithmetic')
            print('epoch: {}, nmi: {}, decrease_num: {}'.format(n, nmi_, decrease_num))
            
            if nmi_old is None:
                nmi_old = nmi_
            else:
                #如果当前的性能小于上一次的，下降次数加1
                if nmi_ <= nmi_old:
                    decrease_num += 1
                #如果当前的性能大于等于上一次的，清零下降次数
                else:
                    decrease_num = 0
                nmi_old = nmi_
                #如果有五次下降，则停止迭代
                if decrease_num > 5:
                    break
            
            if nmi_ > best_nmi:
                best_nmi = nmi_
                best_epoch = n
        
        print('best_nmi_kmeans: ', best_nmi, 'epoch:', best_epoch)