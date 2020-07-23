import numpy as np
import gc
'''
Define a distance metric using trained  𝛼 ,  𝛾  and  𝜖
'''
def theta(x, y):
    re = y * np.tanh(x / y)
    return re

def cost(path_x, path_y, ts_a, ts_b, alpha, gamma, epsilon):
    #i, i_, j, j_ 都是从1开始的索引，要更正过来
    i, j = path_x
    i_, j_ = path_y    
    i -= 1;i_ -= 1;j -= 1;j_ -= 1
    
    ta_i = ts_a[i]
    ta_i_ = ts_a[i_]
    tb_j = ts_b[j]
    tb_j_ = ts_b[j_]
    
    if i_ > i and j_ > j:
        re = theta(np.linalg.norm(ta_i_ - tb_j_), (epsilon/(1-epsilon)))
    elif i_ == i or j_ == j:
        re = (alpha/(1-alpha)) * theta(np.linalg.norm(ta_i_ - tb_j_), (epsilon/(1-epsilon))) + gamma
    else:
        print('Wrong definition of alignment path')
        raise AssertionError   
    return re

class Trained_distance:
    def __init__(self, len_ts_a, len_ts_b, alpha, gamma, epsilon):
        #在第一次调用
        self.len_a = len_ts_a
        self.len_b = len_ts_b
        
        #init params
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon

    
    def caclulate_distance(self, ts_a, ts_b):
        dis_matrix = []
        dis_matrix.append([0]*(self.len_b+1))#第0行均为0
        #dynamic programming
        for ii in range(1, self.len_a+1):
            dis_matrix.append([])#加一行，这一行的索引为ii
            dis_matrix[ii].append(0)#这一行的第一列置为0
                
            for jj in range(1, self.len_b+1):
                dis_matrix[ii].append(np.min([dis_matrix[ii-1][jj]+cost([ii-1,jj],[ii,jj],ts_a,ts_b,self.alpha,self.gamma,self.epsilon),\
                                                     dis_matrix[ii][jj-1]+cost([ii,jj-1],[ii,jj],ts_a,ts_b,self.alpha,self.gamma,self.epsilon),\
                                                     dis_matrix[ii-1][jj-1]+cost([ii-1,jj-1],[ii,jj],ts_a,ts_b,self.alpha,self.gamma,self.epsilon)]))
                
        #至此，通过dynamic programming计算了距离矩阵 
        dist = dis_matrix[self.len_a][self.len_b]
        del(dis_matrix)
        gc.collect() 
        return dist 