import tensorflow as tf
import os
import numpy as np
import gc
from scipy.spatial.distance import pdist, squareform

'''
(1)下面的theta定义
'''
def my_theta(x, y):
    re = y * tf.nn.tanh(x / y)
    return re

'''
(1)
'''
def my_cost(path_x, path_y, ts_a, ts_b, alpha, gamma, epsilon):
    #path_x: (i, j), 分别代表前一步的两个时间序列对齐关系
    #path_y: (i', j'), 分别代表后一步的两个时间序列对齐关系
    #path: (ts_a^i, ts_b^j)
    
    #error; 不能使用i, j = path_x的形式，因为tensor不能迭代，无法转换为图
    
    #i, i_, j, j_ 都是从1开始的索引，要更正过来
    i, j = path_x[0], path_x[1]
    i_, j_ = path_y[0], path_y[1]
    
    i -= 1;i_ -= 1;j -= 1;j_ -= 1
    
    ta_i = ts_a[i]
    ta_i_ = ts_a[i_]
    tb_j = ts_b[j]
    tb_j_ = ts_b[j_]
    
    if i_ > i and j_ > j:
        re = my_theta(tf.norm(ta_i_ - tb_j_), (epsilon/(1-epsilon)))
    elif i_ == i or j_ == j:
        re = (alpha/(1-alpha)) * my_theta(tf.norm(ta_i_ - tb_j_), (epsilon/(1-epsilon))) + gamma
    else:
        re = tf.constant(0, dtype=tf.float32)
                
    return re

class Distance:
    def __init__(self, len_ts_a, len_ts_b, alpha, gamma, epsilon):
        #在第一次调用
        self.len_a = len_ts_a
        self.len_b = len_ts_b
        
        #init params
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        
    def update_params(self, alpha, gamma, epsilon):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
    

    def caclulate_distance(self, ts_a, ts_b):
        dis_matrix = []
        dis_matrix.append([0]*(self.len_b+1))#第0行均为0
        #dynamic programming
        for ii in range(1, self.len_a+1):
            dis_matrix.append([])#加一行，这一行的索引为ii
            dis_matrix[ii].append(0)#这一行的第一列置为0

            for jj in range(1, self.len_b+1):
                dis_matrix[ii].append(tf.reduce_min([dis_matrix[ii-1][jj]+my_cost([ii-1,jj],[ii,jj],ts_a,ts_b,self.alpha,self.gamma,self.epsilon),\
                                                     dis_matrix[ii][jj-1]+my_cost([ii,jj-1],[ii,jj],ts_a,ts_b,self.alpha,self.gamma,self.epsilon),\
                                                     dis_matrix[ii-1][jj-1]+my_cost([ii-1,jj-1],[ii,jj],ts_a,ts_b,self.alpha,self.gamma,self.epsilon)]))
                
                
        #至此，通过dynamic programming计算了距离矩阵 
        dist = dis_matrix[self.len_a][self.len_b]
        del(dis_matrix)
        gc.collect()
        return dist 

class Autowarp:
    def __init__(self, train_data, percentile, lr_rate, batchsize, hidden_units, max_iter):
        self.train_data = train_data
        self.percentile = percentile
        self.lr_rate = lr_rate
        self.batchsize = batchsize#是算法采样的batchsize，不是用于训练神经网络的
        self.hidden_units = hidden_units
        self.max_iter = max_iter
        self.max_len = train_data.shape[1]
        print('max_len: ', self.max_len)
        
        '''
        Initialize the parameters
        '''
        self.alpha = tf.Variable(tf.random.uniform(shape=[], minval=0., maxval=1.), name='alpha')
        self.gamma = tf.Variable(tf.random.uniform(shape=[], minval=0., maxval=1.), name='gamma')
        self.epsilon = tf.Variable(tf.random.uniform(shape=[], minval=0., maxval=1.), name='epsilon')
        #len_ts_a, len_ts_b, alpha, gamma, epsilon
        distance = Distance(len_ts_a=self.train_data.shape[1],#等长的时间序列对比
                            len_ts_b=self.train_data.shape[1],
                            alpha=self.alpha,
                            gamma=self.gamma,
                            epsilon=self.epsilon)   
        self.distance = distance
        print('percentile: {}\nlr_rate: {}\nbatchsize: {}\nhidden_units: {}\nmax_iter: {}\nmax_len: {}'.format(percentile, lr_rate, batchsize, hidden_units, max_iter, self.max_len))
        print('init params: \n{}\n{}\n{}'.format(self.alpha, self.gamma, self.epsilon))
    '''
    Use a sequence-to-sequence autoencoder trained to minimize the 
    reconstruction loss of the trajectories to learn a latent 
    representation hi for each trajectory ti.
    '''
    def init_network(self):
        #使用keras初始化一个seq2seq模型
        #input: self.hidden_units
        #return: keras.model
        model_input = tf.keras.Input(shape=(self.max_len, 1), dtype=tf.float32, name='seq2seq_input')
        lstm_encoder = tf.keras.layers.LSTM(self.hidden_units, return_state=True)
        #final_state_h短期记忆
        #final_state_c长期记忆
        last_output, final_state_h, final_state_c = lstm_encoder(model_input)#output: [batchsize, hidden_units]
        
        decoder_input = tf.expand_dims(last_output, axis=1)#[batchsize, 1, hidden_units]
        decoder_input = tf.tile(decoder_input, [1, self.max_len, 1])#[batchsize, t, hidden_units]
        lstm_decoder = tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        output = lstm_decoder(decoder_input, initial_state=[final_state_h,final_state_c])
        
        final_output = output[:, :, 0]#[b,t]
        final_output = tf.expand_dims(final_output, axis=2)#[b,t,1]
        
        self.recons_model = tf.keras.Model(inputs=model_input, outputs=final_output)
        self.latent_model = tf.keras.Model(inputs=model_input, outputs=last_output)
        return self.recons_model, self.latent_model
    
    
    def train_network(self):
        #使用early stop来停止迭代
        #学习latent space representation
        #input: self.train_data, shape: [batchsize, max_len, 1]
        #input: self.lr_rate
        #return: self.latent_h, shape: [batchsize, batchsize]
        
        #optimizer
        adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.recons_model.compile(optimizer=adam, loss=tf.keras.losses.MeanSquaredError())
        #early stop
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4)
        
        #fit        
        
        print('Start Training.')
        history = self.recons_model.fit(x=self.train_data, y=self.train_data, 
                                        validation_split=0.2,
                                        verbose=1,epochs=1000,
                                        callbacks=[early_stop_callback])
        print('Training End.')
        print('Sum of epoch: ', len(history.history['loss']))
        return self.recons_model, self.latent_model
        
        

    
    def get_latent_vectors(self, latent_model, test_data):
        #使用训练好的数据，得到他们的隐空间表达
        print('shape of test: ', test_data.shape)
        #Test
        latent_h = latent_model.predict(test_data)
        print('shape of latent samples', latent_h.shape)
        return latent_h
        
    
    
    '''
    Compute the pairwise Euclidean distance matrix between each pair 
    of latent representations
    '''
    def distance_matrix(self, latent_data):
        #input: self.latent_h, shape: [batch_size, hidden_units]
        #return: self.matrix, shape: [batchsize, batchsize]

        print('shape of data: ', latent_data.shape)
        pair_dist = pdist(latent_data, metric='euclidean')
        matrix = squareform(pair_dist)
        print('shape of distance matrix', matrix.shape)
        self.matrix = matrix
        
        del(pair_dist)
        gc.collect()
        return self.matrix
    
    '''
    Compute the threshold distance delta defined as the pth percentile 
    of the distribution of distances
    '''
    def init_delta(self):
        #input: self.matrix, self.percentile
        #return: delta, shape: int
        all_dist = self.matrix.flatten()#1d array
        print('length of all distance: ', len(all_dist))
        all_dist = all_dist[all_dist > 0]
        print('length of distance except for 0: ', len(all_dist))
        all_dist = np.sort(all_dist)
        if self.percentile >= len(all_dist):
            raise AssertionError
        print('p = ', int(self.percentile*len(all_dist)))
        self.delta = all_dist[int(self.percentile*len(all_dist))]
        print('delta: ', self.delta)
        
        del(all_dist)
        gc.collect()
        return self.delta
    
    def train_warping_family(self):
        #存储上一次的参数，用于检测是否收敛
        new_alpha = np.inf
        new_gamma = np.inf
        new_epsilon = np.inf
        #本次使用的参数
        alpha = self.alpha
        gamma = self.gamma
        epsilon = self.epsilon
        #self.distance#一个随机初始化参数的距离度量函数
        epoch = 0
        
        #检查更新后的参数与原始参数是否相等
        print('Start training.')
        while((np.abs(alpha.numpy()-new_alpha)>1e-5 or np.abs(gamma.numpy()-new_gamma)>1e-5 or np.abs(epsilon.numpy()-new_epsilon)>1e-5) \
              and (epoch<self.max_iter)):
            print('epoch: ', epoch)
            #更新本次的参数
            if epoch > 0:
                alpha.assign(new_alpha)
                gamma.assign(new_gamma)
                epsilon.assign(new_epsilon)

            '''
            Sample S pairs of trajectories with distance in the latent space 
            < delata (denote the set of pairs as Pc),
            '''
            pc = []#[batchsize, 2]
            #所有小于delta的索引
            pc_x, pc_y = np.where((self.matrix<self.delta)&(self.matrix != 0))
            print('number of dist < delta: ', len(pc_x))
            #从上述坐标中选取batchsize个
            pc_idx = np.random.choice(np.array(list(range(len(pc_x)))), size=self.batchsize, replace=False)
            for idx in pc_idx:
                pc.append([pc_x[idx], pc_y[idx]])
            '''
            Sample S pairs of trajectories from all possible pairs (denote the set of pairs as Pall).
            '''
            p_all = []#[batchsize, 2]
            #从所有样本中，选取batchsize个任意序列对，第一个位置的索引
            p_all_i = np.random.choice(np.array(list(range(len(self.matrix)))), size=self.batchsize, replace=False)
            #从所有样本中，选取batchsize个任意序列对，第一个位置的索引
            p_all_j = np.random.choice(np.array(list(range(len(self.matrix)))), size=self.batchsize, replace=False)
            for idx in range(len(p_all_i)):
                p_all.append([p_all_i[idx], p_all_j[idx]])
            
            '''
            Start Gradient Computation
            '''
            if epoch == 0:
                #[batchsize, 2]，第二维的第一位是path中的前者在train_data的索引
                t_pc = tf.Variable(pc, dtype=tf.int32, name='pc', trainable=False)
                t_p_all = tf.Variable(p_all, dtype=tf.int32, name='p_all', trainable=False)
                
                #第一次初始化的时候声明这个train_ts为常量
                train_ts = np.squeeze(self.train_data)
                #print('time series shape has change to:', train_ts.shape)
                train_ts = tf.constant(train_ts, dtype=tf.float32, name='train_ts')
                #print('convert train_ts to tensor.')
                #print(train_ts)
            else:
                t_pc.assign(pc)
                t_p_all.assign(p_all)
            del(pc)
            del(p_all)
            gc.collect()
                
            #print(pc)
            #print(p_all)
            

            
            #计算多次梯度的时候，设置persistent=True，否则在第一次计算梯度后就会被收回所记录的梯度资源
            #默认情况下，GradientTape将自动监视在上下文中访问的所有可训练变量。
            #如果要对监视哪些变量进行精细控制，可以通过传递watch_accessed_variables=False给磁带构造器来禁用自动跟踪 
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
                g.watch([alpha, gamma, epsilon])#设置要求导的标量
                
                '''
                Define d to be the warping distance parametrized by alpha, gamma, epsilon
                '''
                self.distance.update_params(alpha, gamma, epsilon)

                #beta上部
                beta_up = 0
                print('processing set pc')
                for idx_ in range(self.batchsize):
                    
                                
                    #统计当前内存分配
                    #snapshot = tracemalloc.take_snapshot() # 快照，当前内存分配
                    #top_stats = snapshot.statistics('lineno') # 快照对象的统计
                    #for stat in top_stats[:10]:
                    #    print(stat)
                    

                    i = t_pc[idx_][0]
                    j = t_pc[idx_][1]
                    print('\t sample {}/{}'.format(idx_, self.batchsize-1), end='')
                    beta_up += self.distance.caclulate_distance(train_ts[i], train_ts[j])
                                        
                #beta下部
                beta_down = 0
                print('\nprocessing set p_all')
                for idx_ in range(self.batchsize):
                    i = t_p_all[idx_][0]
                    j = t_p_all[idx_][1]
                    print('\t sample {}/{}'.format(idx_, self.batchsize-1), end='')
                    beta_down += self.distance.caclulate_distance(train_ts[i], train_ts[j])
                    
                beta = beta_up / beta_down
                print('\nbeta: ', beta)

            '''
            Compute analytical gradients and update parameters
            '''   
            new_alpha = alpha - self.lr_rate * g.gradient(beta, alpha)
            new_gamma = gamma - self.lr_rate * g.gradient(beta, gamma)
            new_epsilon = epsilon - self.lr_rate * g.gradient(beta, epsilon)

            self.distance.update_params(new_alpha, new_gamma, new_epsilon)
            print('epoch: {}\tparameters updated as: '.format(epoch))
            print('alpha: %.8f, gamma: %.8f, epsilon: %.8f, beta: %.8f\n'%(new_alpha, new_gamma, new_epsilon, beta))
            
            epoch += 1
            
            # Drop the reference to the tape
            del(g)
            gc.collect()
                
        return beta, new_alpha, new_gamma, new_epsilon


