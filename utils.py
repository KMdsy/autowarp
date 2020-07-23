import numpy as np

def read_dataset_cluster(opts, dataset, label_dict=None):
    if dataset == 'train':
        data = np.loadtxt(opts['train_file'])
        file_name = opts['train_file']
    elif dataset == 'test':
        data = np.loadtxt(opts['test_file'])
        file_name = opts['test_file']
    elif dataset == 'v':
        data = np.loadtxt(opts['train_file'])
        file_name = opts['train_file']
        
        
    label = data[:,0]
    label = label + 10
    label = -1 * label
    
    if label_dict is None:
        label_dict = {}
        label_list = np.unique(label)
        for idx in range(len(label_list)):
            label_dict[str(label_list[idx])] = idx#key：-1*原始label，value：新label

    o_label = list(label_dict.keys())
    for l in o_label:
        label[label == float(l)] = label_dict[l]
        
    label = label.astype(int)
    data = data[:,1:]
        
    #数据集中的类别数量 
    print(dataset)
    print(file_name)
    print('cluster num: ', len(np.unique(label)))
    print('Time Series Length: ', data.shape[1])
    print('sample num', data.shape[0])
            

    '''
    data: 分割后的数据，将数据集中所有序列分段后的结果堆叠在一起了，若有100条数据，每条序列分割成3个子序列，则data中包含300个序列
    fragments_num: 分割后的数据段数
    label:
    normal_cluster: 代表正常的标签，在所有数据集中，将数据占比多的一方视为正常数据
    '''
    return data, label, label_dict