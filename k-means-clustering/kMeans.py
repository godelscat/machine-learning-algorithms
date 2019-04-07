# k-means clustering algorithm from ref 2 in page 203
'''
Input: dataset D={x1, x2, ..., xn}; cluster number n_clusters; maximum iter steps max_iter;
Output: clusters K
Algorithm:
1. randomly choose n_clusters samples as initial mean vector {u1, u2, ... , uk};
2. for i in range(max_iter)；
    set C_k = null, for k in range(n_clusters)
    for j in range(n):
        for k in range(n_clusters):
            calculate the distance of sample xj and mean vector uk, d_jk;
        set the nearest one as sample cluster lambda_j = argmin d_jk 
        put sample xj into cluster C_lambda_j
    for k in range(n_clusters):
        calculate new mean vector u';
        if u' != u : 
            update u
        else :
            continue
    if all vector remain unchanged or within in a tolerance:
        return clusters C_k;
return clusters C_k;
'''
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
    
    def train(self, data):
        nsample = len(data)
        # randomly choose n_clusters sample as mean vectors
        idx = np.arange(nsample)
        init_idx = np.random.choice(nsample, self.n_clusters, replace=False)
        mean_vec = data[init_idx]
        # start iter
        for i in range(self.max_iter):
            Clusters = [None] * self.n_clusters
            distance = np.empty(shape=(0, nsample), dtype=int)
            for k in range(self.n_clusters):
                temp = (data - mean_vec[k]) ** 2
                temp = np.sqrt(np.sum(temp, axis=1)) # cal distance
                distance = np.vstack((distance, temp))
            cluster_idx = np.argmin(distance, axis=0)
            assert len(cluster_idx) == nsample
            self.labels_ = cluster_idx
            Clusters = [idx[cluster_idx == k] for k in range(self.n_clusters)]
            flag = False 
            for k in range(self.n_clusters):
                new_mean_vector = np.sum(data[Clusters[k]], axis=0) / len(Clusters[k])
                if np.sum(np.absolute(new_mean_vector - mean_vec[k])) > self.tol:
                    flag = True
                    mean_vec[k] = new_mean_vector
            if not flag:
                break

if __name__ == "__main__":
    raw_data = pd.read_csv('../dataset/MNIST/train.csv')
    labels = raw_data['label'].values
    data = raw_data.iloc[:,1:].values
    '''
    kmeans = KMeans(n_clusters=10)
    kmeans.train(data)
    '''
    '''
    from sklearn import cluster
    kmeans = cluster.KMeans(n_clusters=10)
    kmeans.fit(data)
    real_cluster = [np.sum(labels == k) for k in range(10)]
    pred_cluster = [np.sum(kmeans.labels_ == k ) for k in range(10)]
    print(real_cluster)
    print('-'*10)
    print(pred_cluster)
    '''
