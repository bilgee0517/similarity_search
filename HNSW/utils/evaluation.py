import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import time 

class KNNEvaluation:
    def __init__(self, train_data, test_data, hnsw, k=10, ef=200, metric='cosine'):
        self.train_data = train_data
        self.test_data = test_data
        self.hnsw = hnsw
        self.k = k
        self.ef = ef
        self.metric = metric

    @staticmethod
    def brute_force_knn(data, query, k, metric='cosine'):
        """
        Find the k nearest neighbors using a brute-force approach.
        
        Parameters:
        - data: A 2D numpy array where each row is a data vector.
        - query: A 2D numpy array where each row is a query vector.
        - k: Number of nearest neighbors to find.
        - metric: Distance metric to use (default is 'cosine').
        
        Returns:
        - distances: Distances to the k nearest neighbors for each query.
        - indices: Indices of the k nearest neighbors for each query.
        """
        # Compute the distance between each query and all data points
        distances = cdist(query, data, metric=metric)
        
        # Find the indices of the k smallest distances
        indices = np.argsort(distances, axis=1)[:, :k]
        
        # Get the corresponding distances
        k_distances = np.take_along_axis(distances, indices, axis=1)
        
        return k_distances, indices

    def evaluate_recall(self):
        recall_at_1 = 0
        recall_at_10 = 0
        smetric = 0
        num_queries = len(self.test_data)

        for i in tqdm(range(num_queries)):
            _, indices_brute = self.brute_force_knn(
                self.train_data, 
                self.test_data[i].reshape(1, -1), 
                self.k, 
                metric=self.metric
            )

            _, indices_hnsw = self.hnsw.search(self.test_data[i], self.k, ef=self.ef)

            # Evaluate recall@1
            if indices_brute[0][0] == indices_hnsw[0]:
                recall_at_1 += 1

            # Evaluate recall@10
        
            for t in range(10):
                for y in range(10):
                    if indices_hnsw[t] == indices_brute[0][y]:
                        recall_at_10 += 1 

        
            for t in range(9):
                for y in range(9):
                    if indices_hnsw[t] == indices_brute[0][y]:
                        smetric += 9 - y
        
        smetric /= num_queries
        recall_at_1 /= num_queries
        recall_at_10 /= num_queries * 10 

        return recall_at_1, recall_at_10, smetric
    
    def compare_time(self):
        brute_force_times = []
        hnsw_times = []
        num_queries = len(self.test_data)

        for i in tqdm(range(num_queries)):
            query = self.test_data[i].reshape(1, -1)

            # Measure brute force time
            start_time = time.time()
            self.brute_force_knn(self.train_data, query, self.k, metric=self.metric)
            brute_force_times.append(time.time() - start_time)

            # Measure HNSW time
            start_time = time.time()
            self.hnsw.search(query[0], self.k, ef=self.ef)
            hnsw_times.append(time.time() - start_time)

        avg_brute_force_time = np.mean(brute_force_times)
        avg_hnsw_time = np.mean(hnsw_times)

        return avg_brute_force_time, avg_hnsw_time