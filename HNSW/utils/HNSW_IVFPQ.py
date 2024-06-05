import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

class CustomIVFPQ:
    def __init__(self, d, nlist, nprobe, m, pca_dim=None, estimator='KMeans', **estimator_kwargs):
        assert d % m == 0, "d must be a multiple of m"
        
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe
        self.pca_dim = pca_dim
        self.m = m
        self.k = 256
        self.ds = d // m

        self.centroids = None
        self.labels = None
        self.codes = None
        self.global_indices = None
        self.PCA = None

        if estimator.lower() == 'kmeans':
            self.estimator = KMeans(n_clusters=self.nlist, **estimator_kwargs)
        elif estimator.lower() == 'minibatchkmeans':
            self.estimator = MiniBatchKMeans(n_clusters=self.nlist, **estimator_kwargs)
        else:
            raise ValueError(f"Unknown estimator `{estimator}`. Choose from [`KMeans`, `MiniBatchKMeans`].")

        if estimator.lower() == 'kmeans':
            self.pq_estimators = [KMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(self.m)]
        elif estimator.lower() == 'minibatchkmeans':
            self.pq_estimators = [MiniBatchKMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(self.m)]
        
        self.is_trained = False

    def train(self, X):
        assert not self.is_trained, "Estimator is already trained"
        
        if self.pca_dim is not None:
            self.PCA = PCA(n_components=self.pca_dim)
            X = self.PCA.fit_transform(X)
        
        print(f"Training data shape after PCA (if applied): {X.shape}")
        
        self.estimator.fit(X)
        self.centroids = self.estimator.cluster_centers_

        self.codes = [[] for _ in range(self.nlist)]
        self.global_indices = [[] for _ in range(self.nlist)]
        
        labels = self.estimator.predict(X)
        
        for i in range(self.nlist):
            cluster_data = X[labels == i]
            if len(cluster_data) > 0:
                for j in range(self.m):
                    X_j = cluster_data[:, j * self.ds : (j + 1) * self.ds]
                    if len(X_j) >= self.k:
                        print(f"Training PQ on cluster {i}, segment {j}, data shape: {X_j.shape}")
                        self.pq_estimators[j].fit(X_j)
                    else:
                        print(f"Skipping PQ training for cluster {i}, segment {j} due to insufficient samples (n_samples={len(X_j)}, n_clusters={self.k})")

        self.is_trained = True

    def encode(self, X):
        X = X.astype(np.float32)
        encoded = np.empty((len(X), self.m), dtype=np.uint8)

        for i in range(self.m):
            estimator_i = self.pq_estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            print(f"Encoding segment {i}, data shape: {X_i.shape}")
            encoded[:, i] = estimator_i.predict(X_i)
        
        return encoded

    def add(self, X):
        assert self.is_trained, "Estimator has to be trained"

        if self.PCA is not None:
            X = self.PCA.transform(X)

        self.labels = self.estimator.predict(X)
        
        for i in range(X.shape[0]):
            cluster_idx = self.labels[i]
            self.codes[cluster_idx].append(X[i, :].astype(np.float32))
            self.global_indices[cluster_idx].append(i)
        
        for i in range(self.nlist):
            if len(self.codes[i]) > 0:
                print(f"Encoding cluster {i}, data shape before encode: {np.vstack(self.codes[i]).shape}")
                self.codes[i] = self.encode(np.vstack(self.codes[i]))
                self.global_indices[i] = np.array(self.global_indices[i])
    
    def find_closest_centroids(self, Y):
        assert self.centroids is not None, "Need to run `train` first to learn centroids"
        distances_to_centroids = -np.dot(Y, self.centroids.T)
        closest_centroids_indices = np.argsort(distances_to_centroids, axis=1)[:, :self.nprobe]
        return closest_centroids_indices

    def aggregate_vectors(self, centroids_indices):
        assert self.codes is not None, "Need to run `add` first to learn labels and codes"
        indices, X = [], []
        for i in range(centroids_indices.shape[0]):
            X_i = np.concatenate([self.codes[ci] for ci in centroids_indices[i]], axis=0)
            indices_i = np.concatenate([self.global_indices[ci] for ci in centroids_indices[i]], axis=0)
            X.append(X_i)
            indices.append(indices_i)
        return indices, X

    def compute_asymmetric_distances(self, Y, centroids_indices):
        assert self.is_trained, "Estimators have to be trained"
        assert self.codes is not None, "Codes were not created, use `add` to create them"
        
        if self.PCA is not None:
            Y = self.PCA.transform(Y)

        n_queries = len(Y)
        all_distances = []
        all_indices = []

        for query_idx in range(n_queries):
            y = Y[query_idx:query_idx+1]
            centroids_indices_query = centroids_indices[query_idx:query_idx+1]
            
            indices, X = self.aggregate_vectors(centroids_indices_query)
            
            total_codes = len(X[0])
            X = np.array(X[0])

            distance_table = np.empty((1, self.m, self.k), dtype=np.float32)

            for i in range(self.m):
                Y_i = y[:, i * self.ds : (i + 1) * self.ds]
                centers = self.pq_estimators[i].cluster_centers_
                distance_table[:, i, :] = cdist(Y_i, centers, metric='euclidean')

            distances = np.zeros((1, total_codes), dtype=np.float32)

            for i in range(self.m):
                distances += distance_table[:, i, X[:, i]]

            all_distances.append(distances)
            all_indices.append(indices)

        return all_distances, all_indices

    def search(self, Y, k):
        Y = np.atleast_2d(Y)

        if self.PCA is not None:
            Y = self.PCA.transform(Y)

        centroids_to_explore = self.find_closest_centroids(Y)
        all_distances, all_global_indices = self.compute_asymmetric_distances(Y, centroids_to_explore)

        result_distances = []
        result_indices = []

        for query_idx in range(len(Y)):
            distances = all_distances[query_idx]
            global_indices = all_global_indices[query_idx]

            sorted_indices = np.argsort(distances, axis=1)[:, :k][0]
            result_distances.append(distances[0, sorted_indices])
            result_indices.append(global_indices[0][sorted_indices])

        return result_distances, result_indices

    def print_pqcenters(self):
        centers = []
        for i in range(self.m):
            centers.append(self.pq_estimators[i].cluster_centers_)
        return centers

class HNSW_IVFPQ:
    def __init__(self, d, nlist, nprobe, m, pca_dim=None, ivf_estimator='KMeans', ivf_kwargs={}, hnsw_distance='euclidean', hnsw_m=5, hnsw_ef=200):
        self.ivfpq = CustomIVFPQ(d, nlist, nprobe, m, pca_dim, ivf_estimator, **ivf_kwargs)
        self.hnsw = HNSW(hnsw_distance, m=hnsw_m, ef=hnsw_ef)

    def train(self, X):
        self.ivfpq.train(X)

    def add(self, X):
        self.ivfpq.add(X)
        for x in X:
            self.hnsw.add(x)

    def search(self, Y, k):
        ivf_result_distances, ivf_result_indices = self.ivfpq.search(Y, k)
        hnsw_results = []
        
        for i, query in enumerate(Y):
            candidates = ivf_result_indices[i]
            hnsw_query_results = []
            
            for candidate_idx in candidates:
                candidate_vector = self.ivfpq.global_indices[candidate_idx]
                hnsw_result = self.hnsw.search(candidate_vector, k)
                hnsw_query_results.extend(hnsw_result)
            
            hnsw_query_results.sort(key=lambda x: x[1])
            hnsw_results.append(hnsw_query_results[:k])
        
        return hnsw_results
