import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


class CustomIndexFlat:
    """ Custom implementation of Flat Index
    d: size of original embeddings
    """

    def __init__(self, d):
        self.d = d
        self.codes = None

        self.is_trained = False


    def train(self, X):
        assert not self.is_trained, "estimators are already trained"

        self.is_trained = True

            
    def add(self, X):
        self.codes = X.astype(np.float32)


    def search(self, Y, k):
        distances_full = cdist(Y, self.codes, metric='euclidean')
        sorted_indices = np.argsort(distances_full, axis=1)[:, :k]

        diag_indices = np.diag_indices(distances_full[:, sorted_indices].shape[0])
        distances = distances_full[:, sorted_indices][diag_indices]

        return distances, sorted_indices
  

class CustomIndexPQ:
    """ Custom implementation of PQ Index in 8 bits 
    d: size of original embeddings
    m: number of segments within embedded vectors
    estimator_kwargs: Additional hyperparameters passed onto the sklearn KMeans/MiniBatchKMeans class
    """

    def __init__(self, d, m, estimator='KMeans', **estimator_kwargs):
        assert d % m == 0, "d must be a multiple of m"

        self.d = d
        self.m = m
        self.k = 2**8       # nb of clusters per segment
        self.ds = d // m
        self.codes = None

        if estimator.lower() == 'kmeans':
            self.estimators = [KMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(self.m)]
        elif estimator.lower() == 'minibatchkmeans':
            self.estimators = [MiniBatchKMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(self.m)]
        else:
            raise ValueError(f"Unknown estimator `{estimator}`. Choose from [`KMeans`, `MiniBatchKMeans`].")

        self.is_trained = False


    def train(self, X):
        assert not self.is_trained, "estimators are already trained"

        for i in range(self.m):
            estimator_i = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]

            estimator_i.fit(X_i)

        self.is_trained = True


    def encode(self, X):
        assert self.is_trained, "estimators have to be trained"

        encoded = np.empty((len(X), self.m), dtype=np.uint8)

        for i in range(self.m):
            estimator_i = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            encoded[:, i] = estimator_i.predict(X_i)
        
        return encoded
            
    
    def add(self, X):
        assert self.is_trained, "estimators have to be trained"

        self.codes = self.encode(X)

    
    def compute_asymmetric_distances(self, Y):
        assert self.is_trained, "estimators have to be trained"
        assert self.codes is not None, "codes were note created, use `add` to create them"

        Y = np.atleast_2d(Y)
        n_queries = len(Y)
        n_codes = len(self.codes)

        distance_table = np.empty((n_queries, self.m, self.k), dtype=np.float32)

        for i in range(self.m):
            Y_i = Y[:, i * self.ds : (i + 1) * self.ds]
            centers = self.estimators[i].cluster_centers_
            distance_table[:, i, :] = cdist(Y_i, centers, metric='euclidean')

        distances = np.zeros((n_queries, n_codes), dtype=np.float32)

        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]
        
        return distances

    
    def search(self, Y, k):
        distances_full = self.compute_asymmetric_distances(Y)
        sorted_indices = np.argsort(distances_full, axis=1)[:, :k]

        diag_indices = np.diag_indices(distances_full[:, sorted_indices].shape[0])
        distances = distances_full[:, sorted_indices][diag_indices]

        return distances, sorted_indices
        

class CustomIndexIVF:
    """ Custom implementation of IVF Index 
    d: size of original embeddings
    nlist: number of centroids to build
    nprobe: number of closest-neighbor centroids to visit during search
    estimator_kwargs: Additional hyperparameters passed onto the sklearn KMeans/MiniBatchKMeans class
    """

    def __init__(self, d, nlist, nprobe, estimator='KMeans', **estimator_kwargs):
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe

        self.centroids = None
        self.labels = None
        self.codes = None

        if estimator.lower() == 'kmeans':
            self.estimator = KMeans(n_clusters=self.nlist, **estimator_kwargs)
        elif estimator.lower() == 'minibatchkmeans':
            self.estimator = MiniBatchKMeans(n_clusters=self.nlist, **estimator_kwargs)
        else:
            raise ValueError(f"Unknown estimator `{estimator}`. Choose from [`KMeans`, `MiniBatchKMeans`].")

        self.is_trained = False

    
    def train(self, X):
        assert not self.is_trained, "estimator is already trained"

        self.estimator.fit(X)

        self.centroids = self.estimator.cluster_centers_

        self.is_trained = True


    def add(self, X):
        assert self.is_trained, "estimator has to be trained"

        self.codes = X.astype(np.float32)
        self.labels = self.estimator.predict(X)
    
    
    def find_closest_centroids(self, Y):
        assert self.centroids is not None, "need to run `train` first, to train and learn centroids"

        distances_to_centroids = euclidean_distances(Y, self.centroids, squared=True)
        
        closest_centroids_indices = np.argsort(distances_to_centroids, axis=1)[:, :self.nprobe]
        return closest_centroids_indices


    def aggregate_vectors(self, centroids_indices):
        assert self.codes is not None, "need to run `add` first to learn labels and codes"

        X = []
        indices = []

        for i in range(len(centroids_indices)):
            X_i = np.empty((0, self.d))
            indices_i = []

            for j in range(len(self.codes)):
                if self.labels[j] in centroids_indices[i]:
                    X_i = np.vstack((X_i, self.codes[j, :]))
                    indices_i.append(j)

            X.append(X_i)
            indices.append(indices_i)

        return indices, X


    def search(self, Y, k):
        Y = np.atleast_2d(Y)

        centroids_to_explore = self.find_closest_centroids(Y)

        indices, X = self.aggregate_vectors(centroids_to_explore)

        distances = []
        closest_indices = []

        for i in range(Y.shape[0]):
            distances.append(euclidean_distances(Y[i, :].reshape(1,-1), X[i], squared=True))
            closest_indices.append([indices[i][idx] for idx in np.argsort(distances[i], axis=1)[:, :k][0]])

        closest_indices = np.stack(closest_indices)

        return closest_indices


class CustomOptimizedIndexIVF:
    """ Custom implementation of IVF Index 
    d: size of original embeddings
    nlist: number of centroids to build
    nprobe: number of closest-neighbor centroids to visit during search
    estimator_kwargs: Additional hyperparameters passed onto the sklearn KMeans/MiniBatchKMeans class
    """

    def __init__(self, d, nlist, nprobe, pca_dim=0, estimator='KMeans', **estimator_kwargs):
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe
        self.pca_dim = pca_dim

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

        self.is_trained = False

    
    def train(self, X):
        assert not self.is_trained, "estimator is already trained"

        if self.pca_dim > 0:
            self.PCA = PCA(n_components=self.pca_dim)
            X = self.PCA.fit_transform(X)

        self.estimator.fit(X)

        self.centroids = self.estimator.cluster_centers_

        self.is_trained = True


    def add(self, X):
        assert self.is_trained, "estimator has to be trained"

        if self.PCA is not None:
            X = self.PCA.transform(X)

        self.labels = self.estimator.predict(X)
        self.codes = [[] for _ in range(self.nlist)]
        self.global_indices = [[] for _ in range(self.nlist)]

        for i in range(X.shape[0]):
            self.codes[self.labels[i]].append(X[i, :].astype(np.float32))
            self.global_indices[self.labels[i]].append(i)
        
        self.codes = [np.concatenate([X_i], axis=0) for X_i in self.codes]
    
    
    def find_closest_centroids(self, Y):
        assert self.centroids is not None, "need to run `train` first, to train and learn centroids"

        distances_to_centroids = cdist(Y, self.centroids, metric='euclidean')
        closest_centroids_indices = np.argsort(distances_to_centroids, axis=1)[:, :self.nprobe]
        return closest_centroids_indices


    def aggregate_vectors(self, centroids_indices):
        assert self.codes is not None, "need to run `add` first to learn labels and codes"

        indices, X = [], []
        for i in range(centroids_indices.shape[0]):
            X_i = np.concatenate([self.codes[ci] for ci in centroids_indices[i]], axis=0)
            indices_i = np.concatenate([self.global_indices[ci] for ci in centroids_indices[i]], axis=0)
            X.append(X_i)
            indices.append(indices_i)

        return indices, X


    def search(self, Y, k):
        Y = np.atleast_2d(Y)

        if self.PCA is not None:
            Y = self.PCA.transform(Y)

        centroids_to_explore = self.find_closest_centroids(Y)

        indices, X = self.aggregate_vectors(centroids_to_explore)

        closest_indices = []

        for i in range(Y.shape[0]):
            distances = cdist(Y[i, :].reshape(1,-1), X[i], metric='euclidean')
            closest_indices.append([indices[i][idx] for idx in np.argsort(distances, axis=1)[:, :k][0]])

        return np.array(closest_indices)



class CustomOptimizedIndexIVF_IP:
    """ Custom implementation of IVF Index 
    d: size of original embeddings
    nlist: number of centroids to build
    nprobe: number of closest-neighbor centroids to visit during search
    estimator_kwargs: Additional hyperparameters passed onto the sklearn KMeans/MiniBatchKMeans class
    """

    def __init__(self, d, nlist, nprobe, pca_dim=0, estimator='KMeans', **estimator_kwargs):
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe
        self.pca_dim = pca_dim

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

        self.is_trained = False

    
    def train(self, X):
        assert not self.is_trained, "estimator is already trained"

        if self.pca_dim > 0:
            self.PCA = PCA(n_components=self.pca_dim)
            X = self.PCA.fit_transform(X)

        self.estimator.fit(X)

        self.centroids = self.estimator.cluster_centers_

        self.is_trained = True


    def add(self, X):
        assert self.is_trained, "estimator has to be trained"

        if self.PCA is not None:
            X = self.PCA.transform(X)

        self.labels = self.estimator.predict(X)
        self.codes = [[] for _ in range(self.nlist)]
        self.global_indices = [[] for _ in range(self.nlist)]

        for i in range(X.shape[0]):
            self.codes[self.labels[i]].append(X[i, :].astype(np.float32))
            self.global_indices[self.labels[i]].append(i)
        
        self.codes = [np.concatenate([X_i], axis=0) for X_i in self.codes]
    
    
    def find_closest_centroids(self, Y):
        assert self.centroids is not None, "need to run `train` first, to train and learn centroids"

        #distances_to_centroids = cdist(Y, self.centroids, metric='euclidean')
        distances_to_centroids = - np.dot(Y, self.centroids.T)
        closest_centroids_indices = np.argsort(distances_to_centroids, axis=1)[:, :self.nprobe]
        return closest_centroids_indices


    def aggregate_vectors(self, centroids_indices):
        assert self.codes is not None, "need to run `add` first to learn labels and codes"

        indices, X = [], []
        for i in range(centroids_indices.shape[0]):
            X_i = np.concatenate([self.codes[ci] for ci in centroids_indices[i]], axis=0)
            indices_i = np.concatenate([self.global_indices[ci] for ci in centroids_indices[i]], axis=0)
            X.append(X_i)
            indices.append(indices_i)

        return indices, X


    def search(self, Y, k):
        Y = np.atleast_2d(Y)

        if self.PCA is not None:
            Y = self.PCA.transform(Y)

        centroids_to_explore = self.find_closest_centroids(Y)

        indices, X = self.aggregate_vectors(centroids_to_explore)

        closest_indices = []

        for i in range(Y.shape[0]):
            #distances = cdist(Y[i, :].reshape(1,-1), X[i], metric='euclidean')
            distances = - np.dot(Y[i, :].reshape(1,-1), X[i].T)
            closest_indices.append([indices[i][idx] for idx in np.argsort(distances, axis=1)[:, :k][0]])

        return np.array(closest_indices)