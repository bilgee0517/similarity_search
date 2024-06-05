import numpy as np
from heapq import heappop, heappush, heapify, nlargest, heapreplace
from random import random
from math import log2
import pickle
from operator import itemgetter

class HNSW(object):
    def __init__(self, distance_type, m=5, ef=200, m0=None, vectorized=False):
        self.data = []
        # Select the distance function based on the specified type
        if distance_type == "euclidean":
            self.distance_func = self.l2_distance
        elif distance_type == "cosine":
            self.distance_func = self.cosine_distance
        elif distance_type == "mips":
            self.distance_func = self.mips_distance
        else:
            raise TypeError('Please check your distance type!')

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = self.distance_func
        else:
            self.distance = self.distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

    def l2_distance(self, a, b):
        """Compute the L2 (Euclidean) distance between vectors a and b."""
        return np.linalg.norm(a - b)
    
    def mips_distance(self, a, b):
        """Compute the maximum inner product (MIPS) distance between vectors a and b."""
        return np.dot(a, b)

    def cosine_distance(self, a, b):
        """Compute the cosine distance between vectors a and b."""
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except ValueError:
            print(a)
            print(b)

    def _distance(self, x, y):
        """Compute the distance between vectors x and y using the selected distance function."""
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        """Compute the distance between vector x and a list of vectors ys using the selected distance function."""
        return [self.distance_func(x, y) for y in ys]

    def add(self, elem, ef=None):
        """Add an element to the HNSW structure."""
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        level = int(-log2(random()) * self._level_mult) + 1
        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = distance(elem, data[point])
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            ep = [(-dist, point)]
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                ep = self._search_graph(elem, ep, layer, ef)
                layer[idx] = layer_idx = {}
                self._select_naive(layer_idx, ep, level_m, layer, heap=True)
                for j, dist in layer_idx.items():
                    self._select_naive(layer[j], (idx, dist), level_m, layer)
        for i in range(len(graphs), level):
            graphs.append({idx: {}})
            self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """Search for the nearest neighbors of query vector q."""
        if ef is None:
            ef = self._ef

        if self._enter_point is None:
            raise ValueError("Empty graph")

        dist = self.distance(q, self.data[self._enter_point])
        point = self._enter_point
        for layer in reversed(self._graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        ep = self._search_graph(q, [(-dist, point)], self._graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        indices, distances = zip(*[(idx, -md) for md, idx in ep])
        return list(distances), list(indices)

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Search for the closest neighbor in a specific layer."""
        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        """Search for neighbors in a specific layer."""
        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break

            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):
        """Select the nearest neighbors without using heuristics."""
        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md

    def __getitem__(self, idx):
        """Get the items in the layer of the graph at index idx."""
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return

    def save_graph(self, filename):
        """Save the graph to a file."""
        with open(filename, 'wb') as f:
            pickle.dump((self.data, self._graphs, self._enter_point), f)

    def load_graph(self, filename):
        """Load the graph from a file."""
        with open(filename, 'rb') as f:
            self.data, self._graphs, self._enter_point = pickle.load(f)

    def print_layer_sizes(self):
        """Print the sizes of each layer in the graph."""
        for i, layer in enumerate(self._graphs):
            print(f"Layer {i}: {len(layer)} elements")
