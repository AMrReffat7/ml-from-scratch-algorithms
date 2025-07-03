import numpy as np


class DBSCANClustering:
    """
    DBSCAN clustering algorithm.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, data):
        """
        Perform clustering on the data.
        """
        n_samples = data.shape[0]
        self.labels = np.full(n_samples, -1)  # -1 means noise
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for sample_idx in range(n_samples):
            if visited[sample_idx]:
                continue

            visited[sample_idx] = True
            neighbors = self._region_query(data, sample_idx)

            if len(neighbors) < self.min_samples:
                # Mark as noise
                self.labels[sample_idx] = -1
            else:
                # Start a new cluster
                self._expand_cluster(data, sample_idx, neighbors, cluster_id, visited)
                cluster_id += 1

    def _region_query(self, data, sample_idx):
        """
        Find all points within eps of the sample.
        """
        distances = np.linalg.norm(data - data[sample_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, data, sample_idx, neighbors, cluster_id, visited):
        """
        Recursively grow the cluster.
        """
        self.labels[sample_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._region_query(data, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))

            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
            i += 1
