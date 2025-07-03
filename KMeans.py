import numpy as np


class KMeansClustering:
    """
    KMeans clustering algorithm.
    """

    def __init__(self, n_clusters=3, max_iterations=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def fit(self, data):
        """
        Perform clustering on the data.
        """
        n_samples, n_features = data.shape

        # Randomly initialize centroids by selecting random samples
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = data[random_indices]

        for iteration in range(self.max_iterations):
            # Assign each point to the nearest centroid
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            self.cluster_labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array(
                [
                    (
                        data[self.cluster_labels == cluster_idx].mean(axis=0)
                        if len(data[self.cluster_labels == cluster_idx]) > 0
                        else self.centroids[cluster_idx]
                    )
                    for cluster_idx in range(self.n_clusters)
                ]
            )

            # Check convergence (if centroids move less than tolerance)
            centroid_shifts = np.linalg.norm(self.centroids - new_centroids, axis=1)
            if np.all(centroid_shifts <= self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, data):
        """
        Assign each sample to the nearest centroid.
        """
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
