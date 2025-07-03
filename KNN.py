import numpy as np


class KNNClassifier:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k=5):
        # Number of neighbors to consider
        self.k = k

    def _majority_vote(self, neighbor_labels):
        """
        Return the most common label among the neighbors.
        """
        # Count occurrences of each label
        label_counts = np.bincount(neighbor_labels.astype("int"))
        # Return the label with the highest count
        return label_counts.argmax()

    def predict(self, test_data, train_data, train_labels):
        """
        Predict labels for test data based on k nearest neighbors.
        """
        # Array to store predictions
        predictions = np.empty(test_data.shape[0])

        # Loop over each test sample
        for test_index, test_sample in enumerate(test_data):
            # Compute distances to all training samples
            distances = np.array(
                [
                    np.sqrt(np.sum((test_sample - train_sample) ** 2))
                    for train_sample in train_data
                ]
            )

            # Get indices of the k nearest neighbors (sorted by distance)
            nearest_neighbor_indices = np.argsort(distances)[: self.k]

            # Retrieve labels of the k nearest neighbors
            nearest_neighbor_labels = train_labels[nearest_neighbor_indices]

            # Determine the predicted label by majority vote
            predicted_label = self._majority_vote(nearest_neighbor_labels)

            # Store the prediction
            predictions[test_index] = predicted_label

        return predictions
