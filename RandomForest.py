import numpy as np


def entropy(labels):
    """
    Compute entropy of a label distribution.
    """
    class_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


class DecisionTreeClassifier:
    """
    Decision Tree classifier using entropy (information gain).
    """

    def __init__(self, max_depth=5, min_samples_split=2, n_features_to_consider=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_to_consider = n_features_to_consider
        self.root = None

    def _find_best_split(self, features, labels):
        """
        Find the feature index and threshold that yield the highest information gain.
        """
        n_samples, n_total_features = features.shape
        if self.n_features_to_consider is None:
            feature_indices = range(n_total_features)
        else:
            feature_indices = np.random.choice(
                n_total_features, self.n_features_to_consider, replace=False
            )

        best_gain = -1
        best_feature_index = None
        best_threshold = None

        parent_entropy = entropy(labels)

        for feature_idx in feature_indices:
            thresholds = np.unique(features[:, feature_idx])
            for threshold in thresholds:
                left_mask = features[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if len(labels[left_mask]) == 0 or len(labels[right_mask]) == 0:
                    continue

                left_entropy = entropy(labels[left_mask])
                right_entropy = entropy(labels[right_mask])

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                weighted_entropy = (
                    n_left * left_entropy + n_right * right_entropy
                ) / n_samples
                info_gain = parent_entropy - weighted_entropy

                if info_gain > best_gain:
                    best_gain = info_gain
                    best_feature_index = feature_idx
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, features, labels, current_depth):
        """
        Recursively build the decision tree.
        """
        n_samples, n_features = features.shape
        n_unique_labels = len(np.unique(labels))

        # Check stopping criteria
        if (
            current_depth >= self.max_depth
            or n_unique_labels == 1
            or n_samples < self.min_samples_split
        ):
            return {"type": "leaf", "label": self._most_common_label(labels)}

        feature_idx, threshold = self._find_best_split(features, labels)

        if feature_idx is None:
            return {"type": "leaf", "label": self._most_common_label(labels)}

        left_indices = features[:, feature_idx] <= threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(
            features[left_indices], labels[left_indices], current_depth + 1
        )
        right_subtree = self._build_tree(
            features[right_indices], labels[right_indices], current_depth + 1
        )

        return {
            "type": "node",
            "feature_index": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def fit(self, features, labels):
        """
        Train the decision tree classifier.
        """
        self.root = self._build_tree(features, labels, current_depth=0)

    def _predict_sample(self, sample, node):
        """
        Predict label for a single sample by traversing the tree.
        """
        if node["type"] == "leaf":
            return node["label"]

        if sample[node["feature_index"]] <= node["threshold"]:
            return self._predict_sample(sample, node["left"])
        else:
            return self._predict_sample(sample, node["right"])

    def predict(self, features):
        """
        Predict labels for all samples.
        """
        return np.array(
            [self._predict_sample(sample, self.root) for sample in features]
        )

    def _most_common_label(self, labels):
        """
        Return the most frequent label.
        """
        label_values, counts = np.unique(labels, return_counts=True)
        return label_values[np.argmax(counts)]


class RandomForestClassifier:
    """
    Random Forest classifier using entropy-based decision trees.
    """

    def __init__(
        self, n_trees=10, max_depth=5, min_samples_split=2, n_features_to_consider=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_to_consider = n_features_to_consider
        self.trees = []

    def fit(self, features, labels):
        """
        Train the Random Forest by fitting multiple decision trees on bootstrap samples.
        """
        self.trees = []
        n_samples = features.shape[0]

        for _ in range(self.n_trees):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_features = features[bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features_to_consider=self.n_features_to_consider,
            )
            tree.fit(bootstrap_features, bootstrap_labels)
            self.trees.append(tree)

    def predict(self, features):
        """
        Predict labels by majority voting across all trees.
        """
        # Collect predictions from each tree
        all_tree_predictions = np.array([tree.predict(features) for tree in self.trees])
        # Transpose shape to (n_samples, n_trees)
        all_tree_predictions = all_tree_predictions.T

        final_predictions = []
        for sample_predictions in all_tree_predictions:
            # Majority vote
            label_values, counts = np.unique(sample_predictions, return_counts=True)
            majority_label = label_values[np.argmax(counts)]
            final_predictions.append(majority_label)

        return np.array(final_predictions)
