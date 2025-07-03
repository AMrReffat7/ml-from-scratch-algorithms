import numpy as np


class GaussianNaiveBayesClassifier:
    """
    Gaussian Naive Bayes classifier for continuous numeric features.
    """

    def fit(self, training_data, training_labels):
        """
        Estimate the mean, variance, and class priors from the training data.

        Parameters:
        - training_data: np.ndarray of shape (n_samples, n_features)
        - training_labels: np.ndarray of shape (n_samples,)
        """
        self.classes = np.unique(training_labels)
        n_features = training_data.shape[1]

        # Dictionaries to store parameters for each class
        self.class_feature_means = {}
        self.class_feature_variances = {}
        self.class_priors = {}

        for class_label in self.classes:
            # Select samples belonging to this class
            samples_in_class = training_data[training_labels == class_label]

            # Compute per-feature mean and variance
            self.class_feature_means[class_label] = samples_in_class.mean(axis=0)
            self.class_feature_variances[class_label] = samples_in_class.var(axis=0)

            # Compute class prior probability
            self.class_priors[class_label] = (
                samples_in_class.shape[0] / training_data.shape[0]
            )

    def _compute_log_likelihood(self, class_label, sample):
        """
        Compute the log likelihood of the sample belonging to the given class.

        Parameters:
        - class_label: class to compute likelihood for
        - sample: 1D array of feature values

        Returns:
        - Array of log likelihoods for each feature
        """
        mean = self.class_feature_means[class_label]
        variance = self.class_feature_variances[class_label]

        # Avoid division by zero in variance
        variance = np.where(variance == 0, 1e-6, variance)

        log_likelihood_numerator = -0.5 * ((sample - mean) ** 2) / variance
        log_likelihood_denominator = -0.5 * np.log(2 * np.pi * variance)

        return log_likelihood_numerator + log_likelihood_denominator

    def _predict_single_sample(self, sample):
        """
        Predict the class label for a single sample.

        Parameters:
        - sample: 1D array of feature values

        Returns:
        - Predicted class label
        """
        log_posteriors = []

        for class_label in self.classes:
            # Log prior probability
            log_prior = np.log(self.class_priors[class_label])

            # Sum of log likelihoods across features
            log_likelihood = np.sum(self._compute_log_likelihood(class_label, sample))

            # Total log posterior
            log_posterior = log_prior + log_likelihood
            log_posteriors.append(log_posterior)

        # Choose class with the highest posterior probability
        return self.classes[np.argmax(log_posteriors)]

    def predict(self, test_data):
        """
        Predict class labels for each sample in the test data.

        Parameters:
        - test_data: np.ndarray of shape (n_samples, n_features)

        Returns:
        - np.ndarray of predicted class labels
        """
        predictions = [self._predict_single_sample(sample) for sample in test_data]
        return np.array(predictions)
