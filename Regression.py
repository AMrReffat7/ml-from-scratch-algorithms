import numpy as np
import math


class BaseRegression:
    def __init__(self, n_iterations, learning_rate):
        # Number of training epochs
        self.n_iterations = n_iterations
        # Step size for gradient descent
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        # Initialize weights uniformly in [-1/sqrt(d), +1/sqrt(d)]
        limit = 1 / math.sqrt(n_features)
        self.weights = np.random.uniform(-limit, limit, size=(n_features,))

    def fit(self, X, y, l2_lambda=0.01):
        """
        Gradient descent with optional L2 regularization
        """
        # Add bias term
        X_bias = np.insert(X, 0, 1, axis=1)

        # Initialize weights
        self.training_errors = []
        self.initialize_weights(n_features=X_bias.shape[1])

        for iteration in range(self.n_iterations):
            # Predictions
            predictions = X_bias.dot(self.weights)

            # L2 penalty
            l2_penalty = (l2_lambda / 2) * np.sum(self.weights**2)

            # MSE + L2 penalty
            mse = np.mean(0.5 * (y - predictions) ** 2 + l2_penalty)
            self.training_errors.append(mse)

            # Gradient of MSE
            gradient_mse = -(y - predictions).dot(X_bias)

            # Gradient of L2 penalty
            gradient_l2 = l2_lambda * self.weights

            # Total gradient
            total_gradient = gradient_mse + gradient_l2

            # Update weights
            self.weights -= self.learning_rate * total_gradient

    def fit_l1(self, X, y, l1_lambda=0.01):
        """
        Gradient descent with L1 regularization (Lasso)
        """
        # Add bias term
        X_bias = np.insert(X, 0, 1, axis=1)

        # Initialize weights
        self.training_errors = []
        self.initialize_weights(n_features=X_bias.shape[1])

        for iteration in range(self.n_iterations):
            # Predictions
            predictions = X_bias.dot(self.weights)

            # L1 penalty
            l1_penalty = l1_lambda * np.sum(np.abs(self.weights))

            # MSE + L1 penalty
            mse = np.mean(0.5 * (y - predictions) ** 2 + l1_penalty)
            self.training_errors.append(mse)

            # Gradient of MSE
            gradient_mse = -(y - predictions).dot(X_bias)

            # Subgradient of L1 penalty
            gradient_l1 = l1_lambda * np.sign(self.weights)

            # Total gradient
            total_gradient = gradient_mse + gradient_l1

            # Update weights
            self.weights -= self.learning_rate * total_gradient

    def predict(self, X):
        X_bias = np.insert(X, 0, 1, axis=1)
        return X_bias.dot(self.weights)
