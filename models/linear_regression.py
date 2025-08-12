import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        X_b = np.c_[np.ones((n_samples, 1)), X]  # add bias term
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias   