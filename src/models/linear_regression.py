import numpy as np

class BaseLinearRegression:
    def __init__(self, lr=0.01, n_iters=100, random_state=21):
        self.lr = lr
        self.n_iters = n_iters
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None

    def predict(self, X):
        if self.weights_ is None or self.bias_ is None:
            raise Exception("Model not trained yet!")
        else:
            return X @ self.weights_ + self.bias_
        
class LinearRegressionAnalytic(BaseLinearRegression):
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.bias_ = theta[0]
        self.weights_  = theta[1:]

        return self

class LinearRegressionGD(BaseLinearRegression):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        rng = np.random.default_rng(self.random_state)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights_ + self.bias_
            error = y_pred - y

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db
        
        return self
    
class LinearRegressionSGD(BaseLinearRegression):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0

        for _ in range(self.n_iters):
            for i in rng.permutation(n_samples):
                Xi, yi = X[i], y[i]

                y_pred = Xi @ self.weights_ + self.bias_
                error = yi - y_pred

                self.weights_ += self.lr * (error * Xi)
                self.bias_ += self.lr * error

        return self
