import numpy as np

class BaseRegularization():
    def __init__(self, lr=0.01, n_iters=100, random_state=21, alpha=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.random_state = random_state
        self.alpha = alpha
        self.weights_ = None
        self.bias_ = None

    def predict(self, X):
        if self.weights_ is None or self.bias_ is None:
            raise Exception("Model not trained yet!")
        else:
            return X @ self.weights_ + self.bias_

class MyRidge(BaseRegularization):
    def __init__(self, lr=0.01, n_iters=100, random_state=21, alpha=0.1):
        super().__init__(lr, n_iters, random_state, alpha)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0

        for _ in range(self.n_iters):
            for i in rng.permutation(n_samples):
                Xi, yi = X[i], y[i]
                y_pred = Xi @ self.weights_ + self.bias_
                error = yi - y_pred

                dw = error * Xi - self.alpha * self.weights_
                db = error

                self.weights_ += self.lr * dw
                self.bias_ += self.lr * db

        return self
    
class MyLasso(BaseRegularization):
    def __init__(self, lr=0.01, n_iters=100, random_state=21, alpha=0.1):
        super().__init__(lr, n_iters, random_state, alpha)    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0

        for _ in range(self.n_iters):
            for i in rng.permutation(n_samples):
                Xi, yi = X[i], y[i]
                y_pred = Xi @ self.weights_ + self.bias_
                error = yi - y_pred

                dw = error * Xi - self.alpha * np.sign(self.weights_)
                db = error

                self.weights_ += self.lr * dw
                self.bias_ += self.lr * db

        return self
    
class MyElasticNet(BaseRegularization):
    def __init__(self, lr=0.01, n_iters=100, random_state=21, alpha=0.1, l1_ratio=0.5):
        super().__init__(lr, n_iters, random_state, alpha)
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0

        for _ in range(self.n_iters):
            for i in rng.permutation(n_samples):
                Xi, yi = X[i], y[i]
                y_pred = Xi @ self.weights_ + self.bias_
                error = yi - y_pred

                dw = error * Xi - self.alpha * (self.l1_ratio * self.weights_ +
                                                (1 - self.l1_ratio) * np.sign(self.weights_))
                db = error

                self.weights_ += self.lr * dw
                self.bias_ += self.lr * db

        return self