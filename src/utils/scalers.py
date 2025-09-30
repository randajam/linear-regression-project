import numpy as np

class MyMinMaxScaler():
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = self.max_ - self.min_
        return self
    
    def transform(self, X):
        if self.min_ is None or self.max_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet!")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        a, b = self.feature_range
        X_scaled = (X - self.min_) / np.where(self.scale_ == 0, 1, self.scale_)
        return a + X_scaled * (b - a)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled, dtype=float)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        a, b = self.feature_range
        return self.min_ + (X_scaled - a) * self.scale_ / (b - a)
    
class MyStandardScaler():
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)

        return self
    
    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet!")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return (X - self.mean_) / np.where(self.std_ == 0, 1, self.std_)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

    def inverse_transform(self, X_scaled):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet!")

        X_scaled = np.asarray(X_scaled, dtype=float)
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)

        return X_scaled * self.scale_ + self.mean_