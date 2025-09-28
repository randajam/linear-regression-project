import pandas as pd
import numpy as np
import os

from .metrics import get_metrics

class ModelComparator:
    def __init__(self, X_train, y_train, X_test, y_test, scaler=None, results_path="../results"):
        self.scaler = scaler
        if scaler:
            self.X_train = self.scaler.fit_transform(X_train)
            self.X_test = self.scaler.transform(X_test)
        else:
            self.X_train = np.asarray(X_train)
            self.X_test = np.asarray(X_test)

        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)

        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)

        self.files = {
            "mae" : os.path.join(results_path, "results_mae.csv"),
            "rmse" : os.path.join(results_path, "results_rmse.csv"),
            "r2" : os.path.join(results_path, "results_r2.csv")
        }

        self.results_mae = self._load_or_create(self.files["mae"])
        self.results_rmse = self._load_or_create(self.files["rmse"])
        self.results_r2 = self._load_or_create(self.files["r2"])

    def _load_or_create(self, filepath):
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            return pd.DataFrame(columns=["model", "train", "test"])

    def evaluate_model(self, model, model_name):
        model.fit(self.X_train, self.y_train)
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        metrics_train = get_metrics(self.y_train, y_train_pred)
        metrics_test = get_metrics(self.y_test, y_test_pred)

        self._update_results(
            self.results_mae,
            model_name,
            metrics_train["mae"],
            metrics_test["mae"]
        )

        self._update_results(
            self.results_rmse,
            model_name,
            metrics_train["rmse"],
            metrics_test["rmse"]
        )

        self._update_results(
            self.results_r2,
            model_name,
            metrics_train["r2"],
            metrics_test["r2"]
        )

    def _update_results(self, df, model_name, train_val, test_val):
        if model_name in df["model"].values:
            df.loc[df["model"] == model_name, ["train", "test"]] = [train_val, test_val]
        else:
            df.loc[len(df)] = [model_name, train_val, test_val]

    def compare_with_library(self, my_model, lib_model):
        my_model.fit(self.X_train, self.y_train)
        lib_model.fit(self.X_train, self.y_train)

        my_pred = my_model.predict(self.X_test)
        lib_pred = lib_model.predict(self.X_test)

        my_metrics = get_metrics(self.y_test, my_pred)
        lib_metrics = get_metrics(self.y_test, lib_pred)

        print("=====MAE=====")
        print("My Model:", my_metrics["mae"])   
        print("Lib Model:", lib_metrics["mae"])
        print("=====RMSE=====")
        print("My Model:", my_metrics["rmse"])   
        print("Lib Model:", lib_metrics["rmse"])
        print("=====R2=====")
        print("My Model:", my_metrics["r2"])   
        print("Lib Model:", lib_metrics["r2"])
        print()

    def get_results(self):
        return self.results_mae, self.results_rmse, self.results_r2
    
    def save_results(self):
        self.results_mae.to_csv(self.files["mae"], index=False)
        self.results_rmse.to_csv(self.files["rmse"], index=False)
        self.results_r2.to_csv(self.files["r2"], index=False)