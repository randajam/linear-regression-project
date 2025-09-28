import numpy as np

def my_mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    return np.mean(np.abs(y_true - y_pred))
    
def my_rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
def my_r2_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)

def get_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    else:
        return {
            "mae": my_mae(y_true, y_pred),
            "rmse": my_rmse(y_true, y_pred),
            "r2": my_r2_score(y_true, y_pred)
        }
    