import numpy as np
"""
Weighted Mean Absolute Error
"""
def weighted_mae(y_true, y_pred, weights):
    return np.mean(weights * np.abs(y_true - y_pred))
