import numpy as np
from math import log

def weighted_mae(y_true, y_pred, weights=None):
    """
    Weighted Mean Absolute Error
    """
    if weights:
        return np.mean(weights * np.abs(y_true - y_pred))
    else:
        return np.mean(np.abs(y_true - y_pred))

def naive_return(price2, price1):
    return (price2 - price1) / price1

def log_return(price2, price1):
    return log(price2 / price1)

def log_prices(returns):
    return returns.cumsum()

