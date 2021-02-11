import numpy as np
from sklearn.metrics import r2_score


def mape(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100


def mse(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean((y1 - y_pred) ** 2)


def rmse(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.sqrt(np.mean((y1 - y_pred) ** 2))


def r2score(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return r2_score(y1, y_pred)
