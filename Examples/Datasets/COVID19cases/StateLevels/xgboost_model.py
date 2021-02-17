import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# from sklearn.model_selection import train_test_split
from global_params import *
from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *

# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t0)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    return yhat[0]


# # walk-forward validation for univariate data
# def walk_forward_validation(data, n_test):
#     predictions = list()
#     # split dataset
#     train, test = train_test_split(data, n_test)
#     # seed history with training dataset
#     history = [x for x in train]
#     # step over each time-step in the test set
#     for i in range(len(test)):
#         # split test row into input and output columns
#         testX, testy = test[i, :-1], test[i, -1]
#         # fit model on history and make a prediction
#         yhat = xgboost_forecast(history, testX)
#         # store forecast in list of predictions
#         predictions.append(yhat)
#         # add actual observation to history for the next loop
#         history.append(test[i])
#         # summarize progress
#         print(">expected=%.1f, predicted=%.1f" % (testy, yhat))
#     # estimate prediction error
#     # error = mean_absolute_error(test[:, -1], predictions)
#     mse_val = mse(test[:, -1], predictions)
#     mape_val = mape(test[:, -1], predictions)
#     rmse_val = rmse(test[:, -1], predictions)
#     r2_val = r2score(test[:, -1], predictions)
#     return mse_val, mape_val, rmse_val, r2_val, test[:, -1], predictions


# # load the dataset
# series = read_csv("daily-total-female-births.csv", header=0, index_col=0)
# values = series.values
# # transform the time series data into supervised learning
# data = series_to_supervised(values, n_in=6)
data = series_to_supervised(case_by_date_florida_np, n_in=6)

# evaluate

mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    data, round(case_by_date_florida_np.shape[0] * 0.15), xgboost_forecast
)
print('done')

# frame_pred_val(
#     y.reshape(-1),
#     array(yhat).reshape(-1),
#     # save_path=BASEPATH + "/Outputs/Models/Performances/Baselines/lstm_pred_val.csv",
# )

# plot(y, yhat, save_path=BASEPATH + "/Outputs/Images/Xgboost/forecasting.jpg")

# # plot expected vs preducted
# # pyplot.plot(y, label="Expected")
# # pyplot.plot(yhat, label="Predicted")
# # pyplot.legend()
# # # plt.savefig(BASEPATH / pathlib.Path("Outputs/Images/Xgboost/forecasting.jpg"))
# # pyplot.show()

# # DataFrame(
# #     [[mse_val, mape_val, rmse_val, r2_val]], columns=["mape", "mse", "rmse", "r2score"]
# # ).to_csv(BASEPATH + "/Outputs/Models/Performances/Baselines/xgboost_performance.csv")
# frame_performance(
#     mse_val,
#     mape_val,
#     rmse_val,
#     r2_val,
#     save_path=BASEPATH
#     + "/Outputs/Models/Performances/Baselines/xgboost_performance.csv",
# )

