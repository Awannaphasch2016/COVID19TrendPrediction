import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from numpy import asarray
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy import array

from global_params import *
from Models.Preprocessing.us_state import *
# from Utils.eval_funcs import *
# from Utils.preprocessing import *
# from Utils.utils import *
from Utils.modelling import *

def linear_regression_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # trainX, trainy = train[1:, :], train[:-1,:]
    # fit model
    model = LinearRegression()
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(asarray([testX]))
    # yhat = model.predict(asarray([testX]).reshape(-1,1))
    return yhat[0]

data = series_to_supervised(case_by_date_florida_np, n_in=6)

mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
 data, round(case_by_date_florida_np.shape[0] * 0.15), linear_regression_forecast
)


frame_performance(
    mse_val,
    mape_val,
    rmse_val,
    r2_val,
    # save_path=BASEPATH
    # + "/Outputs/Models/Performances/Baselines/linear_regression_performance.csv",
)

frame_pred_val(
    y.reshape(-1),
    array(yhat).reshape(-1),
    # save_path=BASEPATH
    # + "/Outputs/Models/Performances/Baselines/linear_regression_pred_val.csv",
)

plot(
    y,
    yhat,
    # save_path=BASEPATH
    # + "/Outputs/Images/LinearRegression/linear_regression_forecasting.jpg",
)
