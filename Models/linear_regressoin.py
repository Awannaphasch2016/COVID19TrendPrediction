import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from numpy import asarray
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from numpy import array

from global_params import *
# from Models.Preprocessing.us_state import *
# from Utils.eval_funcs import *
# from Utils.preprocessing import *
# from Utils.utils import *
from Utils.modelling import *

def linear_regression_model(data, state):
    print(f"applying previous day model to {state}...")
    def linear_regression_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = LinearRegression()
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict(asarray([testX]))
        return yhat[0]

    # data = series_to_supervised(case_by_date_florida_np, n_in=6)
    # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    #     data, round(case_by_date_florida_np.shape[0] * 0.15), linear_regression_forecast
    # )

    case_by_date_per_states = data[data["state"] == state]
    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    data = series_to_supervised(case_by_date_per_states_np, n_in=6)
    mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
        data, round(case_by_date_florida_np.shape[0] * 0.15), linear_regression_forecast
    )

    return y, yhat, mse_val, mape_val, rmse_val, r2_val


if __name__ == "__main__":
    
    apply_model_to_all_states(
        df_by_date,
        (linear_regression_model, 'linear_regression'),
        BASEPATH,
        FRAME_PERFORMANCE_PATH,
        FRAME_PRED_VAL_PATH,
        PLOT_PATH,
        test_mode=False,
    )
