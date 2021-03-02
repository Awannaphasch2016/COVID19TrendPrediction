
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from numpy import array, asarray


from global_params import *
# from Models.Preprocessing.us_state import *
# from Utils.preprocessing import *
# from Utils.utils import *
# from Utils.eval_funcs import *
# from Utils.plotting import *
from Utils.modelling import *

# from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def xgboost_model(data, state, n_in, n_out=1):
    assert n_out == 1, 'ValueError: xgboost_model only support single output'
    print(f"applying previous day model to {state}...")

    # fit an xgboost model and make a one step prediction
    def xgboost_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        # trainX, trainy = train[:, :-1], train[:, -1]
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        # fit model
        model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict(asarray([testX]))
        # return yhat[0]
        return yhat.reshape(-1)


    # data = series_to_supervised(case_by_date_florida_np, n_in=6)
    # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    #     data, round(case_by_date_florida_np.shape[0] * 0.15), xgboost_forecast
    # )
    n_steps_in, n_steps_out = n_in, n_out
    case_by_date_per_states = data[data["state"] == state]
    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    # data = series_to_supervised(case_by_date_per_states_np, n_in=6)
    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)
    # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    #     data, round(case_by_date_florida_np.shape[0] * 0.15), xgboost_forecast
    # )

    mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
        data, round(case_by_date_florida_np.shape[0] * 0.15), xgboost_forecast, n_steps_in, n_steps_out
    )

    eval_metric_df = DataFrame(
        [[mse_val, mape_val, rmse_val, r2_val]],
        columns=["mape", "mse", "rmse", "r2score"],
    )
    # print(eval_metric_df)
    # exit()

    # return y, yhat, mse_val, mape_val, rmse_val, r2_val
    return y, yhat, eval_metric_df

if __name__ == "__main__":
        
    # apply_model_to_all_states(
    #     df_by_date,
    #     (xgboost_model, 'xgboost_model'),
    #     BASEPATH,
    #     FRAME_PERFORMANCE_PATH,
    #     FRAME_PRED_VAL_PATH,
    #     PLOT_PATH,
    #     test_mode=False,
    #     # test_mode=True,
    # )

    beta_apply_model_to_all_states(
        df_by_date,
        (xgboost_model, 'xgboost_model'),
        6,
        1,
        BASEPATH,
        FRAME_PERFORMANCE_PATH,
        FRAME_PRED_VAL_PATH,
        PLOT_PATH,
        test_mode=False,
        # test_mode=True,
    )
