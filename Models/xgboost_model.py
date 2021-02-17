
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

def xgboost_model(data, state):
    print(f"applying previous day model to {state}...")

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

    data = series_to_supervised(case_by_date_florida_np, n_in=6)

    mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
        data, round(case_by_date_florida_np.shape[0] * 0.15), xgboost_forecast
    )
    return y, yhat, mse_val, mape_val, rmse_val, r2_val

if __name__ == "__main__":
        
    apply_model_to_all_states(
        df_by_date,
        (xgboost_model, 'xgboost_model'),
        BASEPATH,
        FRAME_PERFORMANCE_PATH,
        FRAME_PRED_VAL_PATH,
        PLOT_PATH,
        test_mode=True,
    )
