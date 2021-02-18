import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from global_params import *
from Models.Preprocessing.us_state import *
# from Utils.preprocessing import *
# from Utils.utils import *
# from Utils.eval_funcs import *
# from Utils.plotting import *
from Utils.modelling import *

from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle


# data = series_to_supervised(case_by_date_florida_np, n_in=6)
# train, test = train_test_split(data, round(case_by_date_florida_np.shape[0] * 0.15))
# X_train, y_train = train[:,:-1], train[:, -1]
# X_test, y_test = test[:,:-1], test[:, -1]
# reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
# models, predictions = reg.fit(X_train, X_test, y_train, y_test)
# print(models)

test_mode = False

for state in all_states:

    print(f"applying previous day model to {state}...")
    case_by_date_per_states = df_by_date[df_by_date["state"] == state]

    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))

    case_by_date_per_states_train, case_by_date_per_states_test = split(
        case_by_date_per_states_np
    )

    data = series_to_supervised(case_by_date_per_states_np, n_in=6)
    train, test = train_test_split(data, round(case_by_date_per_states_np.shape[0] * 0.15))
    X_train, y_train = train[:,:-1], train[:, -1]
    X_test, y_test = test[:,:-1], test[:, -1]


    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    model, model_name = reg, "lazypredict"
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    # print(models)

    cur_model_prediction = predictions


    specified_path = None if FRAME_PERFORMANCE_PATH is None else BASEPATH + FRAME_PERFORMANCE_PATH.format(state,state, model_name)

    beta_frame_performance(cur_model_prediction, save_path=specified_path)
    
    if test_mode:
        exit()
