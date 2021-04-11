import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from global_params import *
# from Models.Preprocessing.us_state import *
# from Utils.preprocessing import *
# from Utils.utils import *
# from Utils.eval_funcs import *
# from Utils.plotting import *
from Utils.modelling import *
from Utils.eval_funcs import *
from Utils.cli import * 


def previous_day_model(data, state, n_in, n_out, is_multi_step_prediction):
    print(f"applying previous day model to {state}...")
    case_by_date_per_states = data[data["state"] == state]

    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))

    case_by_date_per_states_train, case_by_date_per_states_test = split(
        case_by_date_per_states_np
    )

    def previous_day_forcast_multi_step(n):
        reshape_data = []
        i = 0 
        while i + n <= case_by_date_per_states_test.shape[0]:
            x = case_by_date_per_states_test[i].reshape(-1)
            reshape_data.append([x] * n)
            # print(case_by_date_florida_test[i:i+7])
            i += 1
        return array(reshape_data).reshape(-1, n)

    def previous_day_forcast(n):
        reshape_data = []
        i = 0 
        while i + n <= case_by_date_per_states_test.shape[0]:
            x = case_by_date_per_states_test[i].reshape(-1)
            reshape_data.append([x])
            # print(case_by_date_florida_test[i:i+7])
            i += 1
        return array(reshape_data).reshape(-1)

    # cur_val = case_by_date_per_states_test[1:]
    # pred_val = case_by_date_per_states_test[:-1]
    cur_val = split_by_predict_next_n_day(case_by_date_per_states_test, n_out)
    if is_multi_step_prediction:
        pred_val = previous_day_forcast_multi_step(n_out)
    else:
        cur_val = cur_val[:, -1]
        pred_val = previous_day_forcast(n_out)

    mse_val = mse(cur_val, pred_val)
    mape_val = mape(cur_val, pred_val)
    rmse_val = rmse(cur_val, pred_val)
    r2_val = r2score(cur_val, pred_val)

    eval_metric_df = DataFrame(
        [[mse_val, mape_val, rmse_val, r2_val]],
        columns=["mape", "mse", "rmse", "r2score"],
    )

    return cur_val, pred_val, eval_metric_df

if __name__ == "__main__":
        
    non_cli_params = {
        'data': df_by_date,
        'model' : (previous_day_model, 'previous_val'),
        'base_path' : BASEPATH,
        'frame_performance_path' : FRAME_PERFORMANCE_PATH,
        'frame_pred_val_path' : FRAME_PRED_VAL_PATH,
        'plot_path' : PLOT_PATH,
    }

#     gamma_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
