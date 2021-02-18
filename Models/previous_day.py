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

def previous_day_model(data, state):
    print(f"applying previous day model to {state}...")
    case_by_date_per_states = data[data["state"] == state]

    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))

    case_by_date_per_states_train, case_by_date_per_states_test = split(
        case_by_date_per_states_np
    )

    cur_val = case_by_date_per_states_test[1:]
    pred_val = case_by_date_per_states_test[:-1]
    mse_val = mse(cur_val, pred_val)
    mape_val = mape(cur_val, pred_val)
    rmse_val = rmse(cur_val, pred_val)
    r2_val = r2score(cur_val, pred_val)
    return cur_val, pred_val, mse_val, mape_val, rmse_val, r2_val


if __name__ == "__main__":
        
    apply_model_to_all_states(
        df_by_date,
        (previous_day_model, 'previous_val'),
        BASEPATH,
        FRAME_PERFORMANCE_PATH,
        FRAME_PRED_VAL_PATH,
        PLOT_PATH,
        test_mode=False,
    )

