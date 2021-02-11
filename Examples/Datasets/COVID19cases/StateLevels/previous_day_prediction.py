import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from global_params import *
from Utils.preprocessing import *
from Utils.utils import *
from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.plotting import *

cur_val = case_by_date_florida_train[1:]
pred_val = case_by_date_florida_train[:-1]

mse_val = mse(cur_val, pred_val)
mape_val = mape(cur_val, pred_val)
rmse_val = rmse(cur_val, pred_val)
r2_val = r2score(cur_val, pred_val)

# DataFrame(
#     [[mse_val, mape_val, rmse_val, r2_val]], columns=["mape", "mse", "rmse", "r2score"]
# ).to_csv(
#     BASEPATH + "/Outputs/Models/Performances/Baselines/previous_val_performance.csv"
# )

frame_performance(
    mse_val,
    mape_val,
    rmse_val,
    r2_val,
    # save_path=BASEPATH
    # + "/Outputs/Models/Performances/Baselines/previous_val_performance.csv",
)


frame_pred_val(
    cur_val.reshape(-1),
    pred_val.reshape(-1),
    save_path=BASEPATH
    + "/Outputs/Models/Performances/Baselines/previous_val_pred_val.csv",
)

plot(
    cur_val,
    pred_val,
    # save_path=BASEPATH
    # + "/Outputs/Models/Performances/Baselines/previous_val_forcasting.jpg",
)
