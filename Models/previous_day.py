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


def apply_previous_day_to_all_states(
    base_path, frame_performance_path, frame_pred_val_path, plot_path
):
    for i in all_states:
        print(f"applying previous day model to {i}...")
        case_by_date_per_states = df_by_date[df_by_date["state"] == i]

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

        frame_performance(
            mse_val,
            mape_val,
            rmse_val,
            r2_val,
            save_path=BASEPATH + frame_performance_path.format(i)
            # + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_performance.csv",
        )

        frame_pred_val(
            cur_val.reshape(-1),
            pred_val.reshape(-1),
            save_path=BASEPATH + frame_pred_val_path.format(i)
            # + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_pred_val.csv",
        )

        plot(
            cur_val,
            pred_val,
            save_path=BASEPATH + plot_path.format(i),
            # + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_forcasting.jpg",
            display=False,
        )


if __name__ == "__main__":

    apply_previous_day_to_all_states(
        BASEPATH,
        "/Outputs/Models/Performances/Baselines/{}_previous_val_performance.csv",
        "/Outputs/Models/Performances/Baselines/{}_previous_val_pred_val.csv",
        "/Outputs/Models/Performances/Baselines/{}_previous_val_forcasting.jpg",
    )

    # for i in all_states:
    #     case_by_date_per_states = df_by_date[df_by_date["state"] == i]

    #     case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
    #         "float"
    #     )
    #     case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))

    #     case_by_date_per_states_train, case_by_date_per_states_test = split(
    #         case_by_date_per_states_np
    #     )

    #     cur_val = case_by_date_per_states_test[1:]
    #     pred_val = case_by_date_per_states_test[:-1]

    #     mse_val = mse(cur_val, pred_val)
    #     mape_val = mape(cur_val, pred_val)
    #     rmse_val = rmse(cur_val, pred_val)
    #     r2_val = r2score(cur_val, pred_val)

    #     frame_performance(
    #         mse_val,
    #         mape_val,
    #         rmse_val,
    #         r2_val,
    #         save_path=BASEPATH
    #         + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_performance.csv",
    #     )

    #     frame_pred_val(
    #         cur_val.reshape(-1),
    #         pred_val.reshape(-1),
    #         save_path=BASEPATH
    #         + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_pred_val.csv",
    #     )

    #     plot(
    #         cur_val,
    #         pred_val,
    #         save_path=BASEPATH
    #         + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_forcasting.jpg",
    #     )
