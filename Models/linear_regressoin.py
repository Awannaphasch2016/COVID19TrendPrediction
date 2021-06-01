import os
import pathlib

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
import wandb

class linear_regression_model:
    def __init__(self, data, state, n_in,n_out, max_steps, min_steps, is_multi_step_prediction, model_metadata_str,
            model_params_str, model_params=None):
    # def __init__(self, data, state, n_in,n_out, max_steps, min_steps, is_multi_step_prediction, model_metadata_str,
    #         model_params_str, model_params=None, config_kwargs=None):

        # self.config_kwargs                    = config_kwargs

        self.is_multi_step_prediction = is_multi_step_prediction
        self.model_metadata_str       = model_metadata_str
        self.model_params_str         = model_params_str
        self.state                    = state
        self.max_steps                = max_steps
        self.min_steps                = min_steps

        # epoch = model_params['epoch']
        self.multi_step_folder                  = model_params.multi_step_folder
        self.model_name                         = model_params.model_name
        self.dataset_name                       = model_params.dataset
        self.experiment_id                      = int(model_params.experiment_id)
        
        self.n_steps_in, self.n_steps_out = n_in, n_out
        self.prep_data()
        # self.fit_model()

    def prep_data(self):
        # TMP:
        ###### get input data
        tmp = df_by_date

        case_by_date_per_states = tmp[tmp["state"] == self.state]
        case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)
        case_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")
        case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))


        if self.experiment_id == 0:
            pass 
        elif self.experiment_id == 1:
            # print(case_by_date_per_states_np.shape)
            case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
            case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
            # print(case_by_date_per_states_np.shape)
        elif self.experiment_id == 2:
            case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
            case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
            case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
            case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        elif self.experiment_id == 3:
            case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
            case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
            case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
            case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
            case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
            case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        else:
            raise NotImplementedError

        self.start_steps = self.max_steps - case_by_date_per_states_np.shape[0] + self.n_steps_in + \
         self.n_steps_out - 1

        # print(case_by_date_per_states_np.shape[0])
        # print('xxxxx')
        # exit()


        ### dataset -> tell which state will be used for prediction
        ###data 
        data = series_to_supervised(case_by_date_per_states_np, n_in=self.n_steps_in, n_out=self.n_steps_out)
        self.split = 0.15
        self.n_test = round(case_by_date_per_states.shape[0] * self.split)
        self.train, self.test = train_test_split(data, self.n_test)
        
        
        # print(data.shape)
        # print(self.train.shape)
        # print(self.test.shape)
        # print('done')
        # exit()

    def forecast(self):

        if self.is_multi_step_prediction:
            raise NotImplementedError("update to have the same structure as else condition.")
            # yhat = self.forecast_multiple_step(testy)

        else:
            # testy = self.test[:,:-1]
            # testX, testy = self.test[:, :-1], self.test[:,-1]
            trainX, trainy = self.train[:, :self.n_steps_in], self.train[:, -1].reshape(-1,1)
            testX, testy = self.test[:, :self.n_steps_in], self.test[:, -1].reshape(-1,1)
            mse_val, mape_val, rmse_val, r2_val, y, yhat, model_hist, trainy_hat_list  = delta_walk_forward_validation(
               hstack([trainX, trainy]), hstack([testX, testy]), trainX, trainy, testX, testy,
               self.n_test, self,
               None,
               None,
               None,
            )

        # print(testy-yhat)
        # print('yayyyy')
        # exit()

        mse_val = mse(testy, yhat)
        mape_val = mape(testy, yhat)
        rmse_val = rmse(testy, yhat)
        r2_val = r2score(testy, yhat)


        eval_metric_df = DataFrame(
            [[mse_val, mape_val, rmse_val, r2_val]],
            columns=["mape", "mse", "rmse", "r2score"],
        )

        # print(eval_metric_df)
        # print('ddddddddddddddddddddddddd')

        return testy, yhat, eval_metric_df, model_hist, trainy_hat_list


    # def fit_model(self, train, trainX, trainy):
    def init_model(self):
        self.model = LinearRegression()

    def fit_model(self, train,test, trainX, trainy, testX, testy):
        self.model.fit(trainX, trainy)

        trainy_hat_list = []
        for i in range(trainX.shape[0]):
            ind = i
            trainy_hat = self.forecast_single_step(trainX[i], \
                    trainy[i])['yhat'].reshape(-1).tolist()
            trainy_hat_list = trainy_hat_list + trainy_hat
            self.plot_predict_real(ind, trainy_hat[0])
            # wandb.log({'predict vs real':trainy_hat[0]}, step=i+self.start_steps)

        output = {
                "trainy_hat_list": trainy_hat_list,
                }

        return output


        return 

    def forecast_model(self,  testX, testy):
        if self.is_multi_step_prediction:
            return self.forecast_multiple_step( testX, testy)
        else:
            # return self.forecast_single_step( testX, testy)
            return self.forecast_single_step( testX, testy)

    # def mlp_forecast_multi_step(self, train, testX):
    def forecast_multiple_step(self, testy):
        raise NotImplementedError('this function need to be updated to be similar to mlp_forecast')

    def plot_predict_real(self, ind, pred_val):
        # wandb.log({'predict vs real':pred_val}, step=ind+self.start_steps)
        # print(ind+self.start_steps+1)
        # print('ssssssssssssssss')
        # wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps+1})
        wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps})

    # def mlp_forecast(self, train, testX):
    def forecast_single_step(self, testX, testy):

        yhat = self.model.predict(asarray([testX]))
        # yhat = np.array(yhat)
        
        # # print(yhat, testy)
        # print(yhat - testy)
        # print('xxxxxxxxxxxxxxxxxxxxxx')

        output = {
                "yhat": yhat,
                }

        return output

# def linear_regression_model(data, state, n_in, n_out, is_multi_step_prediction):
#     print(f"applying previous day model to {state}...")

#     def linear_regression_forecast_multi_step(train, testX):
#         raise NotImplementedError
#         # transform list into array
#         train = asarray(train)
#         # split into input and output columns
#         trainX, trainy = train[:, :-1], train[:, -1]
#         # fit model
#         model = LinearRegression()
#         model.fit(trainX, trainy)
#         # make a one-step prediction
#         yhat = model.predict(asarray([testX]))
#         return yhat[0]

#     def linear_regression_forecast(train, testX):
#         # transform list into array
#         train = asarray(train)
#         # split into input and output columns
#         # trainX, trainy = train[:, :-1], train[:, -1]
#         trainX, trainy = train[:, :n_in], train[:, -1]
#         # fit model
#         model = LinearRegression()
#         model.fit(trainX, trainy)
#         # make a one-step prediction
#         yhat = model.predict(asarray([testX]))
#         output = {
#                 "yhat": yhat.reshape(-1),
#                 }
#         # return yhat[0]
#         # return yhat.reshape(-1)
#         return output

#     # data = series_to_supervised(case_by_date_florida_np, n_in=6)
#     # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
#     #     data, round(case_by_date_florida_np.shape[0] * 0.15), linear_regression_forecast
#     # )

#     n_steps_in, n_steps_out = n_in, n_out
#     case_by_date_per_states = data[data["state"] == state]
#     case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
#         "float"
#     )
#     case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
#     data = series_to_supervised(case_by_date_per_states_np, n_in=6)

#     # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
#     #     data, round(case_by_date_florida_np.shape[0] * 0.15), linear_regression_forecast
#     # )

    
#     data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)
#     n_test = round(case_by_date_florida_np.shape[0] * 0.15)
#     train, test = train_test_split(data, n_test)
#     if is_multi_step_prediction:
#         testX, testy = test[:, :n_steps_in], test[:, -n_steps_out:]
#         mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
#             train, test, testX, testy, n_test, linear_regression_forecast_multi_step
#         )
#     else:
#         trainX, trainy = train[:, :n_steps_in], train[:, -1].reshape(-1,1)
#         testX, testy = test[:, :n_steps_in], test[:, -1].reshape(-1,1)
#         mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
#             hstack([trainX, trainy]), hstack([testX, testy]), testX, testy, n_test, linear_regression_forecast
#         )

#     eval_metric_df = DataFrame(
#         [[mse_val, mape_val, rmse_val, r2_val]],
#         columns=["mape", "mse", "rmse", "r2score"],
#     )
#     return y, yhat, eval_metric_df



if __name__ == "__main__":

    non_cli_params = {
        'data': df_by_date,
        'model' : (linear_regression_model, 'linear_regression'),
        'base_path' : BASEPATH,
        'frame_performance_path' : FRAME_PERFORMANCE_PATH,
        'frame_pred_val_path' : FRAME_PRED_VAL_PATH,
        'plot_path' : PLOT_PATH,
    }

    # gamma_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
