import os
import pathlib
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
import wandb

class previous_day_model:
    def __init__(self, data, state, n_in,n_out, max_steps, min_steps, is_multi_step_prediction, model_metadata_str,
            model_params_str, model_params=None):

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
        self.start_steps = self.max_steps - case_by_date_per_states_np.shape[0] + self.n_steps_in \
                + self.n_steps_out - 1

        ### dataset -> tell which state will be used for prediction
        ###data 
        data = series_to_supervised(case_by_date_per_states_np, n_in=self.n_steps_in, n_out=self.n_steps_out)
        self.split = 0.15
        self.n_test = round(case_by_date_per_states.shape[0] * self.split)

        # # NOTE: this is need to make sure that previous_val have same training and test dataset after splitting
        self.train, self.test = train_test_split(data, self.n_test)
        self.num_train_instance = self.train.shape[0]
        num_test_instance = self.test.shape[0]
        self.train, self.test = train_test_split(case_by_date_per_states_np, self.n_test)

        # print(num_train_instance)
        # print(num_test_instance)
        # print('ss')
        # print(self.train.shape)
        # print(self.test.shape)
        # exit()

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

            # # NOTE: training is not used at all, but 
            # trainX, trainy = self.train[:, :self.n_steps_in], self.train[:, -1].reshape(-1,1)
            # testX, testy = self.test[:, :-1], self.test[:,-1].reshape(-1,1)

            
            trainy = self.train[self.n_steps_out:, :].reshape(-1,1)
            trainX = self.train[:trainy.shape[0], :].reshape(-1,1)
            assert trainy.shape[0] == trainX.shape[0]

            testy = self.test[self.n_steps_out:, :]
            testX = self.test[:testy.shape[0], :]
            assert testy.shape[0] == testX.shape[0]


            # output = self.forecast_single_step(testX, testy)
            # yhat = output['yhat']

            mse_val, mape_val, rmse_val, r2_val, y, yhat, model_hist, trainy_hat_list = delta_walk_forward_validation(
               hstack([trainX, trainy]), hstack([testX, testy]), trainX, trainy, testX, testy,
               self.n_test, self,
               None,
               None,
               None,
            )

        # # print(testy - yhat)
        # print(testy.shape)
        # print(yhat.shape)
        # print('here')
        # exit()

        mse_val = mse(testy, yhat)
        mape_val = mape(testy, yhat)
        rmse_val = rmse(testy, yhat)
        r2_val = r2score(testy, yhat)

        # print(mse_val)
        # exit()

        eval_metric_df = DataFrame(
            [[mse_val, mape_val, rmse_val, r2_val]],
            columns=["mape", "mse", "rmse", "r2score"],
        )

        return testy, yhat, eval_metric_df, model_hist, trainy_hat_list


    # def fit_model(self, train, trainX, trainy):
    def init_model(self):
        pass


    def fit_model(self, train,test, trainX, trainy, testX, testy):

        trainy_hat_list = []
        # for i in range(trainX.shape[0]):
        for i in range(self.num_train_instance):
            ind = i
            trainy_hat = self.forecast_single_step(trainX[i], \
                    trainy[i])['yhat']
            trainy_hat_list = trainy_hat_list + trainy_hat
            self.plot_predict_real(ind, trainy_hat[0])
            # wandb.log({'predict vs real':trainy_hat[0]}, step=i+self.start_steps)

            # if i > trainX.shape[0]-10:
            #     print(trainy_hat)
        # exit()

        output = {
                "hist": None,
                "trainy_hat_list": trainy_hat_list,
                }

        return output

    def forecast_model(self,  testX, testy, *args):
        if self.is_multi_step_prediction:
            return self.forecast_multiple_step( testX, testy)
        else:
            # return self.forecast_single_step( testX, testy)
            return self.forecast_single_step( testX, testy, *args)

    # def mlp_forecast_multi_step(self, train, testX):
    def forecast_multiple_step(self, testy):
        raise NotImplementedError('this function need to be updated to be similar to mlp_forecast')
        # reshape_data = []
        # i = 0 
        # while i + n <= case_by_date_per_states_test.shape[0]:
        # while i + n <= self.n_test:
        #     x = case_by_date_per_states_test[i].reshape(-1)
        #     reshape_data.append([x] * n)
        #     # print(case_by_date_florida_test[i:i+7])
        #     i += 1
        # return array(reshape_data).reshape(-1, n)

    def plot_predict_real(self, ind, pred_val):
        # print(ind+self.start_steps)
        # wandb.log({'predict vs real':pred_val}, step=ind+self.start_steps+1)
        # wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps+1})
        wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps})

    # def mlp_forecast(self, train, testX):
    def forecast_single_step(self, testX, testy, *args):

        yhat = []
        yhat.append(testX[-1])

        output = {
                "yhat": yhat,
                }

        return output

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
