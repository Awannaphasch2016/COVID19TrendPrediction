"""reference: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/"""


# univariate mlp example
from numpy import array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import data_adapter

from numpy import asarray
from numpy import array
from numpy import hstack
import numpy as np

from global_params import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *
from Models.Preprocessing.us_state import * 

import click
from wandb.keras import WandbCallback
import wandb
import sys

class mlp_model:
    # def __init__(self, data, state, n_in,n_out, max_steps, min_steps, is_multi_step_prediction,
    #         model_metadata_str, model_params_str, model_params=None):
    def __init__(self, data, state, n_in,n_out, max_steps, min_steps, is_multi_step_prediction,
             model_params=None):
        assert 'epoch' in list(model_params.keys())

        self.is_multi_step_prediction = is_multi_step_prediction
        # self.model_metadata_str       = model_metadata_str
        # self.model_params_str         = model_params_str
        self.state                    = state
        self.max_steps                = max_steps
        self.min_steps                = min_steps

        # epoch = model_params['epoch']
        self.epoch                              = model_params.epoch
        self.multi_step_folder                  = model_params.multi_step_folder
        self.model_name                         = model_params.model_name
        self.dataset_name                       = model_params.dataset
        self.train_model_with_1_run             = model_params.train_model_with_1_run
        self.dont_create_new_model_on_each_run  = model_params.dont_create_new_model_on_each_run
        self.evaluate_on_many_test_data_per_run = model_params.evaluate_on_many_test_data_per_run
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
        elif self.experiment_id == 4:
            tmp = np.arange(1000).reshape(-1,1)
            case_by_date_per_states = tmp 
            case_by_date_per_states_np = tmp 
            # print(case_by_date_per_states_np)
            # exit()
        else:
            raise NotImplementedError

        # print(case_by_date_per_states_np[:10])
        # print(case_by_date_per_states_np[-10:])
        # print('yi')
        # # exit()

        # self.start_steps = self.max_steps - case_by_date_per_states_np.shape[0] + self.n_steps_in
        self.start_steps = self.max_steps - case_by_date_per_states_np.shape[0] + self.n_steps_in + \
         self.n_steps_out - 1 
        

        # print(self.start_steps)
        # print(self.max_steps )
        # print(case_by_date_per_states_np.shape[0] )
        # print(self.n_steps_in)
        # exit()

        assert case_by_date_per_states_np.shape[0] == case_by_date_per_states.shape[0]
        ### dataset -> tell which state will be used for prediction
        ###data 
        data = series_to_supervised(case_by_date_per_states_np, n_in=self.n_steps_in, n_out=self.n_steps_out)

        # print(data.shape)
        # print('sssss')
        # exit()

        self.split = 0.15
        self.n_test = round(case_by_date_per_states.shape[0] * self.split)
        self.train, self.test = train_test_split(data, self.n_test)

    def forecast(self):
        if self.is_multi_step_prediction:
            raise NotImplementedError("update to have the same structure as else condition.")
            testX, testy = self.test[:, :self.n_steps_in], self.test[:, -self.n_steps_out:]
            mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
                self.train, self.test, testX, testy, self.n_test, self.mlp_forecast_multi_step
            )
        else:
            trainX, trainy = self.train[:, :self.n_steps_in], self.train[:, -1].reshape(-1,1)
            testX, testy = self.test[:, :self.n_steps_in], self.test[:, -1].reshape(-1,1)

            # print(trainX.shape, testX.shape)
            # print(trainy.shape, testy.shape)
            # exit()

            # mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
            #    hstack([self.trainX, self.trainy]), hstack([testX, testy]), testX, testy, self.n_test,
            #    self.mlp_forecast
            # )

            mse_val, mape_val, rmse_val, r2_val, y, yhat, model_hist, trainy_hat_list = delta_walk_forward_validation(
               hstack([trainX, trainy]), hstack([testX, testy]), trainX, trainy, testX, testy,
               self.n_test, self, self.train_model_with_1_run,
               self.dont_create_new_model_on_each_run,
               self.evaluate_on_many_test_data_per_run
            )

        eval_metric_df = DataFrame(
            [[mse_val, mape_val, rmse_val, r2_val]],
            columns=["mape", "mse", "rmse", "r2score"],
        )
        return y, yhat, eval_metric_df, model_hist, trainy_hat_list


    # def fit_model(self, train, trainX, trainy):
    def init_model(self):


        self.model = Sequential(
            [
                Dense(100, activation="relu", input_dim=self.n_steps_in),
                Dense(1),
            ]
        )

        # specified_path = None if CHECKPOINTS_PATH is None else BASEPATH + \
        #     CHECKPOINTS_PATH.format(self.multi_step_folder,self.n_steps_out, self.n_steps_in, self.state ,
        #             self. dataset_name, self.state, self.model_name)
        # specified_path = add_file_suffix(specified_path, self.model_metadata_str + self.model_params_str)
        # parent_dir = '/'.join(specified_path.split('/')[:-1])
        # Path(parent_dir).mkdir(parents=True,exist_ok=True)

        # checkpoint_filepath = specified_path
        # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        #     filepath          = checkpoint_filepath,
        #     save_weights_only = False,
        #     save_freq = 'epoch',
        #     period=10
        #     )

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

        # def make_print_data_and_train_step(keras_model):
        #     original_train_step = keras_model.train_step
        #     def print_data_and_train_step(original_data):
        #         # Basically copied one-to-one from https://git.io/JvDTv
        #         data = data_adapter.expand_1d(original_data)
        #         x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
        #         y_pred = keras_model(x, training=True)
        #         # print(y_pred)
        #         # print(K.shape(y_pred))
        #         # print(K.int_shape(y_pred))
        #         # print(y_pred.shape.as_list())
        #         # print(type(y_pred))
        #         # print(tf.shape(y_pred))
        #         # print(tf.print(y_pred, output_stream=sys.stderr))
        #         # print(y_pred.numpy())
        #         # print(y_pred.initializer)
        #         # print(y_pred.eval(session=tf.compat.v1.Session()))
        #         # print(K.eval(y_pred))
        #         # # print(tf.__version__)
        #         # print(tf.executing_eagerly())
        #         # print('------')
        #         # exit()
        #         # # this is pretty much like on_train_batch_begin
        #         # K.print_tensor(w, "Sample weight (w) =")
        #         # K.print_tensor(x, "Batch input (x) =")
        #         # K.print_tensor(y_true, "Batch output (y_true) =")
        #         # K.print_tensor(y_pred, "Prediction (y_pred) =")
        #         result = original_train_step(original_data)
        #         # add anything here for on_train_batch_end-like behavior
        #         return result
        #     return print_data_and_train_step
        # self.model.train_step = make_print_data_and_train_step(self.model)

        self.model.compile(optimizer="adam", loss="mse")
        # self.specified_path = specified_path
        self.es = es
        # self.model_checkpoint_callback = model_checkpoint_callback


    def fit_model(self, train,test, trainX, trainy, testX, testy):

        # specified_path            = self.specified_path 
        es                        = self.es 
        # model_checkpoint_callback = self.model_checkpoint_callback 

        # transform list into array
        train = asarray(train)

        # split into input and output columns
        trainX, trainy = train[:, :self.n_steps_in], train[:, -1].reshape(-1,1)
    
        # print(trainX.shape, testX.shape)
        # print(trainy.shape, testy.shape)
        # print('---------')

        # fit model
        # model_hist = self.model.fit(trainX, trainy, epochs=self.epoch, verbose=1,
        #         # validation_split=self.split,
        #         callbacks=[WandbCallback(), model_checkpoint_callback, es])
        # model_hist = self.model.fit(trainX, trainy, epochs=self.epoch, verbose=1, callbacks=[WandbCallback(),
        #     model_checkpoint_callback])

        # fit model/get test loss by using test data as "validation" data since data is too little and we
        #  don't use validation set 
        #   

        # print(trainX.shape)
        # print('h')
        # exit()

        model_hist = self.model.fit(np.vstack([trainX, testX]), np.vstack([trainy,testy]), epochs=self.epoch,
                verbose=1,
                validation_split=self.split,
                # callbacks=[WandbCallback(), model_checkpoint_callback,PredictionHistory() ])
                callbacks=[WandbCallback(), es])

        # for _ in range(self.epoch):
        #     model_hist = self.model.fit(np.vstack([trainX, testX]), np.vstack([trainy,testy]), epochs=1,
        #             verbose=1,
        #             validation_split=self.split,
        #             # callbacks=[WandbCallback(), model_checkpoint_callback,PredictionHistory() ])
        #             callbacks=[WandbCallback(), model_checkpoint_callback])

        trainy_hat_list = []
        for i in range(trainX.shape[0]):
            ind = i
            trainy_hat = self.forecast_single_step(trainX[i], \
                    trainy[i], ind, trainX)['yhat'].reshape(-1).tolist()
            trainy_hat_list = trainy_hat_list + trainy_hat
            self.plot_predict_real(ind, trainy_hat[0])
            # wandb.log({'predict vs real':trainy_hat[0]}, step=i+self.start_steps)

        # last_loss_val = model_hist.history['loss'][-1]
        loss_vals = model_hist.history['loss']
        n_loss = len(model_hist.history['loss'])

        # wandb.log({'last_window_step_loss': last_loss_val})

        # # NOTE: not sure what this is exactly trying to do, when do i need it?
        # if not self.train_model_with_1_run:
        #     wandb.log({'loss_per_run (avg over number of loss)': sum(loss_vals)/n_loss})

        # return model, hist
        # return hist

        output = {
                "hist": model_hist,
                "trainy_hat_list": trainy_hat_list,
                }

        return output

    def forecast_model(self,  testX, testy, ind, trainX):
        if self.is_multi_step_prediction:
            return self.forecast_multiple_step( testX, testy)
        else:
            # return self.forecast_single_step( testX, testy)
            return self.forecast_single_step( testX, testy, ind, trainX)

    # def mlp_forecast_multi_step(self, train, testX):
    def forecast_multiple_step(self, testX, testy):
        raise NotImplementedError('this function need to be updated to be similar to mlp_forecast')
    
    def plot_predict_real(self, ind, pred_val):
        # print(ind+self.start_steps+1)
        # wandb.log({'predict vs real':pred_val}, step=ind+self.start_steps+1)
        # wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps+1})
        # print(ind)
        # exit()
        wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps})
        # print(ind+self.start_steps)

    def plot_test_loss_per_run(self, ind, testy, yhat):

        test_loss = mse(testy, yhat)

        # print(testy)
        # print(yhat)
        # exit()

        # print(ind+self.start_steps+1)
        # wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps+1})
        wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps})

    # def mlp_forecast(self, train, testX):
    def forecast_single_step(self, testX, testy, ind, trainX):

        yhat = self.model.predict(asarray([testX]))
        # test_loss = mse(testy, yhat)
        # print(yhat, testX)

        # TODO: align runs to start at step = 0 
        # wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps+1+trainX.shape[0]})
        # wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps+1})

        # VALIDATE: check that predicted value is plotted correctly.
        # wandb.log({'predicted value': yhat}) 

        output = {
                "yhat": yhat.reshape(-1),
                # "hist": model_hist,
                }

        # return yhat.reshape(-1)
        return output

if __name__ == "__main__":

    tf.compat.v1.enable_eager_execution()
    # print(tf.executing_eagerly())
    # exit()
    non_cli_params = {
        'data': df_by_date,
        'model' : (mlp_model, 'mlp'),
        'base_path' : BASEPATH,
        'frame_performance_path' : FRAME_PERFORMANCE_PATH,
        'frame_pred_val_path' : FRAME_PRED_VAL_PATH,
        'plot_path' : PLOT_PATH,
    }
    
    # model_config_params = {
    #         'Dense_1': {
    #             'args': [100], 
    #             'kwargs': {
    #                 'activation':"relu",
    #                 }
    #             },
    #         'Dense_2': {
    #             'args': [],
    #             'kwargs': {}
    #             }
    #         ]
    # }

    # gamma_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    # delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params, 'model_config_params': model_config_params})
