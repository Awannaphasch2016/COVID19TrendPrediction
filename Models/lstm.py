
from numpy import array

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers import Dense, LSTM

from numpy import asarray
from numpy import array
from numpy import hstack

from global_params import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *
from wandb.keras import WandbCallback
import wandb

class lstm_model:
    def __init__(self, data, state, n_in,n_out, max_steps, min_steps, is_multi_step_prediction, model_metadata_str,
            model_params_str, model_params=None):
        assert 'epoch' in list(model_params.keys())

        self.is_multi_step_prediction = is_multi_step_prediction
        self.model_metadata_str       = model_metadata_str
        self.model_params_str         = model_params_str
        self.state                    = state
        self.max_steps                = max_steps
        self.min_steps                = min_steps

        # epoch = model_params['epoch']
        self.epoch                             = model_params.epoch
        self.multi_step_folder                 = model_params.multi_step_folder
        self.model_name                        = model_params.model_name
        self.dataset_name                      = model_params.dataset
        self.train_model_with_1_run            = model_params.train_model_with_1_run
        self.dont_create_new_model_on_each_run = model_params.dont_create_new_model_on_each_run
        self.evaluate_on_many_test_data_per_run = model_params.evaluate_on_many_test_data_per_run
        self.experiment_id                      = int(model_params.experiment_id)

        self.n_steps_in, self.n_steps_out = n_in, n_out
        self.prep_data()
        # self.fit_model()

    def prep_data(self):

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

        # self.start_steps = self.max_steps - case_by_date_per_states_np.shape[0] + self.n_steps_in
        self.start_steps = self.max_steps - case_by_date_per_states_np.shape[0] + self.n_steps_in + \
         self.n_steps_out - 1

        ### dataset -> tell which state will be used for prediction
        ###data 
        data = series_to_supervised(case_by_date_per_states_np, n_in=self.n_steps_in, n_out=self.n_steps_out)
        self.split = 0.15
        self.n_test = round(case_by_date_per_states.shape[0] * 0.15)
        self.train, self.test = train_test_split(data, self.n_test)

    def forecast(self):

        if self.is_multi_step_prediction:
            raise NotImplementedError("update to have the same structure as else condition.")
            testX, testy = self.test[:, :self.n_steps_in], self.test[:, -self.n_steps_out:]
            mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
                self.train, self.test, testX, testy, self.n_test, self.forecast_multiple_step
            )
        else:
            trainX, trainy = self.train[:, :self.n_steps_in], self.train[:, -1].reshape(-1,1)
            testX, testy = self.test[:, :self.n_steps_in], self.test[:, -1].reshape(-1,1)

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

    def init_model(self):

        self.model = Sequential()
        self.model.add(LSTM(50, activation="relu", input_shape=(self.n_steps_in, 1)))
        self.model.add(Dense(1))

        specified_path = None if CHECKPOINTS_PATH is None else BASEPATH + \
            CHECKPOINTS_PATH.format(self.multi_step_folder,self.n_steps_out, self.n_steps_in, self.state ,
                    self. dataset_name, self.state, self.model_name)
        specified_path = add_file_suffix(specified_path, self.model_metadata_str + self.model_params_str)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        checkpoint_filepath = specified_path
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath          = checkpoint_filepath,
            save_weights_only = False,
            save_freq = 'epoch',
            period=10
            )

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

        self.model.compile(optimizer="adam", loss="mse")
        self.specified_path = specified_path
        self.es = es
        self.model_checkpoint_callback = model_checkpoint_callback

    def fit_model(self, train,test, trainX, trainy, testX, testy):

        specified_path            = self.specified_path 
        es                        = self.es 
        model_checkpoint_callback = self.model_checkpoint_callback 

        # transform list into array
        train = asarray(train)

        # split into input and output columns
        trainX, trainy = train[:, :self.n_steps_in], train[:, -1].reshape(-1,1)
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
        testX = testX.reshape((testX.shape[0], testX.shape[1], 1))
    
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
        model_hist = self.model.fit(np.vstack([trainX, testX]), np.vstack([trainy,testy]), epochs=self.epoch,
                verbose=1,
                validation_split=self.split,
                callbacks=[WandbCallback(), model_checkpoint_callback])

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



    # def fit_model(self, train, trainX, trainy):

    #     train = asarray(train)
    #     # split into input and output columns
    #     # trainX, trainy = train[:, :-1], train[:, -1]
    #     trainX, trainy = train[:, :self.n_steps_in], train[:, -self.n_steps_out:]
    #     trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

    #     # model = Sequential(
    #     #     [
    #     #         LSTM(50, activation="relu", input_shape=(n_in, 1)),
    #     #         Dense(1),
    #     #     ]
    #     # )

    #     self.model = Sequential()
    #     self.model.add(LSTM(50, activation="relu", input_shape=(self.n_steps_in, 1)))
    #     self.model.add(Dense(1))

    #     specified_path = None if CHECKPOINTS_PATH is None else BASEPATH + \
    #         CHECKPOINTS_PATH.format(self.multi_step_folder,self.n_steps_out, self.n_steps_in, self.state ,
    #                 self.dataset_name, self.state, self.model_name)
    #     specified_path = add_file_suffix(specified_path, self.model_metadata_str + self.model_params_str)
    #     parent_dir = '/'.join(specified_path.split('/')[:-1])
    #     Path(parent_dir).mkdir(parents=True,exist_ok=True)

    #     checkpoint_filepath = specified_path
    #     model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #         filepath          = checkpoint_filepath,
    #         save_weights_only = False,
    #         save_freq = 'epoch',
    #         period=10
    #         )

    #     es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

    #     self.model.compile(optimizer="adam", loss="mse")

    #     # fit model
    #     model_hist = self.model.fit(trainX, trainy, epochs=self.epoch, verbose=1, callbacks=[WandbCallback(),
    #         model_checkpoint_callback, es])

    #     # last_loss_val = model_hist.history['loss'][-1]
    #     loss_vals = model_hist.history['loss']
    #     n_loss = len(model_hist.history['loss'])

    #     # wandb.log({'last_window_step_loss': last_loss_val})
    #     if not self.train_model_with_1_run:
    #         wandb.log({'loss_per_run (avg over number of loss)': sum(loss_vals)/n_loss})

    #     # return model, hist
    #     # return hist

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
        # wandb.log({'predict vs real':pred_val}, step=ind+self.start_steps)
        # wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps+1})
        wandb.log({'predict vs real':pred_val, 'custom step':ind+self.start_steps})


    def plot_test_loss_per_run(self, ind, testy, yhat):
        test_loss = mse(testy, yhat)
        # print(ind+self.start_steps+1)
        # wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps+1})
        wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps})

    # def mlp_forecast(self, train, testX):
    def forecast_single_step(self, testX, testy, ind,trainX):

        testX = asarray(testX).reshape(1, -1, 1)
        # yhat = self.model.predict(asarray([testX]))
        yhat = self.model.predict(testX)
        # test_loss = mse(testy, yhat)
        # print(yhat, testX)

        # wandb.log({'test_loss_per_run': test_loss})
        # wandb.log({'test_loss_per_run': test_loss, 'custom step': ind+self.start_steps+1+trainX.shape[0]})

        output = {
                "yhat": yhat.reshape(-1),
                # "hist": model_hist,
                }

        # return yhat.reshape(-1)
        return output

# # def lstm_model(data,state, n_in,n_out, is_multi_step_prediction, model_params=None):
# def lstm_model(data, state, n_in,n_out, is_multi_step_prediction, model_metadata_str, model_params_str,
#         model_params=None):
#     assert 'epoch' in list(model_params.keys())
    
#     epoch             = model_params.epoch
#     multi_step_folder = model_params.multi_step_folder
#     model_name        = model_params.model_name
#     dataset_name      = model_params.dataset

#     print(f'applying lstm to {state}...')

#     def lstm_forecast_multi_step(train, testX):
#         raise NotImplementedError('this function need to be updated to be similar to lstm_forecast')
#         # transform list into array
#         train = asarray(train)
#         # split into input and output columns
#         # trainX, trainy = train[:, :-1], train[:, -1]
#         trainX, trainy = train[:, :n_in], train[:, -n_out:]
#         trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

#         model = Sequential(
#             [
#                 LSTM(50, activation="relu", input_shape=(n_in, 1)),
#                 Dense(n_out),
#             ]
#         )

#         # model = Sequential()
#         # model.add(LSTM(50, activation="relu", input_shape=(n_in, 1)))
#         # model.add(Dense(n_out))
#         model.compile(optimizer="adam", loss="mse")
#         # fit model
#         # input dim = [sample,timesteps, features]
#         model.fit(trainX, trainy, epochs=epoch, verbose=0)
#         # make a one-step prediction
#         testX = asarray(testX).reshape(1, -1, 1)
#         yhat = model.predict(testX)
#         # return yhat[0]
#         return yhat.reshape(-1)

#     def lstm_forecast(train, testX):
#         # transform list into array
#         train = asarray(train)
#         # split into input and output columns
#         # trainX, trainy = train[:, :-1], train[:, -1]
#         trainX, trainy = train[:, :n_in], train[:, -n_out:]
#         trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

#         # model = Sequential(
#         #     [
#         #         LSTM(50, activation="relu", input_shape=(n_in, 1)),
#         #         Dense(1),
#         #     ]
#         # )

#         model = Sequential()
#         model.add(LSTM(50, activation="relu", input_shape=(n_in, 1)))
#         model.add(Dense(1))

#         specified_path = None if CHECKPOINTS_PATH is None else BASEPATH + \
#             CHECKPOINTS_PATH.format(multi_step_folder,n_out, n_in, state , dataset_name, state, model_name)
#         specified_path = add_file_suffix(specified_path, model_metadata_str + model_params_str)
#         parent_dir = '/'.join(specified_path.split('/')[:-1])
#         Path(parent_dir).mkdir(parents=True,exist_ok=True)

#         checkpoint_filepath = specified_path
#         model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#             filepath          = checkpoint_filepath,
#             save_weights_only = False,
#             save_freq = 'epoch',
#             period=10
#             )

#         es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

#         model.compile(optimizer="adam", loss="mse")
#         # fit model
#         # input dim = [sample,timesteps, features]
#         hist = model.fit(trainX, trainy, epochs=epoch, verbose=1, callbacks=[WandbCallback(),
#             model_checkpoint_callback, es] )
#         # make a one-step prediction
#         testX = asarray(testX).reshape(1, -1, 1)
#         yhat = model.predict(testX)
#         test_loss = mse(testX, yhat)

#         last_loss_val = hist.history['loss'][-1]
#         loss_vals = hist.history['loss']
#         n_loss = len(hist.history['loss'])

#         # wandb.log({'last_window_step_loss': hist.history['loss'][-1]})

#         wandb.log({'last_window_step_loss': last_loss_val})
#         wandb.log({'loss_per_run (avg over number of loss)': sum(loss_vals)/n_loss})
#         wandb.log({'test loss': test_loss})

#         output = {
#                 "yhat": yhat.reshape(-1),
#                 "hist": hist,
#                 }

#         # return yhat.reshape(-1)
#         return output

#     n_steps_in, n_steps_out = n_in, n_out
#     case_by_date_per_states = df_by_date[df_by_date["state"] == state]
#     case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
#         "float"
#     )
#     case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
#     data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

#     n_test = round(case_by_date_florida_np.shape[0] * 0.15)
#     train, test = train_test_split(data, n_test)
#     if is_multi_step_prediction:
#         # mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
#         #     data, round(case_by_date_florida_np.shape[0] * 0.15), lstm_forecast, n_in=n_steps_in, n_out=n_steps_out
#         # )
#         testX, testy = test[:, :n_steps_in], test[:, -n_steps_out:]
#         mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
#             train, test, testX, testy, n_test, lstm_forecast_multi_step
#         )
#     else:
#         trainX, trainy = train[:, :n_steps_in], train[:, -1].reshape(-1,1)
#         testX, testy = test[:, :n_steps_in], test[:, -1].reshape(-1,1)
#         mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
#            hstack([trainX, trainy]), hstack([testX, testy]), testX, testy, n_test, lstm_forecast
#         )

#     eval_metric_df = DataFrame(
#         [[mse_val, mape_val, rmse_val, r2_val]],
#         columns=["mape", "mse", "rmse", "r2score"],
#     )
#     # print(eval_metric_df)
#     # exit()

#     # return y, yhat, mse_val, mape_val, rmse_val, r2_val
#     return y, yhat, eval_metric_df

if __name__ == "__main__":
    non_cli_params = {
        'data': df_by_date,
        'model' : (lstm_model, 'lstm'),
        'base_path' : BASEPATH,
        'frame_performance_path' : FRAME_PERFORMANCE_PATH,
        'frame_pred_val_path' : FRAME_PRED_VAL_PATH,
        'plot_path' : PLOT_PATH,
    }

    # gamma_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
