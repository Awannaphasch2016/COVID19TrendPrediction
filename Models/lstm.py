
from numpy import array

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM

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


def lstm_model(data,state, n_in,n_out, is_multi_step_prediction, model_params=None):
    assert 'epoch' in list(model_params.keys())
    
    epoch = model_params.epoch
    print(f'applying lstm to {state}...')

    def lstm_forecast_multi_step(train, testX):
        raise NotImplementedError('this function need to be updated to be similar to lstm_forecast')
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        # trainX, trainy = train[:, :-1], train[:, -1]
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

        model = Sequential(
            [
                LSTM(50, activation="relu", input_shape=(n_in, 1)),
                Dense(n_out),
            ]
        )

        # model = Sequential()
        # model.add(LSTM(50, activation="relu", input_shape=(n_in, 1)))
        # model.add(Dense(n_out))
        model.compile(optimizer="adam", loss="mse")
        # fit model
        # input dim = [sample,timesteps, features]
        model.fit(trainX, trainy, epochs=epoch, verbose=0)
        # make a one-step prediction
        testX = asarray(testX).reshape(1, -1, 1)
        yhat = model.predict(testX)
        # return yhat[0]
        return yhat.reshape(-1)

    def lstm_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        # trainX, trainy = train[:, :-1], train[:, -1]
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))

        # model = Sequential(
        #     [
        #         LSTM(50, activation="relu", input_shape=(n_in, 1)),
        #         Dense(1),
        #     ]
        # )

        model = Sequential()
        model.add(LSTM(50, activation="relu", input_shape=(n_in, 1)))
        model.add(Dense(1))

        # model = Sequential(
        #     [
        #         Dense(100, activation="relu", input_dim=n_in),
        #         Dense(1),
        #     ]
        # )

        model.compile(optimizer="adam", loss="mse")
        # fit model
        # input dim = [sample,timesteps, features]
        hist = model.fit(trainX, trainy, epochs=epoch, verbose=1, callbacks=[WandbCallback()] )
        # make a one-step prediction
        testX = asarray(testX).reshape(1, -1, 1)
        yhat = model.predict(testX)

        wandb.log({'last_window_step_loss': hist.history['loss'][-1]})
        output = {
                "yhat": yhat.reshape(-1),
                "hist": hist,
                }

        # return yhat.reshape(-1)
        return output

    n_steps_in, n_steps_out = n_in, n_out
    case_by_date_per_states = df_by_date[df_by_date["state"] == state]
    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

    n_test = round(case_by_date_florida_np.shape[0] * 0.15)
    train, test = train_test_split(data, n_test)
    if is_multi_step_prediction:
        # mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
        #     data, round(case_by_date_florida_np.shape[0] * 0.15), lstm_forecast, n_in=n_steps_in, n_out=n_steps_out
        # )
        testX, testy = test[:, :n_steps_in], test[:, -n_steps_out:]
        mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
            train, test, testX, testy, n_test, lstm_forecast_multi_step
        )
    else:
        trainX, trainy = train[:, :n_steps_in], train[:, -1].reshape(-1,1)
        testX, testy = test[:, :n_steps_in], test[:, -1].reshape(-1,1)
        mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
           hstack([trainX, trainy]), hstack([testX, testy]), testX, testy, n_test, lstm_forecast
        )

    eval_metric_df = DataFrame(
        [[mse_val, mape_val, rmse_val, r2_val]],
        columns=["mape", "mse", "rmse", "r2score"],
    )
    # print(eval_metric_df)
    # exit()

    # return y, yhat, mse_val, mape_val, rmse_val, r2_val
    return y, yhat, eval_metric_df

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
