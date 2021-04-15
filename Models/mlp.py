"""reference: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/"""


# univariate mlp example
from numpy import array
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

from numpy import asarray
from numpy import array
from numpy import hstack

from global_params import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *

import click
from wandb.keras import WandbCallback
import wandb



def mlp_model(data, state, n_in,n_out, is_multi_step_prediction, model_metadata_str, model_params_str, model_params=None):
    assert 'epoch' in list(model_params.keys())

    # epoch = model_params['epoch']
    epoch             = model_params.epoch
    multi_step_folder = model_params.multi_step_folder
    model_name        = model_params.model_name
    dataset_name      = model_params.dataset
    

    print(f"applying mlp to {state}...")
    
    # fit an xgboost model and make a one step prediction
    # univariate mlp example
    def mlp_forecast_multi_step(train, testX):
        raise NotImplementedError('this function need to be updated to be similar to mlp_forecast')
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        model = Sequential(
            [
                Dense(100, activation="relu", input_dim=n_in),
                Dense(n_out),
            ]
        )

        # model = Sequential()
        # model.add(Dense(100, activation="relu", input_dim=n_in))
        # model.add(Dense(n_out))
        model.compile(optimizer="adam", loss="mse")
        # fit model
        model.fit(trainX, trainy, epochs=epoch, verbose=0, callbacks=[WandbCallback()])
        # make a one-step prediction
        yhat = model.predict(asarray([testX]))
        return yhat.reshape(-1)

    # fit an xgboost model and make a one step prediction
    # univariate mlp example
    def mlp_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        # trainX, trainy = train[:, :-1], train[:, -1]
        trainX, trainy = train[:, :n_in], train[:, -1]
        model = Sequential(
            [
                Dense(100, activation="relu", input_dim=n_in),
                Dense(1),
            ]
        )

        specified_path = None if CHECKPOINTS_PATH is None else BASEPATH + \
            CHECKPOINTS_PATH.format(multi_step_folder,n_out, n_in, state , dataset_name, state, model_name)
        specified_path = add_file_suffix(specified_path, model_metadata_str + model_params_str)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        checkpoint_filepath = specified_path
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath          = checkpoint_filepath,
            save_weights_only = False,
            save_freq = 'epoch',
            period=10
            )
        
        model.compile(optimizer="adam", loss="mse")
        # fit model
        hist = model.fit(trainX, trainy, epochs=epoch, verbose=1, callbacks=[WandbCallback(),
            model_checkpoint_callback])

        # make a one-step prediction
        yhat = model.predict(asarray([testX]))

        # TODO: add training/validation loss here too.
        wandb.log({'last_window_step_loss': hist.history['loss'][-1]})
        output = {
                "yhat": yhat.reshape(-1),
                "hist": hist,
                }
        # return yhat.reshape(-1)
        return output

    n_steps_in, n_steps_out = n_in, n_out

    # TMP:
    ###### get input data

    # tmp = df_by_date
    data_path = Path(BASEPATH) / 'Experiments/Experiment2/Data/us_state_rate_of_change_melted.csv'
    tmp = pd.read_csv(data_path)
    tmp = tmp.sort_values(by=['state', 'date'])

    case_by_date_per_states = tmp[tmp["state"] == state]
    # case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
    #     "float"
    # )
    case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)
    case_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")
    # case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
    #     "float"
    # )

    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    ## here> datat is not used. everything started with df_by_date
    ### dataset -> tell which state will be used for prediction
    ###data 
    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

    # if is_multi_step_prediction:
    #     mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
    #         data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast_multi_step, n_steps_in, n_steps_out
    #     )
    # else:
    #     mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
    #         data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast, n_steps_in, n_steps_out
    #     )

    # data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)
    n_test = round(case_by_date_per_states.shape[0] * 0.15)
    train, test = train_test_split(data, n_test)
    if is_multi_step_prediction:
        testX, testy = test[:, :n_steps_in], test[:, -n_steps_out:]
        mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
            train, test, testX, testy, n_test, mlp_forecast_multi_step
        )
    else:
        trainX, trainy = train[:, :n_steps_in], train[:, -1].reshape(-1,1)
        testX, testy = test[:, :n_steps_in], test[:, -1].reshape(-1,1)
        mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
           hstack([trainX, trainy]), hstack([testX, testy]), testX, testy, n_test, mlp_forecast
        )

    eval_metric_df = DataFrame(
        [[mse_val, mape_val, rmse_val, r2_val]],
        columns=["mape", "mse", "rmse", "r2score"],
    )
    return y, yhat, eval_metric_df


if __name__ == "__main__":
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
