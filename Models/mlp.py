"""reference: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/"""


# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray
from numpy import array
from numpy import hstack

# from sklearn.model_selection import train_test_split
from global_params import *
# from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *
import click



def mlp_model(data, state, n_in,n_out, is_multi_step_prediction):
    print(f"applying mlp to {state}...")
    
    # fit an xgboost model and make a one step prediction
    # univariate mlp example
    def mlp_forecast_multi_step(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=n_in))
        model.add(Dense(n_out))
        # model.add(Dense(predict_next_n_days))
        model.compile(optimizer="adam", loss="mse")
        # fit model
        model.fit(trainX, trainy, epochs=10, verbose=0)
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
        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=n_in))
        model.add(Dense(1))
        # model.add(Dense(predict_next_n_days))
        model.compile(optimizer="adam", loss="mse")
        # fit model
        model.fit(trainX, trainy, epochs=10, verbose=0)
        # make a one-step prediction
        yhat = model.predict(asarray([testX]))
        return yhat.reshape(-1)

    n_steps_in, n_steps_out = n_in, n_out
    case_by_date_per_states = df_by_date[df_by_date["state"] == state]
    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

    # if is_multi_step_prediction:
    #     mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
    #         data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast_multi_step, n_steps_in, n_steps_out
    #     )
    # else:
    #     mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
    #         data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast, n_steps_in, n_steps_out
    #     )

    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)
    n_test = round(case_by_date_florida_np.shape[0] * 0.15)
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
        n   hstack([trainX, trainy]), hstack([testX, testy]), testX, testy, n_test, mlp_forecast
        )

    eval_metric_df = DataFrame(
        [[mse_val, mape_val, rmse_val, r2_val]],
        columns=["mape", "mse", "rmse", "r2score"],
    )
    return y, yhat, eval_metric_df


if __name__ == "__main__":

    # apply_model_to_all_states(
    #     df_by_date,
    #     (mlp_model, 'mlp'),
    #     BASEPATH,
    #     FRAME_PERFORMANCE_PATH,
    #     FRAME_PRED_VAL_PATH,
    #     PLOT_PATH,
    #     test_mode=False,
    # )


    # beta_apply_model_to_all_states(
    #     df_by_date,
    #     (mlp_model, 'mlp'),
    #     6,
    #     7,
    #     # True,
    #     False,
    #     BASEPATH,
    #     FRAME_PERFORMANCE_PATH,
    #     FRAME_PRED_VAL_PATH,
    #     PLOT_PATH,
    #     test_mode=False,
    #     # test_mode=True,
    # )

    non_cli_params = {
        'data': df_by_date,
        'model' : (mlp_model, 'mlp'),
        'base_path' : BASEPATH,
        'frame_performance_path' : FRAME_PERFORMANCE_PATH,
        'frame_pred_val_path' : FRAME_PRED_VAL_PATH,
        'plot_path' : PLOT_PATH,
    }

    gamma_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
