"""reference: https://machinelearningmastery.com/how-to-develop-multilayer-perceptron-models-for-time-series-forecasting/"""


# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray

# from sklearn.model_selection import train_test_split
from global_params import *
# from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *


def mlp_model(data, state, n_in,n_out):
    print(f"applying mlp to {state}...")

    # fit an xgboost model and make a one step prediction
    # univariate mlp example
    def mlp_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        # trainX, trainy = train[:, :-1], train[:, -1]
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        model = Sequential()
        # model.add(Dense(100, activation="relu", input_dim=6))
        # model.add(Dense(1))
        model.add(Dense(100, activation="relu", input_dim=n_in))
        model.add(Dense(n_out))
        # model.add(Dense(predict_next_n_days))
        model.compile(optimizer="adam", loss="mse")
        # fit model
        model.fit(trainX, trainy, epochs=10, verbose=0)
        # make a one-step prediction
        yhat = model.predict(asarray([testX]))
        # print(yhat)
        # print('done')
        # exit()
        return yhat.reshape(-1)

    # data = series_to_supervised(case_by_date_florida_np, n_in=6)
    # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    #     data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast
    # )
    # print(n)
    # print('done')
    # exit()
    n_steps_in, n_steps_out = n_in, n_out
    case_by_date_per_states = df_by_date[df_by_date["state"] == state]
    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)
    # data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in)
    # print(data.shape)
    # print(data)
    # convert into input/output
    # X, y = split_sequences(data, n_steps_in, n_steps_out) 
    # print(X.shape)
    # print(y.shape)
    # exit()

    # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    #     data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast
    # )

    mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
        data, round(case_by_date_florida_np.shape[0] * 0.15), mlp_forecast, n_steps_in, n_steps_out
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

    # apply_model_to_all_states(
    #     df_by_date,
    #     (mlp_model, 'mlp'),
    #     BASEPATH,
    #     FRAME_PERFORMANCE_PATH,
    #     FRAME_PRED_VAL_PATH,
    #     PLOT_PATH,
    #     test_mode=False,
    # )


    beta_apply_model_to_all_states(
        df_by_date,
        (mlp_model, 'mlp'),
        6,
        7,
        BASEPATH,
        FRAME_PERFORMANCE_PATH,
        FRAME_PRED_VAL_PATH,
        PLOT_PATH,
        test_mode=False,
        # test_mode=True,
    )
