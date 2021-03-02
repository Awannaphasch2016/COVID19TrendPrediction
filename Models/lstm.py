# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from numpy import asarray

# from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *

# # split a univariate sequence into samples
# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the sequence
#         if end_ix > len(sequence) - 1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

def lstm_model(data,state, n_in,n_out):
    print(f'applying lstm to {state}...')

    def lstm_forecast(train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        # trainX, trainy = train[:, :-1], train[:, -1]
        trainX, trainy = train[:, :n_in], train[:, -n_out:]
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
        model = Sequential()
        model.add(LSTM(50, activation="relu", input_shape=(n_in, 1)))
        model.add(Dense(n_out))
        model.compile(optimizer="adam", loss="mse")
        # fit model
        # input dim = [sample,timesteps, features]
        model.fit(trainX, trainy, epochs=200, verbose=0)
        # make a one-step prediction
        testX = asarray(testX).reshape(1, -1, 1)
        yhat = model.predict(testX)
        # return yhat[0]
        return yhat.reshape(-1)

    n_steps_in, n_steps_out = n_in, n_out
    case_by_date_per_states = df_by_date[df_by_date["state"] == state]
    case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
        "float"
    )
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    # data = series_to_supervised(case_by_date_per_states_np, n_in=6)
    data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)
    # mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    #     data, round(case_by_date_florida_np.shape[0] * 0.15), lstm_forecast
    # )
    mse_val, mape_val, rmse_val, r2_val, y, yhat = beta_walk_forward_validation(
        data, round(case_by_date_florida_np.shape[0] * 0.15), lstm_forecast, n_in=n_steps_in, n_out=n_steps_out
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
    #     (lstm_model, 'lstm'),
    #     BASEPATH,
    #     FRAME_PERFORMANCE_PATH,
    #     FRAME_PRED_VAL_PATH,
    #     PLOT_PATH,
    #     test_mode=True,
    # )

    beta_apply_model_to_all_states(
        df_by_date,
        (lstm_model, 'lstm'),
        6,
        7,
        BASEPATH,
        FRAME_PERFORMANCE_PATH,
        FRAME_PRED_VAL_PATH,
        PLOT_PATH,
        test_mode=False,
        # test_mode=True,
    )
