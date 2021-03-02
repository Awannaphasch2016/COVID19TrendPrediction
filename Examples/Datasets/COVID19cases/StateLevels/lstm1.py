# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from numpy import asarray

# from sklearn.model_selection import train_test_split
from global_params import *
from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *


# split a univariate sequence into samples
def split_sequences(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# fit an xgboost model and make a one step prediction
def lstm_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(6, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    # fit model
    # input dim = [sample,timesteps, features]
    model.fit(trainX, trainy, epochs=200, verbose=0)
    # make a one-step prediction
    testX = asarray(testX).reshape(1, -1, 1)
    yhat = model.predict(testX)
    return yhat[0]


data = series_to_supervised(case_by_date_florida_np, n_in=6)

mse_val, mape_val, rmse_val, r2_val, y, yhat = walk_forward_validation(
    # data, round(case_by_date_florida_np.shape[0] * 0.15), lstm_forecast
    data,
    10,
    lstm_forecast,
)

frame_pred_val(
    y.reshape(-1),
    array(yhat).reshape(-1),
    save_path=BASEPATH + "/Outputs/Models/Performances/Baselines/lstm_pred_val.csv",
)

plot(y, yhat, save_path=BASEPATH + "/Outputs/Images/LSTM/lstm_forecasting.jpg")

frame_performance(
    mse_val,
    mape_val,
    rmse_val,
    r2_val,
    save_path=BASEPATH + "/Outputs/Models/Performances/Baselines/lstm_performance.csv",
)
