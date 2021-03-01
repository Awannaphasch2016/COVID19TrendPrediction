import numpy as np
from sklearn.metrics import r2_score
from pandas import DataFrame
# from sklearn.model_selection import train_test_split
from numpy import array
from Utils.aws_services import *


def mape(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100


def mse(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean((y1 - y_pred) ** 2)


def rmse(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.sqrt(np.mean((y1 - y_pred) ** 2))


def r2score(y1, y_pred):
    y1, y_pred = np.array(y1), np.array(y_pred)
    return r2_score(y1, y_pred)

def beta_frame_performance(
    metric_df, save_path=None, display=True
):
    if save_path is not None:
        metric_df.to_csv(save_path)
        # x = DataFrame(
        #     [[mse_val, mape_val, rmse_val, r2_val]],
        #     columns=["mape", "mse", "rmse", "r2score"],
        # )
        # x.to_csv(save_path)
        print(f'save performance to {save_path}')
    if display:
        print(metric_df)

    # if save_path is not None:
    #     x = DataFrame(
    #         [metrics_results],
    #         columns=metrics_name,
    #     )
    #     x.to_csv(save_path)
    #     # x = DataFrame(
    #     #     [[mse_val, mape_val, rmse_val, r2_val]],
    #     #     columns=["mape", "mse", "rmse", "r2score"],
    #     # )
    #     # x.to_csv(save_path)
    #     print(f'save performance to {save_path}')
    # else:
    #     x = DataFrame(
    #         [metrics_result],
    #         columns=metrics_name,
    #     )
    #     # x = DataFrame(
    #     #     [[mse_val, mape_val, rmse_val, r2_val]],
    #     #     columns=["mape", "mse", "rmse", "r2score"],
    #     # )
    # if display:
    #     print(x)

def frame_performance(
    mse_val, mape_val, rmse_val, r2_val, save_path=None, display=True
):
    if save_path is not None:
        x = DataFrame(
            [[mse_val, mape_val, rmse_val, r2_val]],
            columns=["mape", "mse", "rmse", "r2score"],
        )
        x.to_csv(save_path)
        print(f'save performance to {save_path}')
    else:
        x = DataFrame(
            [[mse_val, mape_val, rmse_val, r2_val]],
            columns=["mape", "mse", "rmse", "r2score"],
        )
    if display:
        print(x)


def frame_pred_val(y_test, y_pred, save_path=None):
    if save_path is not None:
        x = DataFrame(
            np.array([y_test, y_pred]).T,
            columns=["y_test", "y_pred"],
        )
        # print(x)
        x.to_csv(save_path)
        print(f'save pred_val to {save_path}')
    else:
        x = DataFrame(
            np.array([y_test, y_pred]).T,
            columns=["y_test", "y_pred"],
        )
        # print(x)

def split_by_predict_next_n_day(data, n):
    reshape_data = []
    i = 0 
    while i + n <= data.shape[0]:
        x = data[i:i+n].reshape(-1)
        reshape_data.append(x)
        # print(case_by_date_florida_test[i:i+7])
        i += 1
    return array(reshape_data)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# # fit an xgboost model and make a one step prediction
# def xgboost_forecast(train, testX, model):
#     # transform list into array
#     train = asarray(train)
#     # split into input and output columns
#     trainX, trainy = train[:, :-1], train[:, -1]
#     # fit model
#     model = XGBRegressor(objective="reg:squarederror", n_estimators=1000)
#     model.fit(trainX, trainy)
#     # make a one-step prediction
#     yhat = model.predict(asarray([testX]))
#     return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, model_forecast):
    """
    model_forcast = see xgboost_forecast() as en example.
    """
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        # testX, testy = test[i, :-1], test[i, -1]
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        # yhat = xgboost_forecast(history, testX)
        yhat = model_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print(">expected=%.1f, predicted=%.1f" % (testy, yhat))
    # estimate prediction error
    # error = mean_absolute_error(test[:, -1], predictions)
    mse_val = mse(test[:, -1], predictions)
    mape_val = mape(test[:, -1], predictions)
    rmse_val = rmse(test[:, -1], predictions)
    r2_val = r2score(test[:, -1], predictions)
    return mse_val, mape_val, rmse_val, r2_val, test[:, -1], array(predictions)
