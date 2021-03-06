import numpy as np
from sklearn.metrics import r2_score
from pandas import DataFrame
# from sklearn.model_selection import train_test_split
from numpy import array
from Utils.aws_services import *
from pathlib import Path
from numpy import hstack

def loss_per_epoch(n_batches, loss_val):
    """for keras, calculate loss for epoch instead of loss per batch."""
    return  loss_val/n_batch


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def mape(y1, y_pred):
    y1, y_pred = np.array(y1).reshape(-1), np.array(y_pred).reshape(-1)
    tmp = y1 - y_pred
    assert tmp.shape[0] == y1.shape[0]
    return np.mean(np.abs(tmp / y1)) * 100


def mse(y1, y_pred):
    y1, y_pred = np.array(y1).reshape(-1), np.array(y_pred).reshape(-1)
    tmp = y1 - y_pred
    output = np.mean(tmp ** 2)

    # print(y1.shape)
    # print(y_pred.shape)
    # print(y1.reshape(-1)-y_pred)
    # print(output)
    # print(y1)
    # print('sssssss')
    # # exit()

    assert tmp.shape[0] == y1.shape[0]
    return output


def rmse(y1, y_pred):
    # y1, y_pred = np.array(y1), np.array(y_pred)
    y1, y_pred = np.array(y1).reshape(-1), np.array(y_pred).reshape(-1)
    tmp = y1 - y_pred
    assert tmp.shape[0] == y1.shape[0]
    return np.sqrt(np.mean(tmp ** 2))


def r2score(y1, y_pred):
    # y1, y_pred = np.array(y1), np.array(y_pred)
    y1, y_pred = np.array(y1).reshape(-1), np.array(y_pred).reshape(-1)
    # tmp = y1 - y_pred
    # assert tmp.shape[0] == y1.shape[0]
    return r2_score(y1, y_pred)

def beta_frame_performance(
    metric_df, save_path=None, display=True, is_save=False
):

    if is_save:
        assert save_path is not None
        metric_df.to_csv(save_path)
        print(f'save frame_performance to local at {save_path}')
        save_to_s3(save_path)
        # x = DataFrame(
        #     [[mse_val, mape_val, rmse_val, r2_val]],
        #     columns=["mape", "mse", "rmse", "r2score"],
        # )
        # x.to_csv(save_path)
        print(f'save frame_performance to S3 at {save_path}')
    else:
        print('frame_performance is not saved')
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

def beta_frame_pred_val(y_test, y_pred, save_path=None, is_save=False):
    if is_save:
        assert save_path is not None
        x = DataFrame(
            np.array([y_test, y_pred]).T,
            columns=["y_test", "y_pred"],
        )
        # print(x)
        x.to_csv(save_path)
        print(f'save pred_val to local at {save_path}')
        save_to_s3(save_path)
        print(f'save pred_val to S3 at {save_path}')
    else:
        x = DataFrame(
            np.array([y_test, y_pred]).T,
            columns=["y_test", "y_pred"],
        )
        # print(x)
        print('pred_val is not saved')


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

def delta_walk_forward_validation(train,test, trainX, trainy, testX, testy, n_test, model,
        train_model_with_1_run, dont_create_new_model_on_each_run, evaluate_on_many_test_data_per_run):
    predictions = list()
    # split dataset
    # seed history with training dataset
    # NOTE: I use train variable and test variable instead of data variable because it limit the possible
    #   logic branches as a result, reduce complexity of code with trade off of more text to write
    n_all_data = train.shape[0] + testy.shape[0]
    history = [x for x in train]

    # print(len(history), train.shape[0] , testy.shape[0])
    # exit()

    test_data_per_run = testy
    if dont_create_new_model_on_each_run:
        n_model_run = 1 
    elif dont_create_new_model_on_each_run is None:
        n_model_run = None 
    else:
        n_model_run = float('inf')

    model_hist = None

    if train_model_with_1_run is None:
        assert evaluate_on_many_test_data_per_run is None 
        assert n_model_run is None 
        model.init_model()
        fit_output = model.fit_model(history, test, trainX, trainy, testX, testy)

        # try:
        #     model.forecast_train_data(history, test, trainX, trainy, testX, testy)
        # except:
        #     pass

        for i in range(len(test)):
            ind = i + trainX.shape[0]
            model_output = model.forecast_model(testX[i], testy[i])

            output_yhat = model_output['yhat']
            model.plot_predict_real( ind, output_yhat)

            # print(testX[i])
            # print(testy[i])
            # print(output_yhat)
            # print('------------------')

            # store forecast in list of predictions
            predictions.append(output_yhat[0])

    # step over each time-step in the test set
    elif train_model_with_1_run:
        assert n_model_run == 1 
        assert evaluate_on_many_test_data_per_run, 'train_model_with_1_run will test for all test data.'

        # fit model on history and make a prediction
        # model.fit_model(history, trainX, trainy)
        model.init_model()

        fit_output = model.fit_model(history, test, trainX, trainy, testX, testy)

        for i in range(len(test)):
            # print(i)
            # print('ssssssssssssssssssss')
            assert testX[i].shape[0] + testy[i].shape[0] == test.shape[1], 'multi-step splitting is not correct'
            ind = i + trainX.shape[0]

            model_output = model.forecast_model( testX[i], testy[i], i, trainX)

            output_yhat = model_output['yhat']
            # print(output_yhat)
            print('---------')
            print(testy[i], output_yhat)
            model.plot_predict_real( ind, output_yhat)
            model.plot_test_loss_per_run(ind, testy[i],output_yhat)
            # store forecast in list of predictions
            predictions.append(output_yhat)

        # summarize progress
        print(f">\n\texpected={testy[i]}\n\tpredicted={output_yhat}")
    else:
        raise NotImplementedError("I haven't throughly tested the case under this condition after adding" \
                "'predict vs real' plot.")
        for i in range(len(test)):
            assert testX[i].shape[0] + testy[i].shape[0] == test.shape[1], 'multi-step splitting is not correct'

            # fit model on history and make a prediction
            if i <= n_model_run:
                model.init_model()
            model.fit_model(history, test_data_per_run, trainX, trainy, testX, testy)

            test_data_per_run = testy[i+1:]
            # evaluate after training is finished.
            if not evaluate_on_many_test_data_per_run:
                model_output = model.forecast_model( testX[i], testy[i])
                output_yhat = model_output['yhat']
                print(f">\n\texpected={testy[i]}\n\tpredicted={output_yhat}")
                predictions.append(output_yhat)
            else:
                raise NotImplementedError('fuck this condition. wack.')
                # NOTE: I implemented this, but, why would anyone what to evaluate their modle with this?
                #     this evaluation process just doesn't seem to be fair (favoring model with higher runs.)
                # test_data_per_run = testy[i+1:]
                model_output = model.forecast_model( testX[i:], testy[i:])
                output_yhat = model_output['yhat']
                print(f">\n\texpected={test_data_per_run}\n\tpredicted={output_yhat}")

                # output_yhat = model_output['yhat']
                # store forecast in list of predictions
                predictions.append(output_yhat[0])
                # predictions = output_yhat

            # add actual observation to history for the next loop
            history.append(test[i])
            assert len(history) + test_data_per_run.shape[0] ==  n_all_data, f"{len(history)} + " \
                f"{test_data_per_run.shape[0]} !=  {n_all_data}"

            # summarize progress
            # print(f">\n\texpected={testy[i]}\n\tpredicted={output_yhat}")
            # print(f">\n\texpected={test_data_per_run.reshape(-1)}\n\tpredicted={output_yhat}")

    if 'hist' in fit_output.keys():
        model_hist = fit_output['hist'] 

    if 'trainy_hat_list' in fit_output.keys():
        trainy_hat_list = fit_output['trainy_hat_list'] 


    # evaluate only on the first test prediction.
    mse_val = mse(testy, predictions)
    mape_val = mape(testy, predictions)
    rmse_val = rmse(testy, predictions)
    r2_val = r2score(testy, predictions)

    # output = {
    #         "mse_val":     mse_val,
    #         "mape_val":    mape_val,
    #         "rmse_val":    rmse_val,
    #         "r2_val":      r2_val,
    #         "testy":       testy,
    #         "predictions": predictions,
    #         "model_hist":  model_hist
    #         }

    return mse_val, mape_val, rmse_val, r2_val, testy, predictions, model_hist, trainy_hat_list

def gamma_walk_forward_validation(train,test, testX, testy, n_test, model_forecast):
    """
    model_forcast = see xgboost_forecast() as en example.
    """
    predictions = list()
    # split dataset
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        assert testX[i].shape[0] + testy[i].shape[0] == test.shape[1], 'multi-step splitting is not correct'
        # fit model on history and make a prediction
        # yhat = model_forecast(history, testX[i])
        model_output = model_forecast(history, testX[i])
        output_yhat = model_output['yhat']
        # store forecast in list of predictions
        predictions.append(output_yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print(f">\n\texpected={testy[i]}\n\tpredicted={output_yhat}")
    mse_val = mse(testy, predictions)
    mape_val = mape(testy, predictions)
    rmse_val = rmse(testy, predictions)
    r2_val = r2score(testy, predictions)
    return mse_val, mape_val, rmse_val, r2_val, testy, predictions


def beta_walk_forward_validation(data, n_test, model_forecast, n_in, n_out):
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
        # testX, testy = test[i, :-1], test[i, -1]
        testX, testy = test[i, :n_in], test[i, -n_out:]
        # print(test.shape)
        # print(testX.shape[0])
        # print(testy.shape[0])
        assert testX.shape[0] + testy.shape[0] == test.shape[1], 'multi-step splitting is not correct'
        # exit()
        # fit model on history and make a prediction
        # yhat = xgboost_forecast(history, testX)
        # yhat = model_forecast(history, testX)
        yhat = model_forecast(history, testX)
        # print(train.shape)
        # print(test.shape)
        # print(testX.shape)
        # print(testy.shape)
        # print(yhat)
        # exit()
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        # print(">expected=%.1f, predicted=%.1f" % (testy, yhat))
        print(f">\n\texpected={testy}\n\tpredicted={yhat}")
    # estimate prediction error
    # error = mean_absolute_error(test[:, -1], predictions)
    mse_val = mse(test[:, -n_out:], predictions)
    mape_val = mape(test[:, -n_out:], predictions)
    rmse_val = rmse(test[:, -n_out:], predictions)
    r2_val = r2score(test[:, -n_out:], predictions)
    return mse_val, mape_val, rmse_val, r2_val, test[:, -n_out:], predictions

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, model_forecast):
    """
    model_forcast = see xgboost_forecast() as en example.
    """
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # print(train.shape)
    # print(test.shape)
    # exit()

    # n_in, n_out = 6, 7 
    # X, y = split_sequences(train, n_in, n_out) 
    # print(X.shape)
    # print(y.shape)
    # exit()

    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        # testX, testy = test[i, :-1], test[i, -1]
        testX, testy = test[i, :-1], test[i, -1]
        # print(test)
        # print(testX)
        # print(testy)
        # exit()
        # fit model on history and make a prediction
        # yhat = xgboost_forecast(history, testX)
        yhat = model_forecast(history, testX)
        # print(train.shape)
        # print(test.shape)
        # print(testX.shape)
        # print(testy.shape)
        # print(yhat)
        # exit()
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
