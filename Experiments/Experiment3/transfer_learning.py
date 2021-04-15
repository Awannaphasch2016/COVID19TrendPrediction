from global_params import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *

from numpy import asarray
from numpy import array
from numpy import hstack

from tensorflow import keras
from tensorflow.keras.layers import Dense 
from tensorflow.keras.models import Sequential 
from pandas import DataFrame

from pathlib import Path

base = Path.cwd()
cur_path = base /'Experiments/Experiment3/'


def model_forecast(train, testX):
    base_model = keras.models.load_model('Outputs/Models/Performances/Baselines/OneStep/PredictNext1/WindowLength1/Florida/ModelCheckpoints/us_state_rate_of_change_melted/Florida/mlp_baseline_OneStep_1_1_Florida_epoch=500/')

    ##### add layer to the base model to be trained.
    model = Sequential()
    for layer in base_model.layers[:-1]:
        layer.trainable = False
        model.add(layer)

        # print("weights:", len(layer.weights))
        # print("trainable_weights:", len(layer.trainable_weights))
        # print("non_trainable_weights:", len(layer.non_trainable_weights))
        # print('-----------')

    model.add(Dense(100, activation="relu"))

    # print('xxxxxxxxxx')
    # print("weights:", len(model.layers[-1].weights))
    # print("trainable_weights:", len(model.layers[-1].trainable_weights))
    # print("non_trainable_weights:", len(model.layers[-1].non_trainable_weights))
    # print('xxxxxxxxxx')

    model.add(base_model.layers[-1])

    specified_path = cur_path /'Outputs/ModelCheckpoints'
    checkpoint_filepath = specified_path
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath          = checkpoint_filepath,
        save_weights_only = False,
        save_freq = 'epoch',
        period=10
        )

    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(trainX, trainy, epochs=epoch, verbose=1)

    yhat = model.predict(asarray([testX]))
    output = {
            "yhat": yhat.reshape(-1),
            "hist": hist,
            }
    # return yhat.reshape(-1)
    return output

# Check its architecture
# model.summary()
n_steps_in = 1
n_steps_out = 1 
epoch = 500

tmp = df_by_date
state = 'Louisiana'
case_by_date_per_states = tmp[tmp["state"] == state]
case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)
case_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")
case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

n_test = round(case_by_date_per_states.shape[0] * 0.15)
train, test = train_test_split(data, n_test)

trainX, trainy = train[:, :n_steps_in], train[:, -1].reshape(-1,1)
testX, testy = test[:, :n_steps_in], test[:, -1].reshape(-1,1)

mse_val, mape_val, rmse_val, r2_val, y, yhat = gamma_walk_forward_validation(
   hstack([trainX, trainy]), hstack([testX, testy]), testX, testy, n_test, model_forecast
)


eval_metric_df = DataFrame(
    [[mse_val, mape_val, rmse_val, r2_val]],
    columns=["mape", "mse", "rmse", "r2score"],
)

save_path = cur_path / 'Outputs/eval_metrics.csv'
eval_metric_df.to_csv(save_path)
