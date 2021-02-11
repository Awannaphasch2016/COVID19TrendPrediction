import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator

from global_params import *
from Utils.preprocessing import *
from Utils.utils import *

# ==params
n_input = 5
n_features = 1

data = pd.read_csv(
    str(BASEPATH / pathlib.Path("Data/Raw/COVID19Cases/StateLevels/us-states.csv"))
)  # (18824, 5)

df_by_date = pd.DataFrame(
    data.fillna("NA")
    .groupby(["state", "date"])["cases"]
    .sum()
    .sort_values()
    .reset_index()
)
case_by_date_florida = df_by_date[df_by_date["state"] == "Florida"]
# print(case_by_date_florida)
# print(case_by_date_florida.shape)
# exit()

case_by_date_florida_np = case_by_date_florida.to_numpy()[:, 2:].astype("float")
# print(case_by_date_florida_np)
# print(case_by_date_florida_np.shape)
# exit()

case_by_date_florida_np = np.reshape(case_by_date_florida_np, (-1, 1))
# print(NtsC.shape
# exit()

case_by_date_florida_train, case_by_date_florida_test = split(case_by_date_florida_np)
# print(case_by_date_florida_train.shape)
# print(case_by_date_florida_test.shape)
# exit()


# test2 = np.reshape(test, (-1, 1))
generator_train = TimeseriesGenerator(
    case_by_date_florida_train, case_by_date_florida_train, length=n_input, batch_size=1
)
# test2 = np.reshape(test, (-1, 1))
generator_test = TimeseriesGenerator(
    case_by_date_florida_test, case_by_date_florida_test, length=n_input, batch_size=1
)
# print(list(generator))
