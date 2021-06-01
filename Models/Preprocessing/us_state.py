import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator

from global_params import *
import wandb
from Utils.utils import *
from Utils.preprocessing import *

data_path = 'Data/Raw/COVID19Cases/StateLevels/us-states.csv'

# # ==wandb
# # os.environ['WANDB_MODE'] = 'dryrun'
# # Start a new run, tracking hyperparameters in config
# run = wandb.init(project=PROJECT_NAME, 
#     # group="OneStep/PredictNextN/WindowLengthN/state",
#     name='data',
#     save_code=True,
#     job_type='dataset-creation',
#     tags=['Data', 'Raw', 'COVID19Cases', 'StateLevels', 'us-states.csv'],
# )
# artifact = wandb.Artifact('Raw_COVID19Cases_StateLevels_us-states.csv', type='dataset')
# artifact.add_file(data_path)
# run.log_artifact(artifact)
# wandb.finish()

# ==params
n_input = 5
n_features = 1

data = pd.read_csv(
    # str(BASEPATH / pathlib.Path("Data/Raw/COVID19Cases/StateLevels/us-states.csv"))
    str(BASEPATH / pathlib.Path(data_path))
)  # (18824, 5)

df_by_date = pd.DataFrame(
    data.fillna("NA")
    .groupby(["state", "date"])["cases"]
    .sum()
    .sort_values()
    .reset_index()
) 

# import numpy as np
# tmp = df_by_date['cases'].to_numpy()
# tmp = np.diff(tmp,n=1)
# daily_new_case_df_by_date = 1000

# df_by_date.to_csv('Data/Processed/COVID19Cases/StateLevels/us-states_groupby_state_date.csv')

all_states = df_by_date["state"].unique()

# case_by_date_florida = df_by_date[df_by_date["state"] == "Florida"]

# case_by_date_florida_np = case_by_date_florida.to_numpy()[
#     :, 2:].astype("float")
# case_by_date_florida_np = np.reshape(case_by_date_florida_np, (-1, 1))

# case_by_date_florida_train, case_by_date_florida_test = split(
#     case_by_date_florida_np)
# generator_train = TimeseriesGenerator(
#     case_by_date_florida_train, case_by_date_florida_train, length=n_input, batch_size=1
# )

# generator_test = TimeseriesGenerator(
#     case_by_date_florida_test, case_by_date_florida_test, length=n_input, batch_size=1
# )
