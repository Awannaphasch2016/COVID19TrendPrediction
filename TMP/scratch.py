from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *
import click

from wandb.keras import WandbCallback
import wandb
## raw data 
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator

from global_params import *
from Utils.preprocessing import *
from Utils.utils import *
import wandb


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
)ase_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")

# VALIDATE: I have yet tested it throguhtly, if sorted will have any side effect, but it seems very safe.
df_by_date = df_by_date.sort_values(by=['state', 'date'])
case_by_date_florida = df_by_date[df_by_date["state"] == "Florida"]

case_by_date_florida_np = case_by_date_florida.to_numpy()[:, 2:].astype("float")


###### get rate_of_change data
# tmp = df_by_date
data_path = Path(BASEPATH) / 'Experiments/Experiment2/Data/us_state_rate_of_change_melted.csv'
tmp = pd.read_csv(data_path)
tmp = tmp.sort_values(by=['state', 'date'])
# case_by_date_per_states = df_by_date[df_by_date["state"] == state]
case_by_date_per_states = tmp[tmp["state"] == state]
# case_by_date_per_states_np = case_by_date_per_states.to_numpy()[:, 2:].astype(
#     "float"
# )

case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)

case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
## here> datat is not used. everything started with df_by_date
### dataset -> tell which state will be used for prediction
###data 
n_in = 3
n_out = 3 
n_steps_in, n_steps_out = n_in, n_out

data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

n_test = round(case_by_date_per_states.shape[0] * 0.15)
train, test = train_test_split(data, n_test)


