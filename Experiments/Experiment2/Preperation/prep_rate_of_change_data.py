'''Prepare dataset by converting us-state to rate of change.'''

import pandas as pd 
import numpy as np
import pathlib
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from pathlib import Path

path = 'Experiments/Experiment2/'
base = Path.cwd()
cur_path = base /'Experiments/Experiment2/'

df_by_date = pd.read_csv(str(cur_path / 'Data/us_state_clean.csv'), index_col=False)
case_by_date_florida = pd.read_csv(str(cur_path /"Data/fl_case.csv"), index_col=False)

groupby_states = df_by_date.groupby('state')["cases"].apply(np.array)
# compute rate of change from raw data
groupby_states = pd.Series([ pd.Series(i).pct_change() for i in groupby_states],index=groupby_states.index)
# groupby_states.to_csv(str(cur_path /'Data/us_state_rate_of_change.csv'))

all_states = groupby_states.index.to_numpy()
groupby_states_np =  groupby_states.to_numpy()
states_ts_len = np.array([groupby_states_np[i].shape[0] for i in range(groupby_states_np.shape[0])])
max_states_ts_len = states_ts_len.max()

states_pad_len = max_states_ts_len - states_ts_len
left_padded_states_ts = np.array([np.hstack((np.zeros(p), groupby_states_np[i])) for i, p in enumerate(states_pad_len)])
# groupby_states_np_ts  = to_time_series_dataset(groupby_states_np) # each instance needs to be aligned by date
tmp = left_padded_states_ts[:, states_pad_len.max():]
groupby_states_rate_of_change = pd.DataFrame(tmp,
        columns=df_by_date.date.unique()[states_pad_len.max():],
        index=df_by_date.state.unique())
groupby_states_rate_of_change.T.to_csv(str(cur_path/ 'Data/us_state_rate_of_change.csv'))
groupby_states_np_ts = to_time_series_dataset(tmp) # each instance needs to be aligned by date

# align time series
groupby_states_np_ts[:, states_pad_len.max():]

