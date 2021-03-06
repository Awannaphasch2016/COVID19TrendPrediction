"""default for centroid = 3, rolling_mean = 2"""
# from Momels.Preprocessing.us_state import *

import pandas as pd 
import numpy as np
from pathlib import Path
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

path = 'Experiments/Experiment2/'
base = Path.cwd()
cur_path = base /'Experiments/Experiment2/'

df_by_date = pd.read_csv(str(cur_path / 'Data/us_state_clean.csv'), index_col=False)
case_by_date_florida = pd.read_csv(str(cur_path /"Data/fl_case.csv"), index_col=False)

groupby_states_rate_of_change = pd.read_csv(str(cur_path /'Data/us_state_rate_of_change.csv'))
groupby_states_rate_of_change_melted = pd.read_csv(str(cur_path /'Data/us_state_rate_of_change_melted.csv'))
groupby_states_rate_of_change.set_index(groupby_states_rate_of_change.columns[0], inplace=True)
groupby_states_rate_of_change.replace(np.nan,0, inplace=True) 
groupby_states_rate_of_change_np = groupby_states_rate_of_change.to_numpy().astype('float').T

rolling_mean = 10
groupby_states_rate_of_change_rolling_mean = groupby_states_rate_of_change.rolling(rolling_mean).mean()
groupby_states_rate_of_change_rolling_mean.to_csv(str(cur_path
    /f'Data/us_state_rate_of_change_rolling_mean={rolling_mean}.csv'))

# groupby_states_rate_of_change_rolling_mean_melted = pd.read_csv(str(cur_path
#     /'Data/us_state_rate_of_change_rolling_mean_melted.csv'), index_col=False)
groupby_states_rate_of_change_rolling_mean_melted = pd.read_csv(str(cur_path
    /f'Data/us_state_rate_of_change_rolling_mean={rolling_mean}_melted.csv'), index_col=False)
groupby_states_rate_of_change_rolling_mean_melted.set_index(groupby_states_rate_of_change_rolling_mean_melted.columns[0],
        inplace=True)
groupby_states_rate_of_change_rolling_mean_melted['roc_rolling_mean'] = groupby_states_rate_of_change_rolling_mean_melted['roc_rolling_mean'].astype('float')


## Error: there 1 nan value. I am not sure how nan is computed, but I replace nan with 0 for wpnow



groupby_states = df_by_date.groupby('state')["cases"].apply(np.array)
all_states = groupby_states.index.to_numpy()
groupby_states_np =  groupby_states.to_numpy()
states_ts_len = np.array([groupby_states_np[i].shape[0] for i in range(groupby_states_np.shape[0])])
max_states_ts_len = states_ts_len.max()

states_pad_len = max_states_ts_len - states_ts_len
left_padded_states_ts = np.array([np.hstack((np.zeros(p), groupby_states_np[i])) for i, p in enumerate(states_pad_len)])
# groupby_states_np_ts  = to_time_series_dataset(groupby_states_np) # each instance needs to be aligned by date

# ## raw case data
tmp = left_padded_states_ts[:, states_pad_len.max():]
groupby_states_np_ts = to_time_series_dataset(tmp) # each instance needs to be aligned by date
# align time series
groupby_states_np_ts[:, states_pad_len.max():]

### rate of change data 

# groupby_states_np_ts = to_time_series_dataset(groupby_states_rate_of_change_np) # each instance needs to be aligned by date


######## Plotting

# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# sns.set_theme(style="darkgrid")
# # tips = sns.load_dataset("tips")
# # tips['index'] = np.arange(tips.shape[0])
# # ax = sns.pointplot(x="tip", y="day", data=tips, join=False)
# # ax = sns.lineplot(x="tip", y="sex", data=tips)
# ax = sns.lineplot(x="date", y="cases",hue="state", data=df_by_date)
# plt.savefig('us_states_cases.png')
# plt.yscale('log')
# plt.show()
# exit()

############# clustering
# X = random_walks(n_ts=50, sz=32, d=1)
tmp = groupby_states_np_ts
n_clusters = 10
km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=5,
                      random_state=0).fit(tmp)

# cluster_member_num = [ km.cluster_centers_[i].shape[0] for i in range(km.cluster_centers_.shape[0])]
cluster_member_num = np.unique(km.labels_, return_counts=True)

cluster_member_dict = {}
for i in cluster_member_num[0]:
    cluster_member_dict[i] = all_states[np.where(km.labels_ == i)[0]]

# km.cluster_centers_
# km.labels_
# km.inertia_

####### plotting 
########## raw cases

cluster_group = []
for i in cluster_member_dict.keys():
    tmp = df_by_date[df_by_date['state'].isin(cluster_member_dict[i])]
    cluster_group.append(tmp)

# cluster_1 = df_by_date[df_by_date['state'].isin(cluster_member_dict[0])]
# cluster_2 = df_by_date[df_by_date['state'].isin(cluster_member_dict[1])]
# cluster_3 = df_by_date[df_by_date['state'].isin(cluster_member_dict[2])]


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# for i, ct in enumerate([cluster_1, cluster_2, cluster_3]):
for i, ct in enumerate(cluster_group):
    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(x="date", y="cases",hue="state", data=ct)
    plt.title(f'cluster_{i}_{n_clusters}')
    plt.yscale('log')
    plt.savefig(str(cur_path/ f'Outputs/cluster_{i}_cases_{n_clusters}.png'))
    plt.close()
    # plt.show()


########## rate of change 

# cluster_group = []
# for i in cluster_member_dict.keys():
#     tmp = df_by_date[df_by_date['state'].isin(cluster_member_dict[i])]
#     cluster_group.append(tmp)

# # cluster_1 = df_by_date[df_by_date['state'].isin(cluster_member_dict[0])]
# # cluster_2 = df_by_date[df_by_date['state'].isin(cluster_member_dict[1])]
# # cluster_3 = df_by_date[df_by_date['state'].isin(cluster_member_dict[2])]


# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # for i, ct in enumerate([cluster_1, cluster_2, cluster_3]):
# for i, ct in enumerate(cluster_group):
#     sns.set_theme(style="darkgrid")
#     ax = sns.lineplot(x="date", y="cases",hue="state", data=ct)
#     plt.title(f'cluster_{i}_{n_clusters}')
#     plt.yscale('log')
#     plt.savefig(str(cur_path/ f'Outputs/cluster_{i}_cases_groupby_rate_of_change_{n_clusters}.png'))
#     plt.close()
    
####### plotting 
########## cases groupby rate of change

# cluster_group = []
# for i in cluster_member_dict.keys():
#     tmp = groupby_states_rate_of_change_melted[groupby_states_rate_of_change_melted['state'].isin(cluster_member_dict[i])]
#     cluster_group.append(tmp)

# # cluster_1 = df_by_date[df_by_date['state'].isin(cluster_member_dict[0])]
# # cluster_2 = df_by_date[df_by_date['state'].isin(cluster_member_dict[1])]
# # cluster_3 = df_by_date[df_by_date['state'].isin(cluster_member_dict[2])]


# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # for i, ct in enumerate([cluster_1, cluster_2, cluster_3]):
# for i, ct in enumerate(cluster_group):
#     sns.set_theme(style="darkgrid")
#     # ax = sns.lineplot(x="date", y="cases",hue="state", data=ct)
#     ax = sns.lineplot(x="date", y="roc",hue="state", data=ct)
#     plt.title(f'cluster_{i}_{n_clusters}')
#     plt.yscale('log')
#     plt.savefig(str(cur_path/ f'Outputs/cluster_{i}_rate_of_change_{n_clusters}.png'))
#     plt.close()
#     # plt.show()

###### plotting 
######### cases groupby rate of change + rolling mean 

# cluster_group = []
# for i in cluster_member_dict.keys():
#     tmp = groupby_states_rate_of_change_rolling_mean_melted[groupby_states_rate_of_change_rolling_mean_melted['state'].isin(cluster_member_dict[i])]
#     cluster_group.append(tmp)

# # cluster_1 = df_by_date[df_by_date['state'].isin(cluster_member_dict[0])]
# # cluster_2 = df_by_date[df_by_date['state'].isin(cluster_member_dict[1])]
# # cluster_3 = df_by_date[df_by_date['state'].isin(cluster_member_dict[2])]


# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np

# # for i, ct in enumerate([cluster_1, cluster_2, cluster_3]):
# for i, ct in enumerate(cluster_group):
#     sns.set_theme(style="darkgrid")
#     # ax = sns.lineplot(x="date", y="cases",hue="state", data=ct)
#     ax = sns.lineplot(x="date", y="roc_rolling_mean",hue="state", data=ct)
#     plt.title(f'cluster_{i}_{n_clusters}')
#     plt.yscale('log')
#     plt.savefig(str(cur_path/ f'Outputs/cluster={i}_rate_of_change_rolling_mean={rolling_mean}_n_cluster={n_clusters}.png'))
#     plt.close()
#     # plt.show()

