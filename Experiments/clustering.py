# from Models.Preprocessing.us_state import *

import pandas as pd 
import numpy as np
import pathlib
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

df_by_date = pd.read_csv('us_state_clean.csv', index_col=False)
case_by_date_florida = pd.read_csv("fl_case.csv", index_col=False)


groupby_states = df_by_date.groupby('state')["cases"].apply(np.array)
all_states = groupby_states.index.to_numpy()
groupby_states_np =  groupby_states.to_numpy()
states_ts_len = np.array([groupby_states_np[i].shape[0] for i in range(groupby_states_np.shape[0])])
max_states_ts_len = states_ts_len.max()

states_pad_len = max_states_ts_len - states_ts_len
left_padded_states_ts = np.array([np.hstack((np.zeros(p), groupby_states_np[i])) for i, p in enumerate(states_pad_len)])
# groupby_states_np_ts  = to_time_series_dataset(groupby_states_np) # each instance needs to be aligned by date
tmp = left_padded_states_ts[:, states_pad_len.max():]
groupby_states_np_ts = to_time_series_dataset(tmp) # each instance needs to be aligned by date

# align time series
groupby_states_np_ts[:, states_pad_len.max():]

# ############# plot
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
km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,
                      random_state=0).fit(tmp)

# cluster_member_num = [ km.cluster_centers_[i].shape[0] for i in range(km.cluster_centers_.shape[0])]
cluster_member_num = np.unique(km.labels_, return_counts=True)

cluster_member_dict = {}
for i in cluster_member_num[0]:
    cluster_member_dict[i] = all_states[np.where(km.labels_ == i)[0]]

km.cluster_centers_
km.inertia_

####### plotting 

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
#     plt.title(f'cluster_{i}')
#     plt.savefig(f'cluster_{i}_cases.png')
#     plt.yscale('log')
#     plt.close()
    
############### modeling


