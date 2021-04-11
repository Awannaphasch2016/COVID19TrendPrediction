"""Correlation of intra-temporal dependencies"""

# * here> lets do data analysis of the dataset.
#     * see "interplay.." paper
#     * here> inter-temporal correlation.
#         * n * n matrix
#     * intra-temporal correlation.
#         * pearson correlation week by weekl. 
#         * n * n matrix
#     * correlation between features.

from Models.Preprocessing import *
from scipy.stats import  pearsonr
from itertools import product
from itertools import permutations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_by_date['date'] = pd.to_datetime(df_by_date['date'], format='%Y-%m-%d')
# tmp = df_by_date.groupby(['state', pd.Grouper(key="date", freq="1W")])["cases"].apply(np.array)
tmp = df_by_date.groupby([ pd.Grouper(key="date", freq="1W"), 'state'])["cases"].apply(np.array)
tmp1 = pd.Series.to_frame(tmp,name='node')
tmp2 = tmp1[tmp1['node'].map(len) == 7 ]
date_indices = np.unique(tmp2.index.get_level_values('date'))

# here> apply correlation on each states => if states doesn't enough case (doesn't exist), value is np.nan
# tmp2.iloc[tmp2.index.get_level_values('date') == date_indices[10]]['node'].sort_index()
# np.array([i for i in tmp2.iloc[tmp2.index.get_level_values('date') == date_indices[-1]]['node'].sort_index().values])
# inter_node_ps_corr = np.array([])
weekly_inter_node = []
# len(weekly_inter_node)
# weekly_inter_node[0]
for d in date_indices:
    tmp = np.array([i for i in tmp2.iloc[tmp2.index.get_level_values('date') == d]['node'].sort_index().values])
    weekly_inter_node.append(tmp)

    # tmp = np.array([i for i in tmp2.iloc[tmp2.index.get_level_values('date') == d]['node'].sort_index().values])
    # if inter_node_ps_corr.shape[0] == 0:
    #     inter_node_ps_corr = tmp
    # else:
    #     inter_node_ps_corr = np.vstack([inter_node_ps_corr, tmp])

def corr_plot():
    week_1 = weekly_inter_node[10]
    x = week_1
    # find all combination of window
    clique_edges = list(product(range(x.shape[0]), range(x.shape[0])))
    clique_edges_ps_corr = []
    # for i,j in clique_edges[1]:
    for i,j in clique_edges:
        # print(i,j)
        clique_edges_ps_corr.append(pearsonr(x[i, :],x[j, :])[0])
        # print(pearsonr(x[i, :],x[j, :])[0])
        # print(i,j)
    clique_edges_ps_corr_np = np.array(clique_edges_ps_corr).reshape(x.shape[0], x.shape[0])
    clique_edges_ps_corr_df = pd.DataFrame(clique_edges_ps_corr_np)
    # np.array([ np.array(i) for i in tmp2.iloc[tmp2.index.get_level_values('date') == date_indices[-1]].to_numpy()])
    plt.matshow(clique_edges_ps_corr_df)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('clique pearson matrix')
    plt.show()

# line plot with vaweekly_inter_nodes
# 1. create correlation for all permutation of 2 states -> 1 x n where row = week and col = node pair 
# 1. create correlation for all permutation of 2 states -> n x 1 where row = node pair of this week.
# for each 
# 2. repeat 1. until finished. 
week_1 = weekly_inter_node[10]
x = week_1
nodes_permutation_index = list(permutations(range(x.shape[0]), r=2))


from datetime import datetime
# datetime.utcfromtimestamp(date_indices[i_])

weekly_inter_node_ps_corr = []
for i_, week in enumerate(weekly_inter_node):
    # print(date_indices[i_])
    utc = date_indices[i_].astype('<M8[s]').astype(int)
    utc = datetime.utcfromtimestamp(utc)
    utc_str = utc.strftime("%m-%d-%y")
    print(f'week = {i_}: {utc_str}')
    if week.shape[0] == 55:
        for n_1,n_2 in nodes_permutation_index:
            tmp = pearsonr(week[n_1,:], week[n_2,:])[0]
            # weekly_inter_node_ps_corr.append((utc_str,tmp))
            weekly_inter_node_ps_corr.append((i_, utc_str, tmp))
            print(tmp)

weekly_inter_node_ps_corr_np = np.array(weekly_inter_node_ps_corr)
weekly_inter_node_ps_corr_df = pd.DataFrame(weekly_inter_node_ps_corr_np, columns=['Index','Date', 'Corr'])
weekly_inter_node_ps_corr_df['DateOfYear'] = pd.to_datetime(weekly_inter_node_ps_corr_df['Date'], format='%m-%d-%y')
weekly_inter_node_ps_corr_df.set_index('DateOfYear', inplace=True)
weekly_inter_node_ps_corr_df.index = weekly_inter_node_ps_corr_df.index.dayofyear

# weekly_inter_node_ps_corr_df =  weekly_inter_node_ps_corr_df.set_index('DayOfYear')
# weekly_inter_node_ps_corr_df['DayOfYear'] = weekly_inter_node_ps_corr_df['DayOfYear'].astype(int)
# weekly_inter_node_ps_corr_df.T.iloc[0]

# for i,j in nodes_permutation_index:  

import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(12,5))
# seaborn.boxplot(weekly_inter_node_ps_corr_df['Index'].values.astype(int), weekly_inter_node_ps_corr_df['Corr'].values.astype(float), ax=ax)
chart = seaborn.boxplot(weekly_inter_node_ps_corr_df['Date'], weekly_inter_node_ps_corr_df['Corr'].values.astype(float), ax=ax)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
# seaborn.boxplot(weekly_inter_node_ps_corr_df.index, weekly_inter_node_ps_corr_df['Corr'].values, ax=ax)
# seaborn.boxplot(list(range(50)), weekly_inter_node_ps_corr_df['Corr'].values[:50], ax=ax)
plt.show()

