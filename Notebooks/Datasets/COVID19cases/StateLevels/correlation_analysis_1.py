"""Correlation of intra-temporal dependencies"""

# * here> lets do data analysis of the dataset.
#     * see "interplay.." paper
#     * inter-temporal correlation.
#         * n * n matrix
#     * here> intra-temporal correlation.
#         * pearson correlation week by weekl. 
#         * n * n matrix
#     * correlation between features.

from Models.Preprocessing import *
from scipy.stats import  pearsonr
from itertools import product 
import pandas as pd
import numpy as np


window_size = 14
state = "Washington"
# state = "Florida"
# window_length = window_size * 51
window_length = int(case_by_date_florida_np.shape[0]/window_size) * window_size

case_by_date_florida = df_by_date[df_by_date["state"] == state]
case_by_date_florida_np = case_by_date_florida.to_numpy()[
    :, 2:].astype("float")

tmp = case_by_date_florida_np[:window_length]
x = tmp.reshape(-1,window_size)

# find all combination of window
clique_edges = list(product(range(x.shape[0]), range(x.shape[0])))

clique_edges_ps_corr = []
# for i,j in clique_edges[1]:
for i,j in clique_edges[:2]:
    # print(i,j)
    clique_edges_ps_corr.append(pearsonr(x[i, :],x[j, :])[0])
    print(pearsonr(x[i, :],x[j, :])[0])
    print(i,j)

clique_edges_ps_corr_np = np.array(clique_edges_ps_corr).reshape(x.shape[0], x.shape[0])
clique_edges_ps_corr_df = pd.DataFrame(clique_edges_ps_corr_np)

i(mport matplotlib.pyplot as plt
plt.matshow(clique_edges_ps_corr_df)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('clique pearson matrix')
plt.show()

