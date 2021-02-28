from Utils.eval_funcs import * 
from numpy import array
from Models.Preprocessing.us_state import *

def split_by_predict_next_n_day(data, n):
    reshape_data = []
    i = 0 
    while i + n <= case_by_date_florida_test.shape[0]:
        x = case_by_date_florida_test[i:i+n].reshape(-1)
        reshape_data.append(x)
        # print(case_by_date_florida_test[i:i+7])
        i += 1
    return array(reshape_data)

def previous_day_forcast(case_by_date_per_states_test,n):
    reshape_data = []
    i = 0 
    while i + n <= case_by_date_per_states_test.shape[0]:
        x = case_by_date_per_states_test[i].reshape(-1)
        reshape_data.append([x] * n)
        # print(case_by_date_florida_test[i:i+7])
        i += 1
    return array(reshape_data).reshape(-1, n)

# cur_val = case_by_date_florida_test[1:]
# pred_val = case_by_date_florida_test[:-1]

data_np =  split_by_predict_next_n_day(case_by_date_florida_test, n=5)
print(data_np.shape)

pred_np = previous_day_forcast(case_by_date_florida_test, n=5)
print(pred_np.shape)

print(mape(data_np, pred_np))

