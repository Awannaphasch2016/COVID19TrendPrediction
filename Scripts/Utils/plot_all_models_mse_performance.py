import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from global_params import *
from pprint import pprint
from pandas import DataFrame, Series
from pandas import concat
from numpy import array
from numpy import concatenate

def load_data():
    sns.set_theme(style="whitegrid")

    # data = sns.load_dataset("tips")

    file_path ='Outputs/DrZhu/all_performance_table_df_1.csv'
    data = read_csv(str(Path(BASEPATH) / file_path))
    models_num = data.shape[1] -1
    new_cols = data.columns.tolist()
    new_cols[0] = 'state'
    data.columns =  new_cols
    data_dict = data.to_dict()

    all_col_names = ['state','model', 'mse']
    all_model_mse = []
    all_states = []
    for i, (key, val) in enumerate(data_dict.items()):
        if key != 'state':
            col_1 = {'model': [key for _ in  list(val.keys())]}
            col_2 = {'mse': list(val.values())}

            # col_1 = [key for _ in  list(val.keys())]
            # col_2 = list(val.values())
            col_np = array([col_1['model'], col_2['mse']]).T
            # print(col_1)
            # print(col_2)
            all_model_mse.append(col_np)
        else:
            col_1 = {'state': [key for key in  list(val.values())]}
            all_states = list(col_1.values())

    all_model_mse_np = array(all_model_mse).reshape(-1,2)
    all_states_np = array(all_states * models_num).reshape(-1, 1)
    all_model_state_mse_np = concatenate([all_states_np, all_model_mse_np], axis=1)
    all_model_state_mse_df = DataFrame(all_model_state_mse_np, columns=all_col_names)
    all_model_state_mse_df = all_model_state_mse_df.astype({all_col_names[-1]: float})

    return all_model_state_mse_df

def barplot(data):

    # ax = sns.barplot(x="day", y="total_bill", hue="sex", data=data)
    # plt.show()
    print(data)
    print(data.dtypes)
    sns.set(font_scale=0.7)
    ax = sns.barplot(x="state", y="mse", hue="model", data=data)
    for item in ax.get_xticklabels():
        item.set_rotation(50)
    ax.set_yscale('log')
    plt.show
    plt.show()

if __name__ == '__main__':
    data = load_data()
    barplot(data)

