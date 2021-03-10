import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from global_params import *
from pprint import pprint
from pandas import DataFrame, Series
from pandas import concat
from numpy import array
from numpy import concatenate
import click


@click.command()
@click.argument('n_out')
@click.argument('n_in')
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--save', is_flag=True)
@click.option('--aggr', is_flag=True)
@click.option('--load_data_n', default=0, type=int)
def run_func(**kwargs):

    load_data_n              = kwargs['load_data_n']
    multi_step_folder        = 'MultiStep' if kwargs['is_multi_step_prediction'] else 'OneStep'
    aggr_op                  = 'mean'

    kwargs['multi_step_folder'] = multi_step_folder
    kwargs['aggr_op'] = aggr_op
    
    if load_data_n == 1:
        data, plot_params = load_data_1(**kwargs)
    elif load_data_n == 2:
        raise NotImplementedError
        data = load_data_2(**kwargs)
    elif load_data_n == 3:
        raise NotImplementedError
        data = load_data_3(**kwargs)
    else:
        raise ValueError("please load one of the availble data. see load_data_n args")
    barplot(data, plot_params)


def load_data_1(**kwargs):

    n_out                    = kwargs['n_out']
    n_in                     = kwargs['n_in']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    is_save                  = kwargs['save']
    is_aggr                  = kwargs['aggr']
    load_data_n              = kwargs['load_data_n']
    multi_step_folder        = kwargs['multi_step_folder']
    aggr_op                  = kwargs['aggr_op']

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

    plot_kwargs = {
            'multi_step_folder': multi_step_folder,
            'n_out':             n_out,
            'n_in':              n_in,
            'x':                 'state',
            'y':                 'mse',
            'hue':               'model'
            }

    return all_model_state_mse_df, plot_kwargs

def barplot(data, plot_kwargs):

    sns.set(font_scale=0.5)
    sns.set(rc={'figure.figsize':(20,13)})

    ax = sns.barplot(x=plot_kwargs['x'], y=plot_kwargs['y'], hue=plot_kwargs['hue'], data=data)

    # save_path = '<Onestep>/<PredictNextN>/<WindowLengthN>/Image/barplot_<x-axis>_<y-axis>_<legend>'
    save_path = 'Outputs/DrZhu/{}/PredictNext{}/WindowLength{}/Images/barplot_{}_{}_{}.png'.format(*list(plot_kwargs.values()))
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    ax.set_yscale('log')
    Path(save_path).parents[0].mkdir(parents=True,exist_ok=True)
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    run_func()
    # data = load_data_1()
    # barplot(data)

