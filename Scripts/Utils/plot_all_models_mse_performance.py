import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from global_params import *
from pprint import pprint
from pandas import DataFrame, Series
from pandas import concat
from numpy import array
from numpy import hstack, vstack
from numpy import concatenate
import click
from itertools import product

@click.group()
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--save', is_flag=True)
@click.option('--display', is_flag=True)
@click.option('--aggr', is_flag=True)
@click.pass_context
def run_func(ctx,**kwargs):
    
    ctx.ensure_object(dict)

    # load_data_n              = kwargs['load_data_n']
    ctx.obj['multi_step_folder'] = 'MultiStep' if kwargs['is_multi_step_prediction'] else 'OneStep'
    ctx.obj['aggr_op']           = 'mean'
    ctx.obj['save']              = kwargs['save']
    ctx.obj['aggr']              = kwargs['aggr']
    ctx.obj['display']              = kwargs['display']
    ctx.obj['plot_func']         = bar_plot

@run_func.command()
@click.pass_context
def load_data_1(ctx, **kwargs):
    """for each WindowLengthN and PredictNextN, x-axis = state and y-axis = mse."""

    load_data_n       = 1
    is_save           = ctx.obj['save']
    is_aggr           = ctx.obj['aggr']
    is_display        = ctx.obj['display']
    multi_step_folder = ctx.obj['multi_step_folder']
    aggr_op           = ctx.obj['aggr_op']
    plot_func         = ctx.obj['plot_func']

    sns.set_theme(style="whitegrid")

    # data = sns.load_dataset("tips")

    file_path ='Outputs/DrZhu/all_performance_table_df_1.csv'
    data = read_csv(str(Path(BASEPATH) / file_path))

    models_num = data.shape[1] -1
    new_cols = data.columns.tolist()
    new_cols[0] = 'state'
    data.columns =  new_cols
    data_dict = data.to_dict()

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

    all_col_names = ['state','model', 'mse']
    all_model_state_mse_np = concatenate([all_states_np, all_model_mse_np], axis=1)
    all_model_state_mse_df = DataFrame(all_model_state_mse_np, columns=all_col_names)
    all_model_state_mse_df = all_model_state_mse_df.astype({all_col_names[-1]: float})
    
    all_n_out_in = product(ALL_WINDOWLENGTHN, ALL_PREDICTNEXTN)
    for n_in,n_out in all_n_out_in:
        plot_kwargs = {
                'load_data_n': load_data_n,
                'multi_step_folder': multi_step_folder,
                'n_out':             n_out,
                'n_in':              n_in,
                'x':                 'state',
                'y':                 'mse',
                'hue':               'model',
                }
        
        save_path = 'Outputs/DrZhu/load_data_n/load_data_{}/{}/PredictNext{}/WindowLength{}/Images/barplot_{}_{}_{}.png'

        data = all_model_state_mse_df
        plot_func(data, save_path, is_save, is_display, plot_kwargs)

        # return all_model_state_mse_df,save_path, plot_kwargs

@run_func.command()
@click.pass_context
def load_data_2(ctx, **kwargs):
    """for each PredictNextN, x-axis = WindowLengthN and y-axis = mse."""

    load_data_n       = 2
    is_save           = ctx.obj['save']
    is_aggr           = ctx.obj['aggr']
    is_display         = ctx.obj['display']
    multi_step_folder = ctx.obj['multi_step_folder']
    aggr_op           = ctx.obj['aggr_op']
    plot_func         = ctx.obj['plot_func']

    all_windowlength_n_aggr_performance = {}

    all_n_out_in = product(ALL_WINDOWLENGTHN, ALL_PREDICTNEXTN)
    for n_in,n_out in all_n_out_in:
        dir_path = 'Outputs/DrZhu/{}/PredictNext{}/WindowLength{}'
        dir_path = Path(BASEPATH) / dir_path.format(multi_step_folder, n_out, n_in)
        # for p in dir_path.rglob("*performance.csv"):
        # print(dir_path)
        # print(dir_path.exists())
        for p in dir_path.rglob(f"*df_{aggr_op}.csv"):
            df = read_csv(p)

            new_cols = df.columns.to_list()
            new_cols.append('n_in')
            new_vals = df.values.reshape(-1).tolist()
            new_vals.append(n_in)
            
            df = DataFrame([new_vals], columns=new_cols)
            all_windowlength_n_aggr_performance.setdefault(n_in, []).append(df)

    cols = list(all_windowlength_n_aggr_performance[1][0].columns)
    tmp = array([])
    for i in all_windowlength_n_aggr_performance.keys():
        for j in all_windowlength_n_aggr_performance[i]:
            j = j.to_numpy().reshape(-1)
            if tmp.reshape(-1).shape[0] == 0:
                tmp = j
            else:
                tmp = vstack([tmp, j])

    data = DataFrame(tmp, columns=cols)
    print('===========')
    # file_path ='Outputs/DrZhu/all_performance_table_df_1.csv'
    # data = read_csv(str(Path(BASEPATH) / file_path))
    # print(data)

    models_num = data.shape[1] - 1 - 1
    new_cols = data.columns.tolist()
    # new_cols[0] = 'state'
    new_cols[0] = 'aggr'
    data.columns =  new_cols
    data_dict = data.to_dict()

    all_model_mse = []
    all_states = []
    # pprint(data_dict)

    assert data.columns[0] == 'aggr'
    assert data.columns[-1] == 'n_in'

    for i, (key, val) in enumerate(data_dict.items()):
        if key not in  [data.columns[0], data.columns[-1]] :
            col_1 = {'model': [key for _ in  list(val.keys())]}
            col_2 = {'mse': list(val.values())}

            # col_1 = [key for _ in  list(val.keys())]
            # col_2 = list(val.values())
            col_np = array([col_1['model'], col_2['mse']]).T
            # print(col_1)
            # print(col_2)
            all_model_mse.append(col_np)
        elif key == data.columns[0]:
            col_1 = {key: [key for key in  list(val.values())]}
            all_aggrs = list(col_1.values())
        elif key == data.columns[-1]:
            col_1 = {key: [str(key) for key in list(val.values())]}
            all_predictnext_n = list(col_1.values())
        else:
            raise ValueError

    all_model_mse_np = array(all_model_mse).reshape(-1,2)
    all_aggrs_np = array(all_aggrs * models_num).reshape(-1, 1)
    all_predictnext_n_np = array(all_predictnext_n * models_num).reshape(-1, 1)
    # print(all_model_mse_np.shape)
    # print(all_aggrs_np.shape)
    # print(all_predictnext_n_np.shape)

    all_col_names = [data.columns[0],'model', 'mse', data.columns[-1]]
    all_model_predictnext_n_mse_np = concatenate([all_aggrs_np, all_model_mse_np, all_predictnext_n_np], axis=1)
    all_model_predictnext_n_mse_df = DataFrame(all_model_predictnext_n_mse_np, columns=all_col_names)
    all_model_predictnext_n_mse_df = all_model_predictnext_n_mse_df.astype({'mse': float})
    # print(all_model_predictnext_n_mse_df)
    # exit()
    
    for n_in in ALL_WINDOWLENGTHN:
        plot_kwargs = {
                'load_data_n': load_data_n,
                'multi_step_folder': multi_step_folder,
                'n_in':             n_in,
                'x':                 'n_in',
                'y':                 'mse',
                'hue':               'model',
                }

        # here> where to save it?
        save_path = 'Outputs/DrZhu/load_data_n/load_data_{}/{}/PredictNext{}/Images/barplot_{}_{}_{}.png'

        data = all_model_predictnext_n_mse_df
        plot_func(data, save_path, is_save, is_display, plot_kwargs)
        
@run_func.command()
@click.pass_context
def load_data_3(ctx, **kwargs):
    """for each WindowLengthN, x-axis = PredictNextN and y-axis = mse."""

    load_data_n       = 3
    is_save           = ctx.obj['save']
    is_aggr           = ctx.obj['aggr']
    is_display         = ctx.obj['display']
    multi_step_folder = ctx.obj['multi_step_folder']
    aggr_op           = ctx.obj['aggr_op']
    plot_func         = ctx.obj['plot_func']

    all_predictnext_n_aggr_performance = {}

    for n_out in ALL_PREDICTNEXTN:
        dir_path = 'Outputs/DrZhu/{}/PredictNext{}'
        dir_path = Path(BASEPATH) / dir_path.format(multi_step_folder, n_out)
        # for p in dir_path.rglob("*performance.csv"):
        # print(dir_path)
        # print(dir_path.exists())
        for p in dir_path.rglob(f"*df_{aggr_op}.csv"):
            df = read_csv(p)

            new_cols = df.columns.to_list()
            new_cols.append('n_out')
            new_vals = df.values.reshape(-1).tolist()
            new_vals.append(n_out)
            
            df = DataFrame([new_vals], columns=new_cols)
            all_predictnext_n_aggr_performance.setdefault(n_out, []).append(df)

    # print(
    #     [j for i in all_predictnext_n_aggr_performance.keys() for j in all_predictnext_n_aggr_performance[i] ]
    #     )
    # exit()
    
    cols = list(all_predictnext_n_aggr_performance[1][0].columns)
    tmp = array([])
    for i in all_predictnext_n_aggr_performance.keys():
        for j in all_predictnext_n_aggr_performance[i]:
            j = j.to_numpy().reshape(-1)
            if tmp.reshape(-1).shape[0] == 0:
                tmp = j
            else:
                try:
                    tmp = vstack([tmp, j])
                except:
                    pass

    data = DataFrame(tmp, columns=cols)
    print('===========')
    # file_path ='Outputs/DrZhu/all_performance_table_df_1.csv'
    # data = read_csv(str(Path(BASEPATH) / file_path))
    # print(data)
    models_num = data.shape[1] - 1 - 1
    new_cols = data.columns.tolist()
    # new_cols[0] = 'state'
    new_cols[0] = 'aggr'
    data.columns =  new_cols
    data_dict = data.to_dict()

    all_model_mse = []
    all_states = []
    # pprint(data_dict)

    assert data.columns[0] == 'aggr'
    assert data.columns[-1] == 'n_out'

    for i, (key, val) in enumerate(data_dict.items()):
        if key not in  [data.columns[0], data.columns[-1]] :
            col_1 = {'model': [key for _ in  list(val.keys())]}
            col_2 = {'mse': list(val.values())}

            # col_1 = [key for _ in  list(val.keys())]
            # col_2 = list(val.values())
            col_np = array([col_1['model'], col_2['mse']]).T
            # print(col_1)
            # print(col_2)
            all_model_mse.append(col_np)
        elif key == data.columns[0]:
            col_1 = {key: [key for key in  list(val.values())]}
            all_aggrs = list(col_1.values())
        elif key == data.columns[-1]:
            col_1 = {key: [str(key) for key in list(val.values())]}
            all_predictnext_n = list(col_1.values())
        else:
            raise ValueError

    all_model_mse_np = array(all_model_mse).reshape(-1,2)
    all_aggrs_np = array(all_aggrs * models_num).reshape(-1, 1)
    all_predictnext_n_np = array(all_predictnext_n * models_num).reshape(-1, 1)
    # print(all_model_mse_np.shape)
    # print(all_aggrs_np.shape)
    # print(all_predictnext_n_np.shape)

    all_col_names = [data.columns[0],'model', 'mse', data.columns[-1]]
    all_model_predictnext_n_mse_np = concatenate([all_aggrs_np, all_model_mse_np, all_predictnext_n_np], axis=1)
    all_model_predictnext_n_mse_df = DataFrame(all_model_predictnext_n_mse_np, columns=all_col_names)
    all_model_predictnext_n_mse_df = all_model_predictnext_n_mse_df.astype({'mse': float})
    # print(all_model_predictnext_n_mse_df)
    # exit()

    for n_out in ALL_PREDICTNEXTN:
        plot_kwargs = {
                'load_data_n': load_data_n,
                'multi_step_folder': multi_step_folder,
                'n_out':             n_out,
                'x':                 'n_out',
                'y':                 'mse',
                'hue':               'model',
                }

        # here> where to save it?
        save_path = 'Outputs/DrZhu/load_data_n/load_data_{}/{}/PredictNext{}/Images/barplot_{}_{}_{}.png'

        data = all_model_predictnext_n_mse_df
        plot_func(data, save_path, is_save, is_display, plot_kwargs)


def bar_plot(data, save_path, is_save, is_display, plot_kwargs):

    sns.set(font_scale=0.5)
    sns.set(rc={'figure.figsize':(20,13)})

    ax = sns.barplot(x=plot_kwargs['x'], y=plot_kwargs['y'], hue=plot_kwargs['hue'], data=data)

    save_path = save_path.format(*list(plot_kwargs.values()))
    # # print(Path(save_path).is_file())
    # exit()
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    ax.set_yscale('log')
    Path(save_path).parents[0].mkdir(parents=True,exist_ok=True)
    print('=============================')
    if is_save:
        print(f'save bar plot to {save_path}')
        plt.savefig(save_path)
    else:
        print('bar plot is not saved')
    print('=============================')
    if is_display:
        plt.show()

if __name__ == '__main__':
    run_func()

