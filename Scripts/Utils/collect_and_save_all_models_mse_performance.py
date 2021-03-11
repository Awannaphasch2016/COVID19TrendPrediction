"""this script is used to concatenate all model mse performance. Note only mse performance."""
import pandas as pd
import numpy as np
from pathlib import Path
from global_params import *
import click

# @click.group()
# def cli():
#     pass

@click.command()
@click.argument('n_out')
@click.argument('n_in')
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--save', is_flag=True)
@click.option('--aggr', is_flag=True)
def run_func(**kwargs):
    # print(kwargs)
    # exit()
    collect_and_save_all_models_mse_performance(**kwargs)
     
def collect_and_save_all_models_mse_performance(**kwargs):
    def load_csv():
        csv = pd.read_csv(str(Path(BASEPATH) / "Data/Raw/COVID19Cases/StateLevels/us-states.csv"))
        return csv
    df = load_csv()
    all_states = np.unique(df.state.values).tolist()
    all_performance_table = []
    # n_out = 5

    n_out              = kwargs['n_out']
    n_in              = kwargs['n_in']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    multi_step_folder        = 'MultiStep' if is_multi_step_prediction else 'OneStep'
    is_save                  = kwargs['save']
    is_aggr                  = kwargs['aggr']
    aggr_op                  = 'mean'

    for state_name in all_states:

        kwargs = {'params':[multi_step_folder,n_out,n_in, state_name, state_name]}
        # print(kwargs)
        # exit()

        notes = []
        performance_table = []
        for model_name in ALL_BASELINES_MODELS:
            model_name = 'previous_val' if model_name == 'previous_day' else model_name
            model_name = "_".join(model_name.split(' '))
            params = kwargs['params'].copy()
            params.append(model_name)

            # performance_result = pd.read_csv(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH.format(*params))))
            # performance_result.index = [model_name]
            # # performance_table.append(performance_result['mse'])
            # x = performance_result['mse']
            # x.index = [state_name]
            # performance_table.append(x)
            # x = pd.DataFrame(x)
            # print(x)
            # y = x.copy() 
            # print(y)
            # all_performance_table_df = pd.concat([x,y], axis=0)
            # print(all_performance_table_df)
            # exit()
            try:
                performance_result = pd.read_csv(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH.format(*params))))
                # print(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH.format(*params))))
                performance_result.index = [model_name]
                # performance_table.append(performance_result['mse'])
                x = pd.DataFrame(performance_result['mse'])
                x.columns = [state_name]
                performance_table.append(x)
                # print(x)
            except:
                try:
                    FRAME_PERFORMANCE_PATH_2 =  "/Outputs/Models/Performances/Baselines/{}/PredictNext{}/WindowLength{}/{}/{}_{}_model_performance.csv"
                    # print(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH_2.format(*params))))
                    performance_result = pd.read_csv(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH_2.format(*params))))
                    performance_result.index = [model_name]
                    # performance_table.append(performance_result['mse'])
                    x = pd.DataFrame(performance_result['mse'])
                    x.columns = [state_name]
                    performance_table.append(x)
                    # print(x)
                except:
                    pass
                    print(f'{model_name} performance result is not recorded')

        if len(performance_table) > 0:
            performance_table_df = pd.concat(performance_table)
            # print(performance_table_df)
            all_performance_table.append(performance_table_df)
            # print(performance_table_df)
            # exit()

    # print(all_performance_table)
    # exit()
    all_performance_table_df = pd.concat(all_performance_table, axis=1)
    # all_performance_table.append(all_performance_table_df)
    file_extension = 'csv'
    # save_path = f'Outputs/DrZhu/all_performance_table_df_n_out{n_out}_n_in{n_in}.{file_extension}'
    save_path = 'Outputs/DrZhu/{}/PredictNext{}/WindowLength{}/all_performance_table_df.{}'.format(multi_step_folder, n_out,n_in, file_extension)
    all_performance_table_df = all_performance_table_df.transpose()
    print('concat result')
    print(all_performance_table_df)
    print('=============================================================')

    output_performance_table_df = all_performance_table_df

    if is_aggr: 
        save_path = Path(save_path)
        file_name = save_path.stem + f'_mean.{file_extension}'
        save_path = str(save_path.parents[0] / file_name)
        print(f'aggr result with {aggr_op} operation')
        output_performance_table_df = all_performance_table_df.agg(['mean'])
        print(output_performance_table_df)
        print('=============================================================')
    if is_save: 
        output_performance_table_df.to_csv(save_path)
        print(f'save to {save_path}')
        print('=============================================================')
    else:
        print(f'outputs are not saved')

if __name__ == '__main__':
    run_func()
