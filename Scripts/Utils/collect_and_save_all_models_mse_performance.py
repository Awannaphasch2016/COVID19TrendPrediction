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
@click.argument('pred_length')
@click.option('--is_multi_step_prediction', is_flag=True)
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
    # pred_length = 5
    pred_length = kwargs['pred_length']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
    for state_name in all_states:

        kwargs = {'params':[multi_step_folder,pred_length, state_name, state_name]}
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
                performance_result.index = [model_name]
                # performance_table.append(performance_result['mse'])
                x = pd.DataFrame(performance_result['mse'])
                x.columns = [state_name]
                performance_table.append(x)
                # print(x)
            except:
                try:
                    FRAME_PERFORMANCE_PATH_2 =  "/Outputs/Models/Performances/Baselines/{}/PredictNext{}/{}/{}_{}_model_performance.csv"
                    performance_result = pd.read_csv(str(Path(BASEPATH + FRAME_PERFORMANCE_PATH_2.format(*params))))
                    performance_result.index = [model_name]
                    # performance_table.append(performance_result['mse'])
                    x = pd.DataFrame(performance_result['mse'])
                    x.columns = [state_name]
                    performance_table.append(x)
                    # print(x)
                except:
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
    save_path = f'Outputs/DrZhu/all_performance_table_df_{pred_length}.csv'
    all_performance_table_df = all_performance_table_df.transpose()
    all_performance_table_df.to_csv(save_path)
    print(all_performance_table_df)
    print(f'save to {save_path}')

    print('done')
if __name__ == '__main__':
    run_func()
