from Utils import * 
from Models.Preprocessing.us_state import *
from pathlib import Path
import pandas
import click
from pprint import pprint


@click.command()
@click.argument('n_in', type=int)
@click.argument('n_out', type=int)
@click.option('--test_mode', is_flag=True)
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--model_param_epoch', type=int)
# @click.option('--model_param_tmp', type=int)
@click.pass_context
def gamma_apply_model_to_all_states(ctx, **kwargs):

    n_in                     = kwargs['n_in']
    n_out                    = kwargs['n_out']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    test_mode                = kwargs['test_mode']
    model_param_epoch        = kwargs['model_param_epoch']
    # model_param_tmp       = kwargs['model_param_tmp']

    model_params = {}

    if model_param_epoch is not None:
        model_params['epoch'] = model_param_epoch

    # if model_param_tmp is not None:
    #     model_params['tmp'] = model_param_tmp

    non_cli_params           = ctx.obj['non_cli_params']

    data                     = non_cli_params['data']
    model                    = non_cli_params['model']
    base_path                = non_cli_params['data']
    frame_performance_path   = non_cli_params['frame_performance_path']
    frame_pred_val_path      = non_cli_params['frame_pred_val_path']
    plot_path                = non_cli_params['plot_path'] 

    pprint(kwargs)
    pprint(non_cli_params)
    model, model_name = model

    def _add_file_suffix(file_path, file_suffix):
        if len(specified_path.split('.')) == 2:
            file_path = file_path.split('.')[0] + file_suffix + '.' + file_path.split('.')[-1]
        else:
            raise NotImplementedError
        return file_path

    for i in all_states:
        # cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)
        try:
            cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction, model_params)
        except TypeError:
            assert len(model_params.keys()) == 0, f'{model_name} doesn"t accept any model_params.'
            cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)

        multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
        model_params_list = ['']
        if len(model_params.keys()) > 0:
            for key,value in model_params.items():
                model_params_list.append(f'{key}={value}')
        model_params_str = '_'.join(model_params_list)

        specified_path = None if frame_performance_path is None else BASEPATH + frame_performance_path.format(multi_step_folder,n_out, n_in, i,i, model_name)
        specified_path = _add_file_suffix(specified_path, model_params_str)
        print(multi_step_folder,n_out, n_in, i,i, model_name)
        print(specified_path)
        exit()
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        beta_frame_performance(
            eval_metric_df,
            save_path=specified_path,
        )
        

        specified_path = None if frame_pred_val_path is None else BASEPATH + frame_pred_val_path.format(multi_step_folder, n_out, n_in, i,i, model_name)
        specified_path = _add_file_suffix(specified_path, model_params_str)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)
        
        beta_frame_pred_val(
            cur_val.reshape(-1),
            array(pred_val).reshape(-1),
            save_path=specified_path,
        )

        pred_val_df = pandas.read_csv(specified_path)
        cur_val, pred_val = pred_val_df['y_test'].tolist(), pred_val_df['y_pred'].tolist()

        specified_path = None if plot_path is None else BASEPATH + plot_path.format(multi_step_folder,n_out,n_in, i,i, model_name)
        specified_path = _add_file_suffix(specified_path, model_params_str)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        beta_plot(
            cur_val,
            pred_val,
            save_path=specified_path,
            display=False,
        )

        if test_mode:
            exit()

def beta_apply_model_to_all_states(
    data, model,n_in, n_out, is_multi_step_prediction, base_path, frame_performance_path=None, frame_pred_val_path=None, 
    plot_path=None, test_mode=False
):
    model, model_name = model
    for i in all_states:
        cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)
        multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
        specified_path = None if frame_performance_path is None else BASEPATH + frame_performance_path.format(multi_step_folder,n_out,i,i, model_name)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        beta_frame_performance(
            eval_metric_df,
            save_path=specified_path,
        )
        

        specified_path = None if frame_pred_val_path is None else BASEPATH + frame_pred_val_path.format(multi_step_folder, n_out,i,i, model_name)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)
        
        beta_frame_pred_val(
            cur_val.reshape(-1),
            array(pred_val).reshape(-1),
            save_path=specified_path,
        )

        pred_val_df = pandas.read_csv(specified_path)
        cur_val, pred_val = pred_val_df['y_test'].tolist(), pred_val_df['y_pred'].tolist()

        specified_path = None if plot_path is None else BASEPATH + plot_path.format(multi_step_folder,n_out, i,i, model_name)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        beta_plot(
            cur_val,
            pred_val,
            save_path=specified_path,
            display=False,
        )

        if test_mode:
            exit()

def apply_model_to_all_states(
    data, model, base_path, frame_performance_path=None, frame_pred_val_path=None, plot_path=None, test_mode=False
):
    model, model_name = model
    
    for i in all_states:

        cur_val, pred_val, mse_val, mape_val, rmse_val, r2_val = model(data,i)

        specified_path = None if frame_performance_path is None else BASEPATH + frame_performance_path.format(i,i, model_name)
        # print(frame_performance_path)
        # print(specified_path)
        # exit()

        frame_performance(
            mse_val,
            mape_val,
            rmse_val,
            r2_val,
            save_path=specified_path,
            # save_path=BASEPATH + frame_performance_path.format(i,i)
            # + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_performance.csv",
        )
        

        specified_path = None if frame_pred_val_path is None else BASEPATH + frame_pred_val_path.format(i,i, model_name)
        # print(frame_pred_val_path)

        frame_pred_val(
            cur_val.reshape(-1),
            pred_val.reshape(-1),
            save_path=specified_path,
            # save_path=BASEPATH + frame_pred_val_path.format(i,i)
            # + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_pred_val.csv",
        )

        specified_path = None if plot_path is None else BASEPATH + plot_path.format( i,i, model_name)
        # print(plot_path)
        # print(specified_path)
        # exit()

        plot(
            cur_val,
            pred_val,
            save_path=specified_path,
            # save_path=BASEPATH + plot_path.format(i,i),
            # + f"/Outputs/Models/Performances/Baselines/{i}_previous_val_forcasting.jpg",
            display=False,
            # display=True,
        )
        if test_mode:
            exit()
