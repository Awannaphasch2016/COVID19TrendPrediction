from Utils import * 
from Models.Preprocessing.us_state import *
from pathlib import Path
import pandas
import click
from pprint import pprint
from Utils.utils import add_file_suffix
from Utils.wandb_utils import log_performance_with_wandb
import ray
import wandb

@ray.remote
def apply_model(state, ctx, **kwargs):
    n_in                     = kwargs['n_in']
    n_out                    = kwargs['n_out']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    test_mode                = kwargs['test_mode']
    model_param_epoch        = kwargs['model_param_epoch']
    is_save                  = kwargs['save_local']
    is_save_wandb            = kwargs['save_wandb']
    is_plot                  = kwargs['plot']
    # model_param_tmp       = kwargs['model_param_tmp']

    # if model_param_tmp is not None:
    #     model_params['tmp'] = model_param_tmp

    non_cli_params = ctx.obj['non_cli_params']
    # model_config_params   = ctx.obj['model_config_params']

    data                     = non_cli_params['data']
    model                    = non_cli_params['model']
    base_path                = non_cli_params['data']
    frame_performance_path   = non_cli_params['frame_performance_path']
    frame_pred_val_path      = non_cli_params['frame_pred_val_path']
    plot_path                = non_cli_params['plot_path'] 

    # pprint(kwargs)
    # pprint(non_cli_params)
    model, model_name = model


    if is_save_wandb:
        os.environ['WANDB_MODE'] = 'run'
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    model_config = {} 
    if model_param_epoch is not None:
        model_config['epoch'] = model_param_epoch

    model_params = {}
    if model_param_epoch is not None:
        model_params['epoch'] = model_config['epoch']

    multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
    model_params_list = ['']
    if len(model_params.keys()) > 0:
        for key,value in model_params.items():
            model_params_list.append(f'{key}={value}')
    model_params_str = '_'.join(model_params_list)

    i = state

    wandb_config = {
            'multi_step_folder': multi_step_folder,
            'PredictNextN': f'PredictNext{n_out}',
            'WindowLengthN': f'WindowLength{n_in}',
            'state': i,
            'model_name': model_name,
            }

    if len(model_config) > 0:
        wandb_config.update(model_config)

    
    config_kwargs = {
        'project': PROJECT_NAME,
        'save_code':True,
        # job_type='train',
        # job_type='train',
        'tags':[
            'Outputs',
            'Models',
            'Performances',
            'Baselines',
            wandb_config['multi_step_folder'],
            wandb_config[f'PredictNextN'],
            wandb_config['WindowLengthN'],
            wandb_config['state'],
            wandb_config['model_name'],
            ],
        'name': model_name + model_params_str,
        'config': wandb_config
        }

    with wandb.init(**config_kwargs) as run:
        # cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)
        try:
            # cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction, model_params)
            cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction, wandb.config)
        except TypeError:
            assert len(model_params.keys()) == 0, f'{model_name} doesn"t accept any model_params.'
            cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)

        log_performance_with_wandb(eval_metric_df, is_save_wandb,**config_kwargs)
        
        # multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
        # model_params_list = ['']
        # if len(model_params.keys()) > 0:
        #     for key,value in model_params.items():
        #         model_params_list.append(f'{key}={value}')
        # model_params_str = '_'.join(model_params_list)

        specified_path = None if frame_performance_path is None else BASEPATH + frame_performance_path.format(multi_step_folder,n_out, n_in, i,i, model_name)
        specified_path = add_file_suffix(specified_path, model_params_str)
        print(multi_step_folder,n_out, n_in, i,i, model_name)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        beta_frame_performance(
            eval_metric_df,
            save_path=specified_path,
            is_save=is_save
        )
        
        specified_path = None if frame_pred_val_path is None else BASEPATH + frame_pred_val_path.format(multi_step_folder, n_out, n_in, i,i, model_name)
        specified_path = add_file_suffix(specified_path, model_params_str)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)
        
        beta_frame_pred_val(
            cur_val.reshape(-1),
            array(pred_val).reshape(-1),
            save_path=specified_path,
            is_save=is_save
        )
        
        
        y_test = cur_val.reshape(-1)
        y_pred = array(pred_val).reshape(-1)
        pred_val_df = DataFrame(
            np.array([y_test, y_pred]).T,
            columns=["y_test", "y_pred"],
        )
        cur_val, pred_val = pred_val_df['y_test'].tolist(), pred_val_df['y_pred'].tolist()

        specified_path = None if plot_path is None else BASEPATH + plot_path.format(multi_step_folder,n_out,n_in, i,i, model_name)
        specified_path = add_file_suffix(specified_path, model_params_str)
        parent_dir = '/'.join(specified_path.split('/')[:-1])
        print(parent_dir)
        Path(parent_dir).mkdir(parents=True,exist_ok=True)

        beta_plot(
            cur_val,
            pred_val,
            save_path=specified_path,
            display=is_plot,
            is_save=is_save
        )

        if test_mode:
            exit()

@click.command()
@click.argument('n_in', type=int)
@click.argument('n_out', type=int)
@click.option('--test_mode', is_flag=True)
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--save_local', is_flag=True)
@click.option('--save_wandb', is_flag=True)
@click.option('--plot', type=int)
@click.option('--model_param_epoch', type=int)
@click.option('--is_distributed', is_flag=True)
@click.option('--num_gpus', type=int)
@click.option('--num_cpus', type=int)
@click.pass_context
def delta_apply_model_to_all_states(ctx, **kwargs):
    if kwargs['is_distributed']:
        ray.init(num_gpus=kwargs['num_gpus'], num_cpus=kwargs['num_cpus'])
        futures = [apply_model.remote(i, ctx, **kwargs) for i in all_states]
        ray.get(futures)
    else:
        gamma_apply_model_to_all_states_no_click(ctx, **kwargs)
        

def gamma_apply_model_to_all_states_no_click(ctx, **kwargs):
    n_in                     = kwargs['n_in']
    n_out                    = kwargs['n_out']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    test_mode                = kwargs['test_mode']
    model_param_epoch        = kwargs['model_param_epoch']
    is_save                  = kwargs['save_local']
    is_save_wandb            = kwargs['save_wandb']
    is_plot                  = kwargs['plot']
    # model_param_tmp       = kwargs['model_param_tmp']

    # if model_param_tmp is not None:
    #     model_params['tmp'] = model_param_tmp

    non_cli_params           = ctx.obj['non_cli_params']

    data                     = non_cli_params['data']
    model                    = non_cli_params['model']
    base_path                = non_cli_params['data']
    frame_performance_path   = non_cli_params['frame_performance_path']
    frame_pred_val_path      = non_cli_params['frame_pred_val_path']
    plot_path                = non_cli_params['plot_path'] 

    # pprint(kwargs)
    # pprint(non_cli_params)
    model, model_name = model

    if is_save_wandb:
        os.environ['WANDB_MODE'] = 'run'
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    model_config = {} 
    if model_param_epoch is not None:
        model_config['epoch'] = model_param_epoch

    model_params = {}
    if model_param_epoch is not None:
        model_params['epoch'] = model_config['epoch']
    multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
    model_params_list = ['']
    if len(model_params.keys()) > 0:
        for key,value in model_params.items():
            model_params_list.append(f'{key}={value}')
    model_params_str = '_'.join(model_params_list)


    # def add_file_suffix(file_path, file_suffix):
    #     if len(specified_path.split('.')) == 2:
    #         file_path = file_path.split('.')[0] + file_suffix + '.' + file_path.split('.')[-1]
    #     else:
    #         raise NotImplementedError
    #     return file_path

    for i in all_states:

        wandb_config = {
                'multi_step_folder': multi_step_folder,
                'PredictNextN': f'PredictNext{n_out}',
                'WindowLengthN': f'WindowLength{n_in}',
                'state': i,
                'model_name': model_name,
                }

        if len(model_config) > 0:
            wandb_config.update(model_config)
    
        # print(wandb_config)
        # exit()

        config_kwargs = {
            'project': PROJECT_NAME,
            'save_code':True,
            # job_type='train',
            # job_type='train',
            'tags':[
                'Outputs',
                'Models',
                'Performances',
                'Baselines',
                wandb_config['multi_step_folder'],
                wandb_config[f'PredictNextN'],
                wandb_config['WindowLengthN'],
                wandb_config['state'],
                wandb_config['model_name'],
                ],
            'name': model_name + model_params_str,
            'config': wandb_config
            }

        with wandb.init(**config_kwargs) as run:
            # cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)
            try:
                # cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction, model_params)
                cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction, wandb.config)
            except TypeError:
                assert len(model_params.keys()) == 0, f'{model_name} doesn"t accept any model_params.'
                cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction)
            except Exception as e:
                raise ValueError(e)

            log_performance_with_wandb(eval_metric_df, is_save_wandb,**config_kwargs)
            
            # multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
            # model_params_list = ['']
            # if len(model_params.keys()) > 0:
            #     for key,value in model_params.items():
            #         model_params_list.append(f'{key}={value}')
            # model_params_str = '_'.join(model_params_list)

            specified_path = None if frame_performance_path is None else BASEPATH + frame_performance_path.format(multi_step_folder,n_out, n_in, i,i, model_name)
            specified_path = add_file_suffix(specified_path, model_params_str)
            print(multi_step_folder,n_out, n_in, i,i, model_name)
            parent_dir = '/'.join(specified_path.split('/')[:-1])
            print(parent_dir)
            Path(parent_dir).mkdir(parents=True,exist_ok=True)

            beta_frame_performance(
                eval_metric_df,
                save_path=specified_path,
                is_save=is_save
            )
            
            specified_path = None if frame_pred_val_path is None else BASEPATH + frame_pred_val_path.format(multi_step_folder, n_out, n_in, i,i, model_name)
            specified_path = add_file_suffix(specified_path, model_params_str)
            parent_dir = '/'.join(specified_path.split('/')[:-1])
            print(parent_dir)
            Path(parent_dir).mkdir(parents=True,exist_ok=True)
            
            beta_frame_pred_val(
                cur_val.reshape(-1),
                array(pred_val).reshape(-1),
                save_path=specified_path,
                is_save=is_save
            )
        
            y_test = cur_val.reshape(-1)
            y_pred = array(pred_val).reshape(-1)
            pred_val_df = DataFrame(
                np.array([y_test, y_pred]).T,
                columns=["y_test", "y_pred"],
            )
            cur_val, pred_val = pred_val_df['y_test'].tolist(), pred_val_df['y_pred'].tolist()

            specified_path = None if plot_path is None else BASEPATH + plot_path.format(multi_step_folder,n_out,n_in, i,i, model_name)
            specified_path = add_file_suffix(specified_path, model_params_str)
            parent_dir = '/'.join(specified_path.split('/')[:-1])
            print(parent_dir)
            Path(parent_dir).mkdir(parents=True,exist_ok=True)

            beta_plot(
                cur_val,
                pred_val,
                save_path=specified_path,
                display=is_plot,
                is_save=is_save
            )

            if test_mode:
                exit()


@click.command()
@click.argument('n_in', type=int)
@click.argument('n_out', type=int)
@click.option('--test_mode', is_flag=True)
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--save_local', is_flag=True)
@click.option('--save_wandb', is_flag=True)
@click.option('--plot', type=int)
@click.option('--model_param_epoch', type=int)
# @click.option('--model_param_tmp', type=int)
@click.pass_context
def gamma_apply_model_to_all_states(ctx, **kwargs):
    gamma_apply_model_to_all_states_no_click(ctx, **kwargs)

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
