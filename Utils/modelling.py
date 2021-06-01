from Utils import * 
from Models.Preprocessing.us_state import *
from pathlib import Path
import pandas
import click
from pprint import pprint
from Utils.utils import add_file_suffix
from Utils.wandb_utils import log_performance_with_wandb
from Utils.eval_funcs import *
from Utils.plotting import *
import ray
import wandb
import yaml

def prepare_apply_model(ctx, **kwargs):

    n_in                     = kwargs['n_in']
    n_out                    = kwargs['n_out']
    is_multi_step_prediction = kwargs['is_multi_step_prediction']
    test_mode                = kwargs['test_mode']
    # model_param_epoch        = kwargs['model_param_epoch']
    is_save                  = kwargs['save_local']
    is_save_wandb            = kwargs['save_wandb']
    is_plot                  = kwargs['plot']
    # experiment_id            = kwargs['experiment']

    non_cli_params           = ctx.obj['non_cli_params']

    # data                   = non_cli_params['data']
    model                  = non_cli_params['model']
    # base_path              = non_cli_params['data']
    frame_performance_path = non_cli_params['frame_performance_path']
    frame_pred_val_path    = non_cli_params['frame_pred_val_path']
    plot_path              = non_cli_params['plot_path']
    model_type             = non_cli_params['model_type']
    project_config         = non_cli_params['project_config']
    run_parameters         = project_config['run_parameters']
    # dataset_full           = non_cli_params['dataset']
    # dataset                = dataset_full.split('/')[-1]

    data = df_by_date

    # if experiment_id == 0:
    #     data = df_by_date
    # elif experiment_id == 1:
    #     # TODO: here>
    #     data = daily_new_case_df_by_date
    # else:
    #     raise NotImplementedError
     
    # model_config = {} 
    # if model_param_epoch is not None:
    #     model_config['epoch'] = model_param_epoch
    # model_params = {}
    # if model_param_epoch is not None:
    #     model_params['epoch'] = model_config['epoch']
    # model_params_list = ['']
    # if len(model_params.keys()) > 0:
    #     for key,value in model_params.items():
    #         model_params_list.append(f'{key}={value}')
    # model_params_str = '_'.join(model_params_list)

    # multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
    # model_metadata = ['model_type', 'is_multi_step_prediction', 'n_in', 'n_out']
    # model_metadata_list = ['']
    # for i in model_metadata:
    #     if i == 'is_multi_step_prediction':
    #         v = multi_step_folder
    #     else:
    #         try:
    #             v = non_cli_params[i]
    #         except:
    #             v = kwargs[i]
    #     v = str(v) if type(v) is not str else v
    #     model_metadata_list.append(v)

    # prepared_apply_model_params = {
    #         'data': data,
    #         'model_config': model_config,
    #         'model_params': model_params,
    #         'multi_step_folder': multi_step_folder,
    #         'model_params_list': model_params_list,
    #         'model_params_str': model_params_str,
    #         'model_metadata_list': model_metadata_list,
    #         # 'model': model,
    #         }

    prepared_apply_model_params = {
            'data': data,
            # 'multi_step_folder': multi_step_folder,
            # 'model_params_list': model_params_list,
            # 'model_metadata_list': model_metadata_list,
            # 'model': model,
            }

    return ctx, kwargs, prepared_apply_model_params


def main_apply_model(selected_state, prepared_apply_model_params, ctx, **kwargs):

    # TMP
    # i = 'Oklahoma'

    ctx,kwargs,prepared_apply_model_params = prepare_apply_model(ctx, **kwargs)

    n_in                               = kwargs['n_in']
    n_out                              = kwargs['n_out']
    is_multi_step_prediction           = kwargs['is_multi_step_prediction']
    test_mode                          = kwargs['test_mode']
    # model_param_epoch                  = kwargs['model_param_epoch']
    is_save                            = kwargs['save_local']
    is_save_wandb                      = kwargs['save_wandb']
    is_plot                            = kwargs['plot']
    is_test_env                        = kwargs['is_test_environment']
    train_model_with_1_run             = kwargs['train_model_with_1_run']
    dont_create_new_model_on_each_run  = kwargs['dont_create_new_model_on_each_run']
    evaluate_on_many_test_data_per_run = kwargs['evaluate_on_many_test_data_per_run']
    # experiment_id                      = kwargs['experiment']

    is_test_env                       = 'test' if is_test_env else 'development'
    # train_model_with_1_run_str            = str(train_model_with_1_run)
    # dont_create_new_model_on_each_run_str = str(dont_create_new_model_on_each_run)

    non_cli_params           = ctx.obj['non_cli_params']

    # data                   = non_cli_params['data']
    model                  = non_cli_params['model']
    # base_path              = non_cli_params['data']
    frame_performance_path = non_cli_params['frame_performance_path']
    frame_pred_val_path    = non_cli_params['frame_pred_val_path']
    plot_path              = non_cli_params['plot_path']
    model_type             = non_cli_params['model_type']
    # dataset_full           = non_cli_params['dataset']
    project_config         = non_cli_params['project_config']
    # dataset                = dataset_full.split('/')[-1]
    run_parameters         = project_config['run_parameters']
    experiment_id          = run_parameters['experiment']['id']

    experiment_config      = project_config['experiment']

    # NOTE: for backward compatibility when config.yaml was not yet created. 
    if list(experiment_config.keys())[experiment_id] == 0:
        # assert experiment_config[experiment]['dataset_name'] == dataset_full
        assert experiment_config[experiment_id]['dataset_name'] == 'COVID19Cases/StateLevels/us-states'

    data                = prepared_apply_model_params['data']
    # model_config        = prepared_apply_model_params['model_config']
    # model_params        = prepared_apply_model_params['model_params']
    # multi_step_folder   = prepared_apply_model_params['multi_step_folder']
    # model_params_list   = prepared_apply_model_params['model_params_list']
    # model_params_str    = prepared_apply_model_params['model_params_str']
    # model_metadata_list = prepared_apply_model_params['model_metadata_list']

    model_config = {} 

    # if model_param_epoch is not None:
    #     model_config['epoch'] = model_param_epoch
    # model_params = {}
    # if model_param_epoch is not None:
    #     model_params['epoch'] = model_config['epoch']
    # model_params_list = ['']
    # if len(model_params.keys()) > 0:
    #     for key,value in model_params.items():
    #         model_params_list.append(f'{key}={value}')
    # model_params_str = '_'.join(model_params_list)

    multi_step_folder = 'MultiStep' if is_multi_step_prediction else 'OneStep'
    model_metadata = ['model_type', 'is_multi_step_prediction', 'n_in', 'n_out']
    model_metadata_list = ['']
    for i in model_metadata:
        if i == 'is_multi_step_prediction':
            v = multi_step_folder
        else:
            try:
                v = non_cli_params[i]
            except:
                v = kwargs[i]
        v = str(v) if type(v) is not str else v
        model_metadata_list.append(v)

    model, model_name = model
    # model_metadata_list.append(i)
    model_metadata_str = '_'.join(model_metadata_list)

    min_steps = float('inf')
    max_steps = float('-inf')
    for state in all_states:
        tmp = df_by_date
        case_by_date_per_states = tmp[tmp["state"] == state]
        case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)
        case_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")
        case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
        if case_by_date_per_states_np.shape[0] < min_steps:
            min_steps = case_by_date_per_states_np.shape[0] 
        if case_by_date_per_states_np.shape[0] > max_steps:
            max_steps = case_by_date_per_states_np.shape[0] 

    # print(max_steps)
    # print(min_steps)
    # print(case_by_date_per_states_np.shape)
    # print('--')
    # # exit()

    # TMP: 
    # dataset_full = 'us_state_rate_of_change_melted'

    wandb_config = {

            'multi_step_folder':                  multi_step_folder,
            'PredictNextN':                       f'PredictNext{n_out}',
            'WindowLengthN':                      f'WindowLength{n_in}',
            'state':                              state,
            'model_name':                         model_name,
            # 'model_name': 'covid19case-raw data',
            'model_type':                         model_type,
            # 'dataset': dataset_full,
            'dataset':                            experiment_config[experiment_id]['dataset_name'],
            'experiment_name':                    experiment_config[experiment_id]['name'],
            'experiment_id':                      str(experiment_id),
            'env':                                is_test_env,
            'train_model_with_1_run':             train_model_with_1_run,
            'dont_create_new_model_on_each_run':  dont_create_new_model_on_each_run,
            'evaluate_on_many_test_data_per_run': evaluate_on_many_test_data_per_run,
            # ~/parameters
            'epoch': run_parameters['epoch']
            }

    # print(wandb_config['dataset'])
    # print('ho')
    # exit()

    wandb_tags = []

    for k,v  in wandb_config.items():
        if isinstance(v, bool):
            # add tags for config keys with boolean value
            wandb_tags.append(k)
        # elif isinstance(v, (str, int, float)):
        elif isinstance(v, str):
            # add tags for config keys with str/float/int value.
            wandb_tags.append(v)
        elif isinstance(v, (int, float)):
            wandb_tags.append(str(v))
        else:
            raise ValueError()

    if len(model_config) > 0:
        wandb_config.update(model_config)

    # wandb_config = dict(
    #     dropout=0.2,
    #     hidden_layer_size=128,
    #     layer_1_size=16,
    #     layer_2_size=32,
    #     learn_rate=0.01,
    #     decay=1e-6,
    #     momentum=0.9,
    #     epochs=27,
    # )

    config_kwargs = {
        # 'project': PROJECT_NAME,
        'project': project_config['project_name'],
        'save_code':True,
        # job_type='train',
        'tags':[
            'Outputs',
            'Models',
            'Performances',
             'Baselines',
            *wandb_tags
            # wandb_config['multi_step_folder'],
            # wandb_config[f'PredictNextN'],
            # wandb_config['WindowLengthN'],
            # wandb_config['state'],
            # wandb_config['model_name'],
            # wandb_config['model_type'],
            # wandb_config['dataset'],
            # wandb_config['env'],
            # str(wandb_config['train_model_with_1_run']),
            # str(wandb_config['dont_create_new_model_on_each_run']),
            # str(wandb_config['evaluate_on_many_test_data_per_run']),
            ],
        # 'name': model_name + model_metadata_str + model_params_str,
        # 'name': model_name + model_metadata_str,
        'name': model_name,
        'config': wandb_config,
        # hyper_parameters 
        # 'epochs':100,
        }
    with wandb.init(**config_kwargs) as run:
        try:

            # print(wandb.run.config)
            # print(wandb.config)
            # exit()

            # cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, is_multi_step_prediction,
            #          model_metadata_str, model_params_str, wandb.config )

            # m = model(data,i, n_in, n_out, max_steps, min_steps, is_multi_step_prediction,
            #          model_metadata_str, model_params_str, wandb.config, config_kwargs=config_kwargs)

            # print(data)
            # exit()

            # m = model(data,i, n_in, n_out, max_steps, min_steps, is_multi_step_prediction,
            #          model_metadata_str, model_params_str, wandb.config)

            m = model(data,state, n_in, n_out, max_steps, min_steps, is_multi_step_prediction, wandb.config)

            cur_val, pred_val, eval_metric_df, model_hist, trainy_hat_list = m.forecast()

        except TypeError:
            raise NotImplementedError('refactor the rest of the *_model$ function to from function to class')
            # assert len(model_params.keys()) == 0, f'{model_name} doesn"t accept any model_params.'
            cur_val, pred_val, eval_metric_df = model(data,i, n_in, n_out, train_model_with_1_run)
                    # config_kwargs) 
        except Exception as e:
            raise ValueError(e)

        log_performance_with_wandb(eval_metric_df, is_save_wandb,**config_kwargs)

#         specified_path = None if frame_performance_path is None else BASEPATH +
#         frame_performance_path.format(multi_step_folder,n_out, n_in, state,state, model_name)
#         specified_path = add_file_suffix(specified_path, model_metadata_str + model_params_str)
#         parent_dir = '/'.join(specified_path.split('/')[:-1])
#         print(parent_dir)
#         print(specified_path)
#         Path(parent_dir).mkdir(parents=True,exist_ok=True)

#         beta_frame_performance(
#             eval_metric_df,
#             save_path=specified_path,
#             is_save=is_save
#         )
        
#         specified_path = None if frame_pred_val_path is None else BASEPATH +
#         frame_pred_val_path.format(multi_step_folder, n_out, n_in, state,state, model_name)
#         specified_path = add_file_suffix(specified_path, model_metadata_str + model_params_str)
#         parent_dir = '/'.join(specified_path.split('/')[:-1])
#         print(parent_dir)
#         print(specified_path)
#         Path(parent_dir).mkdir(parents=True,exist_ok=True)
        
#         beta_frame_pred_val(
#             cur_val.reshape(-1),
#             array(pred_val).reshape(-1),
#             save_path=specified_path,
#             is_save=is_save
#         )
    
#         y_test = cur_val.reshape(-1)
#         y_pred = array(pred_val).reshape(-1)
#         pred_val_df = DataFrame(
#             np.array([y_test, y_pred]).T,
#             columns=["y_test", "y_pred"],
#         )
#         cur_val, pred_val = pred_val_df['y_test'].tolist(), pred_val_df['y_pred'].tolist()

#         specified_path = None if plot_path is None else BASEPATH +
#         plot_path.format(multi_step_folder,n_out,n_in, state,state, model_name)
#         specified_path = add_file_suffix(specified_path, model_metadata_str + model_params_str)
#         parent_dir = '/'.join(specified_path.split('/')[:-1])
#         print(parent_dir)
#         print(specified_path)
#         Path(parent_dir).mkdir(parents=True,exist_ok=True)

#         beta_plot(
#             cur_val,
#             pred_val,
#             save_path=specified_path,
#             display=is_plot,
#             is_save=is_save
#         )

#         if test_mode:
#             exit()

    output = {
            'pred_val': pred_val,
            'eval_metric_df': eval_metric_df,
            'model_hist': model_hist,
            'trainy_hat_list': trainy_hat_list
            }
    return output

@ray.remote
def apply_model(state, ctx, **kwargs):
    ctx,kwargs,prepared_apply_model_params = prepare_apply_model(ctx, **kwargs)
    main_apply_model(state, prepared_apply_model_params,ctx,**kwargs)

@click.command()
@click.argument('n_in', type=int)
@click.argument('n_out', type=int)
# @click.argument('states', type=str, nargs=1)
# @click.option('--states', type=str, nargs=1) # argument have to be separated by ','
# @click.option('--data', type=str, default= 'COVID19Cases/StateLevels/us-states')
@click.option('--test_mode', is_flag=True)
@click.option('--is_multi_step_prediction', is_flag=True)
@click.option('--save_local', is_flag=True)
@click.option('--save_wandb', is_flag=True)
@click.option('--plot', type=int)
# @click.option('--model_param_epoch', type=int)
@click.option('--is_distributed', is_flag=True)
@click.option('--num_gpus', type=int)
@click.option('--num_cpus', type=int)
@click.option('--is_test_environment', is_flag=True)
@click.option('--train_model_with_1_run', is_flag=True)
@click.option('--dont_create_new_model_on_each_run', is_flag=True)
@click.option('--evaluate_on_many_test_data_per_run', is_flag=True)
@click.option('--epoch', type=int)
# @click.option('--repeat', type=int, default=1)
# @click.option('--experiment', type=int) # argument have to be separated by ','
@click.pass_context
def delta_apply_model_to_all_states(ctx, **kwargs):

    stream = open("config.yaml", 'r')
    project_config = yaml.load(stream)

    # for key, value in project_config.items():
    #     pprint (key + " : " + str(value))
    # pprint(project_config )
    # exit()

    model_name = ctx.obj['non_cli_params']['model'][1] 
    if model_name in ALL_BASELINES_MODELS or model_name == 'previous_val':
        model_type = 'baseline'
    else:
        raise NotImplementedError()

    if kwargs['save_wandb']:
        os.environ['WANDB_MODE'] = 'run'
    else:
        os.environ['WANDB_MODE'] = 'dryrun'
    
    
    ctx.obj['non_cli_params']['model_type'] = model_type

    # ctx.obj['non_cli_params']['dataset'] = 'COVID19Cases/StateLevels/us-states'
    # ctx.obj['non_cli_params']['dataset'] = kwargs['data']
    # ctx.obj['non_cli_params']['dataset'] = kwargs['data']

    ctx.obj['non_cli_params']['project_config'] = project_config

    # print(ctx.obj['non_cli_params']['dataset'] )
    # print('s')
    # exit()

    # print(DATASETS_DICT['COVID19Cases/StateLevels/us-states'])
    # print(kwargs['states'])
    # print('done')
    # exit()
    # pprint(kwargs)
    # exit()

    # TMP:
    # all_states = ['Florida', 'Louisiana', 'Tennessee', 'Oklahoma']
    # all_states = ['Florida']
    # all_states = ['Louisiana', 'Tennessee', 'Oklahoma']
    # all_states = [ 'Oklahoma']

    ctx.obj['non_cli_params']['all_states'] = all_states

    if kwargs['is_distributed']:
        raise NotImplementedError('implement what I did for else condition.')
        ray.init(num_gpus=kwargs['num_gpus'], num_cpus=kwargs['num_cpus'])
        futures = [apply_model.remote(i, ctx, **kwargs) for i in all_states]
        ray.get(futures)
    else:
        gamma_apply_model_to_all_states_no_click(ctx, **kwargs)
        

def gamma_apply_model_to_all_states_no_click(ctx, **kwargs):
    ctx,kwargs,prepared_apply_model_params = prepare_apply_model(ctx, **kwargs)

    non_cli_params = ctx.obj['non_cli_params']

    all_states     = non_cli_params['all_states']
    project_config = non_cli_params['project_config']
    run_parameters = project_config['run_parameters']
    repeat         = run_parameters['repeat']
    selected_states         = run_parameters['states']

    # repeat = kwargs['repeat']
    # selected_states = kwargs['states'].split(',')

    for i in selected_states:
        assert i in all_states, i

    assert isinstance(repeat, int)
    
    # for i in all_states:
    # TMP
    # for i in ['Oklahoma', 'Florida', 'Louisiana', 'Tennessee']:
    # for i in ['Oklahoma']:
    # for i in ['Louisiana']:
    for i in selected_states:
        pred_val_list       = []
        eval_metric_df_list = []
        model_hist_list     = []
        all_trainy_hat_list = []
        for r in range(repeat):
            output = main_apply_model(i, prepared_apply_model_params,ctx,**kwargs)

            pred_val = output['pred_val']
            eval_metric_df = output['eval_metric_df']
            model_hist = output['model_hist']
            trainy_hat_list = output['trainy_hat_list']

            pred_val_list.append(pred_val)
            eval_metric_df_list.append(eval_metric_df)
            if model_hist is not None:
                model_hist_list.append(model_hist)
            all_trainy_hat_list.append(trainy_hat_list)

            # print(len(pred_val_list[0]))
            # print(eval_metric_df_list[0].shape)
            # print(model_hist_list[0].history['loss'])
            # print(model_hist_list[0].history['val_loss'])
            # print(all_trainy_hat_list[0])
            # if r > 1:
            #     break

        # print(r)
        # print('here')
        # exit()
        # for i in range(r):
            # print(len(pred_val_list[i]))
            # print(eval_metric_df_list[i].shape)
            # print(len(model_hist_list[i].history['loss']))
            # print(len(model_hist_list[i].history['val_loss']))
            # print(all_trainy_hat_list[i])

        # # TMP
        # if i == 'Oklahoma':
        #     break

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

