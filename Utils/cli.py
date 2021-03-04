import click
import os

# @cli.command()
@click.command()
@click.argument('n_in')
@click.argument('n_out')
@click.option('--test_mode', is_flag=True)
@click.option('--is_multi_step_prediction', is_flag=True)
@click.pass_context
# def run_func(apply_model_to_all_states, **kwargs):
def run_func(ctx, **kwargs):
# def run_func():
    non_cli_params = ctx.obj['non_cli_params']
    kwargs.update(non_cli_params)
    print(kwargs)
    # pprint(list(kwargs.keys()))
    fake_apply_model_to_all_states(**kwargs)


def fake_apply_model_to_all_states(
    data, model,n_in, n_out, is_multi_step_prediction, base_path, frame_performance_path=None, frame_pred_val_path=None, 
    plot_path=None, test_mode=False
):
    print('complete')
    exit()

if __name__ == '__main__':

    non_cli_params = {
        'data': None,
        'model' : (None, 'None'),
        'base_path' : None,
        'frame_performance_path' : None,
        'frame_pred_val_path' : None,
        'plot_path' : None,
    }

    run_func(obj={'non_cli_params': non_cli_params})
