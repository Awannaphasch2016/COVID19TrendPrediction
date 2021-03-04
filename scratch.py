from Utils.cli import *

non_cli_params = {
    'data': None,
    'model' : (None, 'None'),
    'base_path' : None,
    'frame_performance_path' : None,
    'frame_pred_val_path' : None,
    'plot_path' : None,
}

run_func(obj={'non_cli_params': non_cli_params})
