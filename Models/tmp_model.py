
from TMP.scratch1 import model_class

class tmp_model(model_class):
    pass

if __name__ == "__main__":

    tf.compat.v1.enable_eager_execution()
    # print(tf.executing_eagerly())
    # exit()
    non_cli_params = {
        'data': df_by_date,
        'model' : (model_class, 'model_class'),
        'base_path' : BASEPATH,
        'frame_performance_path' : FRAME_PERFORMANCE_PATH,
        'frame_pred_val_path' : FRAME_PRED_VAL_PATH,
        'plot_path' : PLOT_PATH,
    }
    
    # model_config_params = {
    #         'Dense_1': {
    #             'args': [100], 
    #             'kwargs': {
    #                 'activation':"relu",
    #                 }
    #             },
    #         'Dense_2': {
    #             'args': [],
    #             'kwargs': {}
    #             }
    #         ]
    # }

    # gamma_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params})
    # delta_apply_model_to_all_states(obj={'non_cli_params': non_cli_params, 'model_config_params': model_config_params})
