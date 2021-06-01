import wandb
from global_params import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *
from Utils.plotting import *
from Utils.modelling import *


min_val = float('inf')
max_val = float('-inf')
# for i in range(df_by_date.to_numpy().shape[0]):
for state in all_states:
    tmp = df_by_date
    case_by_date_per_states = tmp[tmp["state"] == state]
    case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)
    case_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))
    if case_by_date_per_states_np.shape[0] < min_val:
        min_val = case_by_date_per_states_np.shape[0] 
    if case_by_date_per_states_np.shape[0] > max_val:
        max_val = case_by_date_per_states_np.shape[0] 

# all_states = ['Oklahoma']
# all_states = ['Florida', 'Louisiana', 'Tennessee']
all_states = ['Oklahoma']

# here> get pred value on to the tracking 
for state in all_states:

    tmp = df_by_date

    case_by_date_per_states = tmp[tmp["state"] == state]
    case_by_date_per_states = case_by_date_per_states.drop(['date', 'state'], axis=1)
    case_by_date_per_states_np = case_by_date_per_states.to_numpy().astype("float")
    case_by_date_per_states_np = np.reshape(case_by_date_per_states_np, (-1, 1))


    stream = open("config.yaml", 'r')
    project_config = yaml.load(stream)

    experiment_id = 0
    # experiment_id = 1
    # experiment_id = 2
    # experiment_id = 3
    # experiment_id = 4
    dataset_name = project_config['experiment'][experiment_id]['dataset_name']
    experiment_name = project_config['experiment'][experiment_id]['name']

    if experiment_id == 0:
        name = 'covid19case-raw data'
    elif experiment_id == 1:

        name = 'daily-new-covid19case-raw data'
        # print(case_by_date_per_states_np[-10:])
        case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
        case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        # print(case_by_date_per_states_np[-10:])
        # print(case_by_date_per_states_np)
        # exit()

    elif experiment_id == 2:

        name = 'diff2-daily-new-covid19case-raw data'
        # print(case_by_date_per_states_np[-10:])
        case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
        case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
        case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        # print(case_by_date_per_states_np[-10:])
        # print(case_by_date_per_states_np)
        # exit()
    elif experiment_id == 3:

        name = 'diff3-daily-new-covid19case-raw data'
        # print(case_by_date_per_states_np[-10:])
        case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
        case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
        case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        case_by_date_per_states_np = np.diff(case_by_date_per_states_np, n=1, axis=0)
        case_by_date_per_states_np = np.vstack(([[0]], case_by_date_per_states_np))
        # print(case_by_date_per_states_np[-10:])
        # print(case_by_date_per_states_np)
        # exit()
    elif experiment_id == 4:

        name = 'straight line data'
        tmp = np.arange(1000).reshape(-1,1)
        case_by_date_per_states = tmp 
        case_by_date_per_states_np = tmp 
    else:
        raise NotImplementedError

    # print(name)
    # print(dataset_name)
    # exit()

    # case_by_date_per_states_np = case_by_date_per_states_np[:-min_val, :]
    # print(case_by_date_per_states_np)

### dataset -> tell which state will be used for prediction
###data 
    # data = series_to_supervised(case_by_date_per_states_np, n_in=n_steps_in, n_out=n_steps_out)

    data  = case_by_date_per_states_np

    # print(data.shape)
    # exit()

    # split = 0.15
    # n_test = round(case_by_date_per_states.shape[0] * split)
    # train, test = train_test_split(data, n_test)
    # print(train.shape)
    # print(test.shape)
    # exit()


    wandb_config = {
            'multi_step_folder': 'NA',
            'PredictNextN': 'NA',
            'WindowLengthN': 'NA',
            'state': state,
            'model_name': name,
            # 'model_name': 'covid19case-raw data',
            'model_type': name,
            # 'dataset': 'COVID19Cases/StateLevels/us-states',
            'dataset': dataset_name,
            'env': 'NA',
            'train_model_with_1_run': 'NA',
            'dont_create_new_model_on_each_run': 'NA',
            'evaluate_on_many_test_data_per_run': 'NA',
            'experiment_id': experiment_id,
            'experiment_name': experiment_name
            }

    wandb_tags = []

    config_kwargs = {
        'project': 'COVID19TrendPrediction',
        'name': name,
        'tags': { state, dataset_name},
        # 'config': {"model_name": name}
        'config': wandb_config
        }

    wandb.init(**config_kwargs) 

    start = max_val - data.shape[0]
    for i in range(data.shape[0]):
        # trainy_hat = self.forecast_single_step(trainX[i], \
        #         trainy[i])['yhat'].reshape(-1).tolist()

        # wandb.log({'predict vs real':data[i][0]}, step=i+start)
        # wandb.log({'test something' :data[i][0]}, step=i+start)
        print(i+start)
        wandb.log({'predict vs real':data[i][0], 'custom step':i+start})
# tmp = list(range(100))
# for i in tmp:
#     wandb.log({'predict vs real':trainy_hat[0]}, step=i)
    wandb.finish()
    # exit()
