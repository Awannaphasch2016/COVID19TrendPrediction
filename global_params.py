import os
import pathlib
from pathlib import Path

PROJECT_NAME = 'COVID19TrendPrediction'

ALL_METRICS = ['mse', 'rmse', 'r2score', 'mape']
ALL_BASELINES_MODELS = ['mlp', 'linear regression', 'xgboost','previous_day', 'lstm']

HOME_PATH = str(Path.home())
PROJECT_PATH = 'Documents/Working/COVID19TrendPrediction'

# BASEPATH = pathlib.Path(os.getcwd())
# BASEPATH = '/home/awannaphasch2016/Documents/Working/CovidTrendPrediction'
# BASEPATH = '/home/awannaphasch2016/Documents/Working/COVID19TrendPrediction'
BASEPATH = os.path.dirname(os.path.realpath(__file__))

# print(BASEPATH)
FRAME_PERFORMANCE_PATH = "/Outputs/Models/Performances/Baselines/{}/PredictNext{}/WindowLength{}/{}/{}_{}_performance.csv"
FRAME_PRED_VAL_PATH = "/Outputs/Models/Performances/Baselines/{}/PredictNext{}/WindowLength{}/{}/{}_{}_pred_val.csv"
PLOT_PATH =  "/Outputs/Models/Performances/Baselines/{}/PredictNext{}/WindowLength{}/{}/Images/{}_{}_forcasting.jpg"
CHECKPOINTS_PATH = "/Outputs/Models/Checkpoints"
SCRATCHES_OUTPUT_PATH = '/Outputs/Scratches'


ALL_PREDICTNEXTN = [1,7,14,30]
ALL_WINDOWLENGTHN = [1,7,14,30]
