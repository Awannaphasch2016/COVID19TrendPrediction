import os
import pathlib

# BASEPATH = pathlib.Path(os.getcwd())
BASEPATH = '/home/awannaphasch2016/Documents/Working/CovidTrendPrediction'
BASEPATH = '/home/awannaphasch2016/Documents/Working/COVID19TrendPrediction'
BASEPATH = os.path.dirname(os.path.realpath(__file__))

# print(BASEPATH)
FRAME_PERFORMANCE_PATH = "/Outputs/Models/Performances/Baselines/{}/{}_{}_performance.csv"
FRAME_PRED_VAL_PATH = "/Outputs/Models/Performances/Baselines/{}/{}_{}_pred_val.csv"
PLOT_PATH =  "/Outputs/Models/Performances/Baselines/{}/Images/{}_{}_forcasting.jpg"
CHECKPOINTS_PATH = "/Outputs/Models/Checkpoints"
