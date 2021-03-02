from matplotlib import pyplot
from Utils.aws_services import *


def beta_plot(y, yhat, save_path=None, display=True):
    pyplot.plot(y, label="Expected")
    pyplot.plot(yhat, label="Predicted")
    pyplot.legend()
    if save_path is not None:
        # pyplot.savefig(BASEPATH / pathlib.Path("Outputs/Images/Xgboost/forecasting.jpg"))
        pyplot.savefig(save_path)
        save_to_s3(save_path)
        
        print(f"save plot to {save_path}")
        pyplot.clf()
    if display:
        pyplot.show()

def plot(y, yhat, save_path=None, display=True):
    pyplot.plot(y, label="Expected")
    pyplot.plot(yhat, label="Predicted")
    pyplot.legend()
    if save_path is not None:
        # pyplot.savefig(BASEPATH / pathlib.Path("Outputs/Images/Xgboost/forecasting.jpg"))
        pyplot.savefig(save_path)
        print(f"save plot to {save_path}")
        pyplot.clf()
    if display:
        pyplot.show()
