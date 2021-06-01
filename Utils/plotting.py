from Utils.aws_services import *

def beta_plot(y, yhat, save_path=None, display=False, is_save=False):
    if display:
        from matplotlib import pyplot
        pyplot.plot(y, label="Expected")
        pyplot.plot(yhat, label="Predicted")
        pyplot.legend()
        pyplot.show()
    else:
        print('plot is not displayed.')

    if is_save:
        assert save_path is not None
        # pyplot.savefig(BASEPATH / pathlib.Path("Outputs/Images/Xgboost/forecasting.jpg"))
        pyplot.savefig(save_path)
        print(f"save plot to local at {save_path}")
        save_to_s3(save_path)
        print(f"save plot to S3 at {save_path}")
        pyplot.clf()
    else:
        print("plot is not saved.")

def plot(y, yhat, save_path=None, display=True):
    raise NotImplementedError('Deprecated.')
    from matplotlib import pyplot
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
