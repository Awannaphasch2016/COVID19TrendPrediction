from matplotlib import pyplot


def plot(y, yhat, save_path=None, display=True):
    pyplot.plot(y, label="Expected")
    pyplot.plot(yhat, label="Predicted")
    pyplot.legend()
    if save_path is not None:
        # pyplot.savefig(BASEPATH / pathlib.Path("Outputs/Images/Xgboost/forecasting.jpg"))
        pyplot.savefig(save_path)
        print(f"save plot to {save_path}")
    if display:
        pyplot.show()
