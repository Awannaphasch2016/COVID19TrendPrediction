from pandas import DataFrame, concat
import numpy as np

# ==helper
def split(ts):
    size = int(len(ts) * 0.85)
    train = ts[:size]
    test = ts[size:]
    return (train, test)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Do the same job as keras.preprocessing.sequence.TimesereiesGenerator but (# instances,n_in + n_out)"""
    # NOTE: this function doesn't seems so stable, but lets make minimum changes to avoid possible bugs.

    # # NOTE: keep line below in case this change generate bugs in the future.
    # n_vars = 1 if type(data) is list else data.shape[1]
    
    n_vars = 1 if type(data) is list else data.shape[1]

    if n_vars != 1:
        raise NotImplementedError('only accept univariance time series currenlty.')

    df = DataFrame(data)  # (353,1)
    cols = list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))

    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values

    # print(agg.shape)  # (353,31)

    if dropnan:
        agg.dropna(inplace=True)

    # print(agg.shape)  # (317,31)

    return agg.values
