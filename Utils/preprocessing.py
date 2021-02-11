from pandas import DataFrame, concat

# ==helper
def split(ts):
    size = int(len(ts) * 0.85)
    train = ts[:size]
    test = ts[size:]
    return (train, test)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Do the same job as keras.preprocessing.sequence.TimesereiesGenerator but (# instances,n_in + n_out)"""
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
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
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values
