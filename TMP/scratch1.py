from pandas import DataFrame, concat
import numpy as np
# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
# from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from pathlib import Path
import numpy as np
from numpy import array

# load a single file as a numpy array
def load_file(filepath):
    base = Path('/home/awannaphasch2016/Data/MachineLearningMastery/UCI HAR Dataset/')
    dataframe = read_csv(str(base/filepath), header=None, delim_whitespace=True)
    # print(base/filepath)
    # print(dataframe.shape)
    # exit()
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    # exit()
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    # print(trainX[:,0].head)
    # print(trainy[:10,:])
    # exit()

    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    # print(trainX.shape, trainy.shape)
    # load all test
    # print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    # exit()
    return trainX, trainy, testX, testy

# summarize scores
def summarize_results(scores, params):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = mean(scores[i]), std(scores[i])
        print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    pyplot.boxplot(scores, labels=params)
    pyplot.savefig('exp_cnn_kernel.png')

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 32

    n_filters = 128
    n_kernels = 11

    #256 n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    n_timesteps, n_outputs = trainX.shape[1], trainy.shape[1]
    batch, n_timesteps, n_features = trainX.shape[0], trainX.shape[1], 1
    # trainX = trainX.reshape( n_timestep, n_feature)
    trainX = trainX.reshape(batch, n_timesteps, n_features)
    batch, n_timesteps, n_features = testX.shape[0], testX.shape[1], 1
    # testX = testX.reshape(batch, n_timestep, n_feature)
    testX = testX.reshape(batch, n_timesteps, n_features)
    # print('hi')
    # exit()

    # print(trainX.shape[1], 1, trainy.shape[1])
    # exit()

    # print(batch, n_timesteps, n_features)
    # exit()

    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels, activation='relu', input_shape=(n_timesteps,n_features)))
    # model.add(Conv1D(filters=n_filters, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(n_outputs, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.add(Dense(n_outputs))
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # fit network

    # model = Sequential(
    #     [
    #         Dense(100, activation="relu", input_dim=n_timesteps),
    #         Dense(1),
    #     ]
    # )
    # model.compile(optimizer="adam", loss="mse")

    # trainX: (7352, 128, 9)
    # trainy: (7352, 6)
    # testX: (2947, 128, 9)
    # testy: (2947, 6)

    # (836, 14, 1)
    # (836, 1)
    # (150, 14, 1)
    # (150, 1)


    # print(trainX.shape)
    # print(trainy.shape)
    # print(testX.shape)
    # print(testy.shape)
    # exit()

    # trainX = trainX.reshape(trainX.shape[0],-1)
    # trainy = trainy.reshape(trainX.shape[0],-1)
    # testX = testX.reshape(trainX.shape[0],-1)
    # testy = testy.reshape(trainX.shape[0],-1)

    # # for i,j in zip(testX,testy):
    # for i,j in zip(trainX,trainy):
    #     tmp1 = model.predict(np.array([i]),verbose=0)
    #     # print(i)
    #     print(i, j)
    # exit()

    def mse(y1, y_pred):
        y1, y_pred = np.array(y1).reshape(-1), np.array(y_pred).reshape(-1)
        tmp = y1 - y_pred
        output = np.mean(tmp ** 2)

        # print(y1.shape)
        # print(y_pred.shape)
        # print(y1.reshape(-1)-y_pred)
        # print(output)
        # print(y1)
        # print('sssssss')
        # # exit()

        assert tmp.shape[0] == y1.shape[0]
        return output



    results = []
    # model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    model.fit(trainX, trainy, epochs=epochs, verbose=verbose)
    tmp1 = model.predict(testX,verbose=0)
    ans = mse(tmp1, testy)
    results.append(ans)

    # summarize_results(ans, )


    # evaluate model
    tmp, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)


    import matplotlib.pyplot as plt
    plt.plot(tmp1.reshape(-1), label='pred')
    # plt.plot(testy.reshape(-1), label='true' )
    plt.plot(testy.reshape(-1), label='true' )
    plt.show()
    exit()


    # tmp1 = model.predict(trainX,verbose=0)
    # print(mse(tmp1, trainy))
    # tmp1 = model.predict(np.array([testX]),verbose=0)
    # import matplotlib.pyplot as plt
    # plt.plot(tmp1.reshape(-1), label='pred')
    # # plt.plot(testy.reshape(-1), label='true' )
    # plt.plot(trainy.reshape(-1), label='true' )
    # plt.show()
    # exit()


    # for i,j in zip(testX,testy):
    for i,j in zip(trainX,trainy):
        tmp1 = model.predict(np.array([i]),verbose=0)
        # print(i)
        print(i, tmp1, j)
        print(mse(tmp1,j))
        print('sssssssss')
    exit()
    
    # print(accuracy)


    import math
    print(abs(testy - tmp1).mean())
    exit()

    return accuracy


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

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

# run an experiment
def run_experiment(repeats=1):
    # load data

    # trainX, trainy, testX, testy = load_dataset()

    tmp = np.arange(1000).reshape(-1,1)
    # tmp = np.arange(10000).reshape(-1,1)
    # trainX, testy = split_sequences(tmp, 14, 1)
    n_steps_in = 14
    n_steps_out = 1
    data = series_to_supervised(tmp, n_in=n_steps_in, n_out=n_steps_out)
    split = 0.15
    n_test = round(tmp.shape[0] * split)
    train, test = train_test_split(data, n_test)
    trainX, trainy = train[:, :n_steps_in], train[:, -1].reshape(-1,1)
    testX, testy = test[:, :n_steps_in], test[:, -1].reshape(-1,1)

    evaluate_model(trainX, trainy, testX, testy)
    exit()

    # print(trainX.shape) # (823, 14) vs (7352, 128, 9)
    # print(trainy.shape) # (823, 14) vs (7352, 128, 9)
    # print('ss')
    # print(testX.shape)  # (150, 1) vs (2947, 6)
    # print(testy.shape)  # (150, 1) vs (2947, 6)
    # exit()

    # # for i,j in zip(testX,testy):
    # for i,j in zip(trainX,trainy):
    #     # tmp1 = model.predict(np.array([i]),verbose=0)
    #     evaluate_model()
    #     # print(i)
    #     print(i, j)
    # exit()

# run the experiment
run_experiment()
