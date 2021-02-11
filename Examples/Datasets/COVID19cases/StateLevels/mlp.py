import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

from global_params import *
from Utils.preprocessing import *
from Utils.utils import *
from Models.Preprocessing.us_state import *

# ==params
n_input = 5
n_features = 1

model = Sequential()
model.add(LSTM(150, activation="relu", input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")
# print(model.summary())
# exit()

# checkpoint_path = "training_1/cp.ckpt"
checkpoint_path = pathlib.Path("Outputs/ModelsCheckPoints/LSTM/nyd-FL.ckpt")
checkpoint_dir = os.path.dirname(BASEPATH / checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

# model.fit_generator(generator_train, epochs=3, callbacks=[cp_callback])
model.fit_generator(generator_train, epochs=3)
loss_per_epoch = model.history.history["loss"]
fig = plt.figure(dpi=120, figsize=(8, 4))
ax = plt.axes()
ax.set(xlabel="Number of Epochs", ylabel="MSE Loss", title="Loss Curve of RNN LSTM")
plt.plot(range(len(loss_per_epoch)), loss_per_epoch, lw=2)
# plt.savefig(BASEPATH / pathlib.Path("Outputs/Images/LSTM/loss_function.jpg"))
plt.show()
# exit()

model.load_weights(checkpoint_path)
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

# load pima indians dataset
# split into input (X) and output (Y) variables
# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(case_by_date_florida_test, case_by_date_florida_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
