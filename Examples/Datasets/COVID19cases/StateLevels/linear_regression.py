import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from global_params import *
from Models.Preprocessing.us_state import *
from Utils.eval_funcs import *
from Utils.preprocessing import *
from Utils.utils import *

X_train, X_test = split(case_by_date_florida_np)

y_train = X_train[:-1]
X_train = X_train[1:]

y_test = X_test[:-1]
X_test = X_test[1:]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse(y_test, y_pred)

rmse(y_test, y_pred)

mape(y_test, y_pred)

r2score(y_test, y_pred)
