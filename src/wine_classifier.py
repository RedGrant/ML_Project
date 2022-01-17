# Idilson
# Pedro Guedes
# Rafael Cabral
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# Libraries' import

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from dataset_loader import dataset_loader
from wine_logistic_regression import wine_log_regression
import pathlib
import os
import pickle

# load the dataset without any missing data and divided in training, validation and test sets
dataset = dataset_loader()

# decompose each set
X_train, Y_train = dataset[0]
X_test, Y_test = dataset[1]
X_val, Y_val = dataset[2]

# The input variables are those that the network takes on the input or visible layer in order to make a prediction.
# The scale and distribution of the data drawn from the domain may be different for each variable.
# Input variables may have different units (e.g. feet, kilometers, and hours) that, in turn, 
# may mean the variables have different scales.
# Differences in the scales across input variables may increase the difficulty of the problem being modeled.
scaler = StandardScaler().fit(X_train)

# Perform standardization by centering and scaling.
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_Test = scaler.transform(X_test)

actual_dir = pathlib.Path().absolute()
path = str(actual_dir) + '/models/log_reg_model_py3_8.sav'

# modeling the Logistic Regression Model. If one is already trained and optimized, it is going to be loaded instead
if not os.path.exists(path):
    log_reg_model = wine_log_regression(X_train, Y_train, X_val, Y_val)
else:
    log_reg_model = pickle.load(open(path, 'rb'))

