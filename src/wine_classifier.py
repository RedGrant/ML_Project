# Idilson
# Pedro Guedes
# Rafael Cabral
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# Libraries' import

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataset_loader import dataset_loader

dataset = dataset_loader()

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
X_Test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
