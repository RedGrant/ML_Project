# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# Libraries' import
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

# created functions
from dataset_loader import dataset_loader
from train_models import train_models

class_type = ['binary', 'multiclass_3', 'multiclass_5']
# load the dataset without any missing data and divided in training, validation and test sets
dataset_binary, dataset_multiclass_3, dataset_multiclass_5 = dataset_loader(class_type)

# each class type will save the most optimal model of each methodology (logistic regression, svc, rfc, etc)
for class_to_train in class_type:
    if class_to_train == 'binary':
        binary_models = train_models(dataset_binary, class_to_train)
    elif class_to_train == 'multiclass_3':
        multiclass3_models = train_models(dataset_multiclass_3, class_to_train)
    elif class_to_train == 'multiclass_5':
        multiclass5_models = train_models(dataset_multiclass_5, class_to_train)


# # decompose each set
# X_train, Y_train = dataset[0]
# X_test, Y_test = dataset[1]
# X_val, Y_val = dataset[2]
#
# # The input variables are those that the network takes on the input or visible layer in order to make a prediction.
# # The scale and distribution of the data drawn from the domain may be different for each variable.
# # Input variables may have different units (e.g. feet, kilometers, and hours) that, in turn,
# # may mean the variables have different scales.
# # Differences in the scales across input variables may increase the difficulty of the problem being modeled.
# scaler = StandardScaler().fit(X_train)
#
# # Perform standardization by centering and scaling.
# X_train = scaler.transform(X_train)
# X_val = scaler.transform(X_val)
# X_Test = scaler.transform(X_test)
#
# actual_dir = pathlib.Path().absolute()
# path = str(actual_dir) + '/models/log_reg_model_py3_8.sav'
#
# # modeling the Logistic Regression Model. If one is already trained and optimized, it is going to be loaded instead
# if not os.path.exists(path):
#     log_reg_model = wine_log_regression(X_train, Y_train, X_val, Y_val)
# else:
#     log_reg_model = pickle.load(open(path, 'rb'))
#
# path = str(actual_dir) + '/models/svc_model_py3_8.sav'
#
# # modeling the SVC Model. If one is already trained and optimized, it is going to be loaded instead
# if not os.path.exists(path):
#     svc_model = wine_svc(X_train, Y_train, X_val, Y_val)
# else:
#     svc_model = pickle.load(open(path, 'rb'))
#
# path = str(actual_dir) + '/models/rfc_model_py3_8.sav'
#
# # modeling the Random Forest Classifier Model. If one is already trained and optimized, it is going to be loaded instead
# if not os.path.exists(path):
#     rfc_model = wine_rfc(X_train, Y_train, X_val, Y_val)
# else:
#     rfc_model = pickle.load(open(path, 'rb'))

# TODO - Generate classification reports and print and save them as txt