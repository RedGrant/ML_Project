# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

from wine_logistic_regression import wine_log_regression
from wine_svc import wine_svc
from wine_rfc import wine_rfc
from sklearn.preprocessing import StandardScaler

import pathlib
import os
import pickle


def train_models(dataset, class_type):
    """
    Trains models with different approaches, returning the most optimal of each one.
    :param dataset: the split datasets (train, validation and test sets) for each class type
    :param class_type: different class type (different set of outputs)
    :return: the optimized models are returned
    """
    # decompose each set
    X_train, Y_train = dataset[0]
    X_val, Y_val = dataset[1]
    X_test, Y_test = dataset[2]

    # The input variables are those that the network takes on the input or visible layer in order to make a prediction.
    # The scale and distribution of the data drawn from the domain may be different for each variable.
    # Input variables may have different units (e.g. feet, kilometers, and hours) that, in turn,
    # may mean the variables have different scales.
    # Differences in the scales across input variables may increase the difficulty of the problem being modeled.
    scaler = StandardScaler().fit(X_train)
    # Perform standardization by centering and scaling.
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    # TODO - Perform the scaler in the testing algorithm
    X_Test = scaler.transform(X_test)

    # Transform series to numpy
    Y_train = Y_train.to_numpy()
    Y_val = Y_val.to_numpy()

    # check if the current model was already trained
    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/models/' + class_type + '_log_reg_model_py3_8.sav'

    # modeling the Logistic Regression Model. If one is already trained and optimized, it is going to be loaded instead
    if not os.path.exists(path):
        log_reg_model = wine_log_regression(X_train, Y_train, X_val, Y_val, class_type)
    else:
        log_reg_model = pickle.load(open(path, 'rb'))

    path = str(actual_dir) + '/models/'+class_type+'_svc_model_py3_8.sav'
    # modeling the SVC Model. If one is already trained and optimized, it is going to be loaded instead
    if not os.path.exists(path):
        svc_model = wine_svc(X_train, Y_train, X_val, Y_val, class_type)
    else:
        svc_model = pickle.load(open(path, 'rb'))

    path = str(actual_dir) + '/models/'+class_type+'_rfc_model_py3_8.sav'
    # modeling the Random Forest Classifier Model. If one is already trained and optimized,
    # it is going to be loaded instead
    if not os.path.exists(path):
        rfc_model = wine_rfc(X_train, Y_train, X_val, Y_val, class_type)
    else:
        rfc_model = pickle.load(open(path, 'rb'))

    optimized_models = [log_reg_model, svc_model, rfc_model]
    return optimized_models
