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


# TODO - Generate classification reports and print and save them as txt