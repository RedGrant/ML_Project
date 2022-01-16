# Idilson
# Pedro Guedes
# Rafael Cabral
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# Libraries' import

import numpy as np
import pandas as pd
import matplotlib as plt
import plotly.express as px
import sys
import os


def dataset_loader():
    path = 'src/data/winequality-red.csv'

    # load the entire dataset and labels for each attribute through pandas csv reader method
    raw_dataset = pd.read_csv(path)

    raw_dataset = check_for_missing_values(raw_dataset)

    # load a figure to the local host ip with the second argument being the attribute/output to be read
    fig = px.histogram(raw_dataset, x='quality')
    fig.show()

    # maximum wine quality found in the dataset
    print("The minimum wine quality found in the dataset:", (min(raw_dataset.quality)))
    # maximum wine quality found in the dataset
    print("The maximum wine quality found in the dataset:", (max(raw_dataset.quality)))


def check_for_missing_values(raw_dataset):
    # check if the dataset is broken: missing values
    missing_values = raw_dataset.isnull().sum()
    values_to_remove = []
    if missing_values.sum() != 0:
        print("The dataset has the following corrupt data:")
        for line_index in range(raw_dataset.T.shape[0]):
            for col_index in range(raw_dataset.T.shape[1]):
                if np.isnan(raw_dataset.T[col_index][line_index]):
                    values_to_remove.append(col_index)

        values_to_remove = sorted(set(values_to_remove))
        print("Line index: ", line_index, "and collumn index:", col_index)
        print("Do Y - remove line. Any other input - keep it.")
        decision = str(input())
        if decision == 'Y':
            for removal_index in reversed(range(values_to_remove)):
                raw_dataset = raw_dataset.drop(labels=values_to_remove[removal_index], axis=0)
                raw_dataset = raw_dataset.reset_index(drop=True)

        else:
            print("Fix the dataset manually so that it can be loaded")
            exit()

    else:
        print("The dataset does not contain missing values in any attribute")

    print(raw_dataset.shape)



dataset_loader()
