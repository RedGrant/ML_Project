# Idilson
# Pedro Guedes
# Rafael Cabral
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# Libraries' import

import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
import seaborn as sb
import plotly.express as px
import pathlib
from os import path

def dataset_loader():
    '''
    Loads the dataset. The path can be changed to consider another dataset.
    Missing values can be removed, or the option of dealing with that manually is
    available for the user. TODO - update this description
    :return: dataset - returns the fixed dataset
    '''
    actual_dir = pathlib.Path().absolute()

    # TODO - REMOVE THIS DEBUGGER LINE
    #debbuger actual_dir:
    #actual_dir = '/home/pedroguedes/PycharmProjects/ML_Project/src'
    #actual_dir = '/home/pguedes/PycharmProjects/ML_Project/src'
    path = str(actual_dir) + '/data/winequality-red.csv'

    # load the entire dataset and labels for each attribute through pandas csv reader method
    raw_dataset = pd.read_csv(path)
    raw_dataset.head()
    # check if the dataset contains missing values
    raw_dataset = check_for_missing_values(raw_dataset)

    # maximum wine quality found in the dataset
    print("The minimum wine quality found in the dataset:", (min(raw_dataset.quality)))
    # maximum wine quality found in the dataset
    print("The maximum wine quality found in the dataset:", (max(raw_dataset.quality)))

    # if the figure is not saved yet, it will be generated
    if not path.exists('pairplot_figure_wine_dataset.png'):
        # pair plotting the data
        pairplot_figure = sb.pairplot(raw_dataset)
        pairplot_figure = pairplot_figure.fig
        pairplot_figure.savefig("pairplot_figure_wine_dataset.png")





    # TODO - This might change in the meantime
    dataset = raw_dataset

    return dataset

def check_for_missing_values(raw_dataset):
    '''
    Returns a fixed dataset, with missing values being removed, or allowing
    the user to remove or complete them manually. It prints the indices of each missing element
    for easier fixing.
    :param raw_dataset: the dataset is loaded
    :return: returns the fixed dataset
    '''

    # check if the dataset is broken: missing values
    missing_values = raw_dataset.isnull().sum()

    # array to store the indices of rows to be removed
    values_to_remove = np.array([])
    if missing_values.sum() != 0:
        print("The dataset has the following corrupt data:")
        # Store the indices of the missing values to be removed and notify the user which are missing
        for line_index in range(raw_dataset.T.shape[0]):
            for col_index in range(raw_dataset.T.shape[1]):
                if np.isnan(raw_dataset.T[col_index][line_index]):
                    values_to_remove = np.append(values_to_remove, col_index)
                    print("Line index: ", line_index, "and collumn index:", col_index)

        print("Do Y - remove line. Any other input - keep it.")
        decision = str(input())
        if decision == 'Y':
            # if the missing values are to be removed, replicated elements are not considered and
            # the elements sorted
            values_to_remove = np.unique(values_to_remove)
            for removal_index in range(np.size(values_to_remove) - 1, -1, -1):
                print("Removing", values_to_remove[removal_index])
                raw_dataset = raw_dataset.drop(labels=values_to_remove[removal_index], axis=0)
                raw_dataset = raw_dataset.reset_index(drop=True)

        else:
            print("Fix the dataset manually so that it can be loaded")
            exit()

    else:
        print("The dataset does not contain missing values in any attribute")

    return raw_dataset

dataset_loader()