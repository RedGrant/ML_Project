# Idilson
# Pedro Guedes
# Rafael Cabral
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# Libraries' import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sb
import pathlib
import os


def dataset_loader():
    '''
    Loads the dataset. The path can be changed to consider another dataset.
    Missing values can be removed, or the option of dealing with that manually is
    available for the user. TODO - update this description
    :return: dataset - returns the training, test and validation sets
    '''
    actual_dir = pathlib.Path().absolute()

    # TODO - REMOVE THIS DEBUGGER LINE
    # debbuger actual_dir:
    # actual_dir = '/home/pedroguedes/PycharmProjects/ML_Project/src'
    # actual_dir = '/home/pguedes/PycharmProjects/ML_Project/src'
    path = str(actual_dir) + '/data/winequality-red.csv'

    # load the entire dataset and labels for each attribute through pandas csv reader method
    raw_dataset = pd.read_csv(path)
    raw_dataset.head()
    # check if the dataset contains missing values
    raw_dataset = check_for_missing_values(raw_dataset)

    path = str(actual_dir) + '/figures/pairplot_figure_wine_dataset.png'
    # if the figure is not saved yet, it will be generated
    if not os.path.exists(path):
        # pair plotting the data
        pairplot_figure = sb.pairplot(raw_dataset)
        pairplot_figure = pairplot_figure.fig
        pairplot_figure.savefig(path)
        plt.close(pairplot_figure)

    training_set, test_set, validation_set = dataset_splitter(raw_dataset)

    dataset = [training_set, test_set, validation_set]

    return dataset


def check_for_missing_values(raw_dataset):
    """
    Returns a fixed dataset, with missing values being removed, or allowing
    the user to remove or complete them manually. It prints the indices of each missing element
    for easier fixing.
    :param raw_dataset: the dataset is loaded
    :return: returns the fixed dataset
    """

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


def dataset_splitter(raw_dataset):
    """
    Splits the dataset so that three sets are made available being the return of this function.
    A class field is made available in the loaded raw dataset: the classification of each wine.
    The 10 levels of classification might be used, but the dataset does not contain enough data to
    consider all the 10 levels.
    A histogram is generated with the new classification output.
    The attributes and outputs are separated from one another.
    Afterwards those are split so that the training set, validation set and testing set are made
    available.

    :param raw_dataset
    :return: training_set - 60 % of the data
             test_set - 20 % of the data
             validation_set - 20 % of the data
    """

    # new field in the raw dataset - class output with specified thresholds
    class_output = []

    # class creation depending on the quality - suggested by the dataset creators
    # however the quality is not dropped, since it can still be considered
    for quality_index in raw_dataset['quality']:
        if 0 <= quality_index <= 3:
            class_output.append('0')
        elif 3 < quality_index <= 7:
            class_output.append('1')
        elif 7 < quality_index <= 10:
            class_output.append('2')
    raw_dataset['classification'] = class_output

    # plot the labeled output as a histogram

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/figures/classification_histogram.png'

    # if the figure is not saved yet, it will be generated
    if not os.path.exists(path):
        # pair plotting the data
        histogram_figure = sb.histplot(raw_dataset['classification'], stat="percent", discrete=True)
        histogram_figure = histogram_figure.get_figure()
        histogram_figure.savefig(path)
        plt.close(histogram_figure)

    # separating attributes and outputs
    attributes = raw_dataset.iloc[:, :11]
    output = raw_dataset.iloc[:, 12]

    # set aside 20% of train and test data for evaluation
    X_train, X_test, Y_train, Y_test = train_test_split(attributes, output,
                                                        test_size=0.2, shuffle=True, random_state=1)

    # Use the same function above for the validation set, using the X_Train and Y_Train so that they are split again
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                      test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

    # generating easier output
    training_set = [X_train, Y_train]
    test_set = [X_test, Y_test]
    validation_set = [X_val, Y_val]

    return training_set, test_set, validation_set
