import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def confusionmatrix_generator(log_reg_conf_matrix, class_type, class_outputs, path, title):
    """
    Generates confusion matrices depending on the unique class outputs and class type.
    Saves one if none exist.
    :param log_reg_conf_matrix: computed confusion matrix to be plotted and saved
    :param class_type: type of expected output
    :param class_outputs: outputs to search for unique values
    :param path: path to check if exist. Otherwise, the figure will be saved there
    :param title: Title of the figure to be saved.
    :return: void
    """
    # Generate a Model Validation Confusion Matrix
    # if the figure is not saved yet, it will be generated
    if not os.path.exists(path):
        confusion_matrix_plot = plt.subplot()
        sns.heatmap(log_reg_conf_matrix, annot=True, fmt='g', ax=confusion_matrix_plot)
        # labels, title and ticks
        confusion_matrix_plot.set_xlabel('Predicted labels')
        confusion_matrix_plot.set_ylabel('True labels')
        confusion_matrix_plot.set_title(title)
        if class_type == 'binary':
            confusion_matrix_plot.xaxis.set_ticklabels(['Bad', 'Good'])
            confusion_matrix_plot.yaxis.set_ticklabels(['Bad', 'Good'])

        elif class_type == 'multiclass_3':
            confusion_matrix_plot.xaxis.set_ticklabels(['Awful', 'Average', 'Excellent'])
            confusion_matrix_plot.yaxis.set_ticklabels(['Awful', 'Average', 'Excellent'])
        elif class_type == 'multiclass_5':
            confusion_matrix_plot.xaxis.set_ticklabels(stars_printer(class_outputs))
            confusion_matrix_plot.yaxis.set_ticklabels(stars_printer(class_outputs))

        confusion_matrix_plot.figure.savefig(path)
        plt.close('all')


def stars_printer(class_values):
    """
    Generates the stars from 1 to 5 depending on the classifcation type
    :param class_values: search for the different values and attribute the correspondent star to each axis
    :return: the start list to be plotted
    """
    possible_stars = np.unique(class_values)
    possible_stars = np.sort(possible_stars)
    star_list = []
    for index in possible_stars:
        if index == '1':
            star_list.append('*')
        elif index == '2':
            star_list.append('**')
        elif index == '3':
            star_list.append('***')
        elif index == '4':
            star_list.append('****')
        elif index == '5':
            star_list.append('*****')

    return star_list
