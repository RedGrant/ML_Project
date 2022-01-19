# FEUP | PDEEC | 2021/2022 | MACHINE LEARNING
# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt

# *********************************************************************************************************************#

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os


def wine_rfc(X_train, Y_train, X_val, Y_val):
    """
    Random Forest classifier optimal model generator.
    a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and
    uses averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default),
    otherwise the whole dataset is used to build each tree.
    The optimal model is returned after fine tuning a set of parameters.
    :param X_train: attributes training set
    :param Y_train: output training set
    :param X_val: attributes validation set
    :param Y_val: output validation set
    :return: returns the most optimal logistic regression model
    """


    # number of trees in the forest
    n_estimators_array = np.arange(10, 300, 10)

    # Number of features to consider at every split
    max_features_array = ['auto', 'sqrt', 'log2']

    # Minimum number of samples required to split a node
    min_samples_split_array = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf_array = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap_array = [True, False]

    # preparing for a set of different models
    model = []
    model_accuracy = []
    model_predictions = []
    model_index = 0

    # creating and training the models
    for n_estimators in n_estimators_array:
        for max_features in max_features_array:
            for min_samples_split in min_samples_split_array:
                for min_samples_leaf in min_samples_leaf_array:
                    for bootstrap in bootstrap_array:
                        # append all the generated models with different parameters
                        model.append(RandomForestClassifier(n_estimators=n_estimators,
                                                            max_features=max_features,
                                                            min_samples_split=min_samples_split,
                                                            min_samples_leaf=min_samples_leaf,
                                                            bootstrap=bootstrap))
                        # fit the data to each model
                        model[model_index].fit(X_train, Y_train)
                        # predict the values with the validation set
                        model_predictions.append(model[model_index].predict(X_val))
                        # compute the current model's accuracy, normalized
                        model_accuracy.append(accuracy_score(Y_val, model_predictions[model_index], normalize=True))
                        model_index += 1

    # find which model had the best accuracy
    best_accuracy = np.argmax(model_accuracy)
    # print the parameters
    optimized_parameters = model[best_accuracy].get_params(deep=True)
    print('Overall optimal parameters in Random Forest Classifier: ', optimized_parameters)

    # compute the individual predictions for the training and validation sets
    model_training_prediction = model[best_accuracy].predict(X_train)
    model_validation_prediction = model[best_accuracy].predict(X_val)

    model_training_accuracy = accuracy_score(Y_train, model_training_prediction, normalize=True)
    model_validation_accuracy = accuracy_score(Y_val, model_validation_prediction, normalize=True)

    # helps the user to visualize the results through the console
    print('RFC Score')
    print('Training Accuracy Score is: ', model_training_accuracy)
    print('Validation Accuracy Score is: ', model_validation_accuracy)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/models/rfc_model_py3_8.sav'
    pickle.dump(model[best_accuracy], open(path, 'wb'))

    rfc_conf_matrix = confusion_matrix(Y_val, model_predictions[best_accuracy])
    print(rfc_conf_matrix)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/figures/rfc_cmatrix_val.png'

    # Generate a Model Validation Confusion Matrix
    # if the figure is not saved yet, it will be generated
    if not os.path.exists(path):
        confusion_matrix_plot = plt.subplot()
        sns.heatmap(rfc_conf_matrix, annot=True, fmt='g', ax=confusion_matrix_plot)
        # labels, title and ticks
        confusion_matrix_plot.set_ylabel('True labels')
        confusion_matrix_plot.set_title('RFC Confusion Matrix - Validation Set')
        confusion_matrix_plot.xaxis.set_ticklabels(['Awful', 'Average', 'Excellent'])
        confusion_matrix_plot.yaxis.set_ticklabels(['Awful', 'Average', 'Excellent'])
        confusion_matrix_plot.figure.savefig(path)

    return model[best_accuracy]
