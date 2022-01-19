# FEUP | PDEEC | 2021/2022 | MACHINE LEARNING
# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt

# *********************************************************************************************************************#

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle
from confusionmatrix_generator import confusionmatrix_generator
import pathlib


def wine_svc(X_train, Y_train, X_val, Y_val, class_type):
    """
    SVC classifier optimal model generator.
    :param X_train: attributes training set
    :param Y_train: output training set
    :param X_val: attributes validation set
    :param Y_val: output validation set
    :return: returns the most optimal logistic regression model
    """

    # Specifies the kernel type to be used in the algorithm.
    kernel_array = ['linear', 'poly', 'rbf', 'sigmoid']

    # C parameter - Regularization parameters for SVC models
    # Inverse of regularization strength. Smaller the stronger the regularization
    c_starting = 0.5
    c_ending = 5
    c_step = 0.1

    C_parameter = np.arange(c_starting, c_ending, c_step)

    # degree - polynomial degree parameters for SVC models (only considered in poly, ignored in the other kernels)
    degree_array = np.arange(1, 6, 1)

    gamma_array = np.arange(0.1, 2, 0.1)

    # preparing for a set of different models
    model = []
    model_accuracy = []
    model_predictions = []
    model_index = 0

    # creating and training the models
    for kernel in kernel_array:
        for degree in degree_array:
            for C in C_parameter:
                for gamma in gamma_array:
                    # append all the generated models with different parameters
                    model.append(svm.SVC(C=C,
                                         kernel=kernel,
                                         degree=degree,
                                         gamma=gamma))

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
    print('Overall optimal parameters for ' + class_type + ' in SVC are:', optimized_parameters)

    # compute the individual predictions for the training and validation sets
    model_training_prediction = model[best_accuracy].predict(X_train)
    model_validation_prediction = model[best_accuracy].predict(X_val)

    model_training_accuracy = accuracy_score(Y_train, model_training_prediction, normalize=True)
    model_validation_accuracy = accuracy_score(Y_val, model_validation_prediction, normalize=True)

    # helps the user to visualize the results through the console
    print('SVC Score for ' + class_type)
    print('Training Accuracy Score is: ', model_training_accuracy)
    print('Validation Accuracy Score is: ', model_validation_accuracy)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/models/' + class_type + '_vc_model_py3_8.sav'
    pickle.dump(model[best_accuracy], open(path, 'wb'))

    svc_conf_matrix = confusion_matrix(Y_val, model_predictions[best_accuracy])
    print(svc_conf_matrix)

    path = str(actual_dir) + '/figures/' + class_type + '_svc_cmatrix_val.png'

    # generate a confusion matrix with the validation set
    confusionmatrix_generator(svc_conf_matrix, class_type, Y_val,
                              path, 'SVC Confusion Matrix - Validation Set')

    return model[best_accuracy]
