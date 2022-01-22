# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle
import pathlib
from confusionmatrix_generator import confusionmatrix_generator
from tqdm import tqdm

def wine_log_regression(X_train, Y_train, X_val, Y_val, class_type):
    """
    Computes the Logistic Regression for the Red wine classification problem. It returns the most optimal model out of a
    set of generated model with different parameters.
    :param X_train: attributes training set
    :param Y_train: output training set
    :param X_val: attributes validation set
    :param Y_val: output validation set
    :param class_type: defines the type of output expected from the loaded training and validation sets
    :return: returns the most optimal logistic regression model
    """

    # solver parameter. Algorithm to use in the optimization problem. It depends on the class output type
    if class_type != 'binary':
        solver_array = ['newton-cg', 'sag', 'saga', 'lbfgs']
        multi_class = 'multinomial'
    else:
        solver_array = ['liblinear', 'sag', 'saga', 'lbfgs']
        multi_class = 'ovr'

    # penalty type
    penalty = 'l2'

    # max_iter parameter
    maximum_iterations = np.arange(100, 300, 10)

    # Inverse of regularization strength. Smaller the stronger the regularization
    c_starting = 0.5
    c_ending = 10
    c_step = 0.5

    C_parameter = np.arange(c_starting, c_ending, c_step)

    model = []
    model_predictions = []
    model_accuracy = []
    model_index = 0

    print("Training " + class_type + " Log Reg")
    # test which is the best model varying the parameters set above
    with tqdm(total=len(solver_array)) as pbar:
        for solver in solver_array:
            for iteration in maximum_iterations:
                for C in C_parameter:
                    # appending the models in the list
                    model.append(LogisticRegression(penalty='l2',
                                                    C=C,
                                                    solver=solver,
                                                    max_iter=iteration,
                                                    multi_class=multi_class))
                    # fit the data to the current model
                    model[model_index].fit(X_train, Y_train)
                    # predict the values with the validation set
                    model_predictions.append(model[model_index].predict(X_val))
                    # compute the current model's accuracy, normalized
                    model_accuracy.append(accuracy_score(Y_val, model_predictions[model_index], normalize=True))
                    model_index += 1
            pbar.update(1)


    # find which model had the best accuracy
    best_accuracy = np.argmax(model_accuracy)
    # print the parameters
    optimized_parameters = model[best_accuracy].get_params(deep=True)
    print('Overall optimal parameters for ' + class_type + ' are:', optimized_parameters)

    # compute the individual predictions for the training and validation sets
    model_training_prediction = model[best_accuracy].predict(X_train)
    model_validation_prediction = model[best_accuracy].predict(X_val)

    model_training_accuracy = accuracy_score(Y_train, model_training_prediction, normalize=True)
    model_validation_accuracy = accuracy_score(Y_val, model_validation_prediction, normalize=True)

    # helps the user to visualize the results through the console
    print('Logistic Regression Score for ' + class_type)
    print('Training Accuracy Score is: ', model_training_accuracy)
    print('Validation Accuracy Score is: ', model_validation_accuracy)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/models/' + class_type + '_log_reg_model_py3_8.sav'
    pickle.dump(model[best_accuracy], open(path, 'wb'))

    log_reg_conf_matrix = confusion_matrix(Y_val, model_predictions[best_accuracy])
    print(log_reg_conf_matrix)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/figures/' + class_type + '_logreg_cmatrix_val.png'

    # generate a confusion matrix with the validation set
    confusionmatrix_generator(log_reg_conf_matrix, class_type, Y_val,
                              path, class_type + ' ' + ' Log Reg Confusion Matrix - Validation Set')

    return model[best_accuracy]
