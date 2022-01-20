# FEUP | PDEEC | 2021/2022 | MACHINE LEARNING
# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt

# *********************************************************************************************************************#

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle
import pathlib
from confusionmatrix_generator import confusionmatrix_generator


def wine_stochastic(X_train, Y_train, X_val, Y_val, class_type):
    """
    Stochastic Gradient Descent Classifier optimal model generator.
    This estimator implements regularized linear models with stochastic gradient descent (SGD) learning:
    the gradient of the loss is estimated each sample at a time and the model is updated
    along the way with a decreasing strength schedule (aka learning rate).
    The optimal model is returned after fine-tuning a set of parameters.
    :param X_train: attributes training set
    :param Y_train: output training set
    :param X_val: attributes validation set
    :param Y_val: output validation set
    :param class_type: class output type
    :return: returns the most optimal Stochastic Gradient Descent Classifier model
    """

    # The loss function to be used
    loss_array = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']

    # Regularization term, the penalty to be applied during the training
    penalty_array = ['l2', 'l1', 'elasticnet']

    # Constant that multiplies the regularization term. The higher the value, the stronger the regularization
    alpha_array = np.arange(0.0001, 0.01, 0.0001)

    # Number of epochs
    max_iterations = np.arange(1000, 2000, 100)

    # when validation score is not improving. If set to True, it will automatically set aside a stratified fraction
    # of training data as validation and terminate training
    # when validation score returned by the score method is not improving
    early_stopping = True

    # The optimal learning rate where
    # eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou
    learning_rate = 'optimal'

    # preparing for a set of different models
    model = []
    model_accuracy = []
    model_predictions = []
    model_index = 0

    print("Training " + class_type + "SGD")
    # creating and training the models
    for loss in loss_array:
        for penalty in penalty_array:
            for alpha in alpha_array:
                for max_iter in max_iterations:
                    # append all the generated models with different parameters
                    model.append(SGDClassifier(loss=loss,
                                               penalty=penalty,
                                               alpha=alpha,
                                               max_iter=max_iter,
                                               early_stopping=early_stopping,
                                               learning_rate=learning_rate))
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
    print('Overall optimal parameters for ' + class_type + ' in Stochastic Gradient Descent '
                                                           'Classifier are:', optimized_parameters)

    # compute the individual predictions for the training and validation sets
    model_training_prediction = model[best_accuracy].predict(X_train)
    model_validation_prediction = model[best_accuracy].predict(X_val)

    model_training_accuracy = accuracy_score(Y_train, model_training_prediction, normalize=True)
    model_validation_accuracy = accuracy_score(Y_val, model_validation_prediction, normalize=True)

    # helps the user to visualize the results through the console
    print('SGD Score for ' + class_type)
    print('Training Accuracy Score is: ', model_training_accuracy)
    print('Validation Accuracy Score is: ', model_validation_accuracy)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/models/' + class_type + '_sgd_model_py3_8.sav'

    pickle.dump(model[best_accuracy], open(path, 'wb'))

    sgd_conf_matrix = confusion_matrix(Y_val, model_predictions[best_accuracy])
    print(sgd_conf_matrix)

    path = str(actual_dir) + '/figures/' + class_type + '_sgd_cmatrix_val.png'

    # generate a confusion matrix with the validation set
    confusionmatrix_generator(sgd_conf_matrix, class_type, Y_val,
                              path, 'Stochastic Gradient Descent Classifier Confusion Matrix - Validation Set')

    return model[best_accuracy]
