# FEUP | PDEEC | 2021/2022 | MACHINE LEARNING
# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt

# *********************************************************************************************************************#

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pickle
import pathlib
from confusionmatrix_generator import confusionmatrix_generator


def wine_rfc(X_train, Y_train, X_val, Y_val, class_type):
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
    print('Overall optimal parameters for ' + class_type + ' in Random Forest Classifier are:', optimized_parameters)

    # compute the individual predictions for the training and validation sets
    model_training_prediction = model[best_accuracy].predict(X_train)
    model_validation_prediction = model[best_accuracy].predict(X_val)

    model_training_accuracy = accuracy_score(Y_train, model_training_prediction, normalize=True)
    model_validation_accuracy = accuracy_score(Y_val, model_validation_prediction, normalize=True)

    # helps the user to visualize the results through the console
    print('RFC Score for ' + class_type)
    print('Training Accuracy Score is: ', model_training_accuracy)
    print('Validation Accuracy Score is: ', model_validation_accuracy)

    actual_dir = pathlib.Path().absolute()
    path = str(actual_dir) + '/models/' + class_type + '_rfc_model_py3_8.sav'

    pickle.dump(model[best_accuracy], open(path, 'wb'))

    rfc_conf_matrix = confusion_matrix(Y_val, model_predictions[best_accuracy])
    print(rfc_conf_matrix)

    path = str(actual_dir) + '/figures/' + class_type + '_rfc_cmatrix_val.png'

    # generate a confusion matrix with the validation set
    confusionmatrix_generator(rfc_conf_matrix, class_type, Y_val,
                              path, 'Random Forest Classifier Confusion Matrix - Validation Set')

    return model[best_accuracy]
