# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #
# import libraries
import pandas as pd
import pathlib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
# import created functions
from confusionmatrix_generator import confusionmatrix_generator


def test_models(dataset, models, class_type):
    """
    Tests the trained models with the already split test set.
    Classification reports are generated and saved as txt files.
    Confusion matrices are saved as figures.
    :param dataset: the dataset for the specific class output type
    :param models: the trained models
    :param class_type: the class output type
    :return: None
    """
    model_name = ['Log_Reg_Model', 'SVC_Model', 'RFC_Model', 'SGD_Model', 'DL_Deep_Model', 'DL_Shallow_Model']

    # extract the split test data from the already pre-processed data set
    X_test, Y_test = dataset[2]

    # scale the attributes
    scaler = StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)

    # Transform series to numpy
    Y_test = Y_test.to_numpy()
    Y_test = np.round(Y_test)

    model_accuracy = []
    model_f1score = []
    model_predictions = []
    # test all the classic models
    for model_index in range(len(model_name)):
        print(model_name[model_index])
        model_predictions.append(np.round(models[model_index].predict(X_test)))

        # encode the data for the deep learning models
        if (class_type != 'binary') and (model_index > 3):
            unique_values_label_encoder = np.sort(np.unique(Y_test))
            model_predictions[model_index] = np.argmax(model_predictions[model_index], axis=1)
            encoder = LabelEncoder()
            encoder.fit(Y_test)
            Y_test = encoder.fit_transform(Y_test)

        # compute the current model's accuracy, normalized
        model_accuracy.append(accuracy_score(Y_test, model_predictions[model_index], normalize=True))
        if class_type == 'binary':
            model_f1score.append(f1_score(Y_test, model_predictions[model_index], average='binary'))
        else:
            # this average was chosen due to label imbalance in the dataset
            model_f1score.append(f1_score(Y_test, model_predictions[model_index], average='weighted'))

        # convert the hot encoder to original values
        if (class_type != 'binary') and (model_index > 3):

            # first for the test set
            for index in range(len(Y_test)):
                Y_test[index] = unique_values_label_encoder[Y_test[index]]

            # then for the predicted values
            for index in range(len(model_predictions[model_index])):
                model_predictions[model_index][index] = unique_values_label_encoder[
                    model_predictions[model_index][index]]

        # save classification report as txt file
        actual_dir = pathlib.Path().absolute()
        path = str(actual_dir) + '/classification_reports/' + model_name[
            model_index] + " Classification Report " + \
               class_type + ".txt"
        print(classification_report(Y_test, model_predictions[model_index]))
        report = classification_report(Y_test, model_predictions[model_index])
        with open(path, "w") as text_file:
            text_file.write(report)

        # print classification report
        print(report)

        conf_matrix = confusion_matrix(Y_test, model_predictions[model_index])

        path = str(actual_dir) + '/figures/test_results/' + class_type + '_' + model_name[
            model_index] + 'sgd_cmatrix_test.png'

        # generate a confusion matrix with the validation set
        confusionmatrix_generator(conf_matrix, class_type, Y_test,
                                  path, class_type + ' ' + model_name[model_index] + ' Confusion Matrix - Test Set')

    sorted_models = np.argsort(model_accuracy)

    print("The accuracies from worse to best in " + class_type + " are:")
    for index in sorted_models:
        print(str(model_name[index]) + " - " + str(model_accuracy[index]))

    return None
