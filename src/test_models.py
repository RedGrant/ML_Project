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

# import created functions
from confusionmatrix_generator import confusionmatrix_generator


def test_models(dataset, models, class_type):
    model_name = ['Log_Reg_Model', 'SVC_Model', 'RFC_Model', 'SGD_Model', 'DL_Deep_Model', 'DL_Shallow_Model']

    # extract the split test data from the already pre-processed data set
    X_test, Y_test = dataset[2]

    # scale the attributes
    scaler = StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)

    # Transform series to numpy
    Y_test = Y_test.to_numpy()

    model_accuracy = []
    model_f1score = []
    model_predictions = []
    # test all the classic models
    for model_index in range(len(model_name)):
        print(model_name[model_index])
        model_predictions.append(models[model_index].predict(X_test))
        # compute the current model's accuracy, normalized
        model_accuracy.append(accuracy_score(Y_test, model_predictions[model_index], normalize=True))
        if class_type == 'binary':
            model_f1score.append(f1_score(Y_test, model_predictions[model_index], average='binary'))
        else:
            # this average was chosen due to label imbalance in the dataset
            model_f1score.append(f1_score(Y_test, model_predictions[model_index], average='weighted'))

        # save classification report as csv file
        actual_dir = pathlib.Path().absolute()
        path = str(actual_dir) + '/classification_reports/' + model_name[
            model_index] + '_classification_report_' + class_type + '.csv'
        df = classification_report(Y_test, model_predictions[model_index])
        dataframe = pd.DataFrame.from_dict(df)
        dataframe.to_csv(path, index=False)

        print(classification_report(Y_test, model_predictions[model_index]))

        print(model_name[model_index] + ' for ' + class_type + ':')
        print('Test Accuracy Score is: ', model_accuracy[model_index])
        print('Test f1 Score is: ', model_f1score[model_index])

        # generate confusion matrices for each model test
        conf_matrix = confusion_matrix(Y_test, model_predictions[model_index])

        path = str(actual_dir) + '/figures/test_results/' + class_type + '_' + model_name[
            model_index] + 'sgd_cmatrix_test.png'

        # generate a confusion matrix with the validation set
        confusionmatrix_generator(conf_matrix, class_type, Y_test,
                                  path, model_name[model_index] + ' Confusion Matrix - Test Set')
