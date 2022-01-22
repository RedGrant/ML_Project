# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #

# created functions
from dataset_loader import dataset_loader
from train_models import train_models
from test_models import test_models

class_type = ['binary', 'multiclass_3', 'multiclass_stars', 'multiclass_6']
# load the dataset without any missing data and divided in training, validation and test sets
dataset_binary, dataset_multiclass_3, dataset_multiclass_stars, dataset_multiclass_6 = dataset_loader(class_type)

# each class type will save the most optimal model of each methodology (logistic regression, svc, rfc, etc)
for class_to_train in class_type:
    if class_to_train == 'binary':
        binary_models = train_models(dataset_binary, class_to_train)
    elif class_to_train == 'multiclass_3':
        multiclass3_models = train_models(dataset_multiclass_3, class_to_train)
    elif class_to_train == 'multiclass_stars':
        multiclass_stars_models = train_models(dataset_multiclass_stars, class_to_train)
    elif class_to_train == 'multiclass_6':
        multiclass6_models = train_models(dataset_multiclass_6, class_to_train)

# test the models only if all the models were trained
if len(binary_models) == 6 and len(multiclass3_models) == 6 and len(multiclass_stars_models) == 6 and len(
        multiclass6_models) == 6:
    print("---------------- Test binary models ----------------")
    test_models(dataset_binary, binary_models, class_type[0])
    print("---------------- Test 3 class models ----------------")
    test_models(dataset_multiclass_3, multiclass3_models, class_type[1])
    print("---------------- Test star class models ----------------")
    test_models(dataset_multiclass_stars, multiclass_stars_models, class_type[2])
    print("---------------- Test quality class models ----------------")
    test_models(dataset_multiclass_6, multiclass6_models, class_type[3])
