# Pedro Guedes - up202101510@up.pt
# Rafael Cabral - up201609762@edu.fe.up.pt
# Idilson Nhamage - up202011161@edu.fe.up.pt
# FEUP | PDEEC | Machine Learning 2021/2022

# *********************************** ---------------------------------- *********************************** #
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pathlib
from tensorflow.keras.layers import Dense
import tensorflow as tf
from matplotlib import pyplot


def wine_deep(X_train, Y_train, X_val, Y_val, class_type):
    if class_type != 'binary':

        output_neurons = (np.unique(Y_val)).size
        encoder = LabelEncoder()
        encoder.fit(Y_train)
        Y_train = encoder.fit_transform(Y_train)
        encoder = LabelEncoder()
        encoder.fit(Y_val)
        Y_val = encoder.fit_transform(Y_val)

        loss = "categorical_crossentropy"
        Y_train = tf.keras.utils.to_categorical(Y_train, output_neurons)
        Y_val = tf.keras.utils.to_categorical(Y_val, output_neurons)
        output_activation = 'softmax'

    else:
        output_neurons = 1
        loss = "binary_crossentropy"
        output_activation = 'relu'

    # shallow model training:
    shallow_model = tf.keras.models.Sequential()

    # input layer
    shallow_model.add(tf.keras.Input(shape=(11,)))
    # added only one fat hidden layer
    shallow_model.add(Dense(50, activation='relu', name='layer'))
    # output layer
    shallow_model.add(Dense(output_neurons, activation=output_activation))

    shallow_model.compile(loss=loss,
                          optimizer='adam',
                          metrics=['accuracy'])

    shallow_model_training_history = shallow_model.fit(X_train, Y_train,
                                                      epochs=20,
                                                      batch_size=1,
                                                      validation_data=(X_val, Y_val),
                                                      validation_steps=10)

    # deep model training:
    deep_model = tf.keras.models.Sequential()

    # input layer
    deep_model.add(tf.keras.Input(shape=(11,)))
    for layer_index in range(12, 2, -2):
        deep_model.add(Dense(layer_index, activation='relu'))

    # output layer
    deep_model.add(Dense(output_neurons, activation=output_activation))

    deep_model.compile(loss=loss,
                       optimizer='adam',
                       metrics=['accuracy'])

    deep_model_training_history = deep_model.fit(X_train, Y_train,
                                                 epochs=20,
                                                 batch_size=1,
                                                 validation_data=(X_val, Y_val),
                                                 validation_steps=10)
    deep_model.summary()
    shallow_model.summary()

    actual_dir = pathlib.Path().absolute()

    # save models
    path = str(actual_dir) + '/models/' + class_type + '_dl_deep_py3_8.sav'
    deep_model.save(path)
    path = str(actual_dir) + '/models/' + class_type + '_dl_shallow_py3_8.sav'
    shallow_model.save(path)

    # save accuracy and loss plots

    # Deep Model
    path = str(actual_dir) + '/figures/' + class_type + '_dl_deep_accuracy.png'
    pyplot.plot()
    pyplot.title('Deep Learning Deep Accuracy ' + class_type)
    pyplot.plot(deep_model_training_history.history['accuracy'], label='train')
    pyplot.legend()
    pyplot.savefig(path)
    pyplot.close('all')

    path = str(actual_dir) + '/figures/' + class_type + '_dl_deep_loss.png'
    pyplot.plot()
    pyplot.title('Deep Learning Deep Loss ' + class_type)
    pyplot.plot(deep_model_training_history.history['loss'], label='train')
    pyplot.legend()
    pyplot.savefig(path)
    pyplot.close('all')

    # Shallow Model
    path = str(actual_dir) + '/figures/' + class_type + '_dl_shallow_accuracy.png'
    pyplot.plot()
    pyplot.title('Deep Learning Shallow Accuracy ' + class_type)
    pyplot.plot(shallow_model_training_history.history['accuracy'], label='train')
    pyplot.legend()
    pyplot.savefig(path)
    pyplot.close('all')

    path = str(actual_dir) + '/figures/' + class_type + '_dl_shallow_loss.png'
    pyplot.plot()
    pyplot.title('Deep Learning Shallow Loss ' + class_type)
    pyplot.plot(shallow_model_training_history.history['loss'], label='train')
    pyplot.legend()
    pyplot.savefig(path)
    pyplot.close('all')

    return shallow_model, deep_model
