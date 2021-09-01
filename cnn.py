import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc_values"])
    Y = np.array(data["labels"])
    return X, Y


def split_data(test_size, validation_size):
    # load data
    X, Y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=validation_size)

    # adding a new dimension so the array has depth of 1
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(input_shape):

    # create model
    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(
         32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization()) # normalizes the activation outputs and speeds up learning and adds robustness
    # 2nd conv layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    # 3rd conv layer
    model.add(keras.layers.Conv2D(
        32, (2, 2), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    # output layer
    model.add(keras.layers.Dense(10,activation="softmax"))

    return model

# create train, validation and test sets

X_train, X_validation, X_test, Y_train, Y_validation, Y_test = split_data(
    0.25, 0.2)


# CNN model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = build_model(input_shape)

# compile model
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# train model
model.fit(X_train, Y_train, validation_data=(X_validation,Y_validation), batch_size=32, epochs=30)

# evaluate the CNN
test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy on test set is: {}".format(test_accuracy))

model.save('cnn-mfcc')