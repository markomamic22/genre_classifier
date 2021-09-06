import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator

# create image data generators
train_dir = "spectrograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(
    288, 432), color_mode="rgba", class_mode='categorical', batch_size=128)

validation_dir = "spectrograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir, target_size=(
    288, 432), color_mode='rgba', class_mode='categorical', batch_size=128)

# model creation


def GenreModel(input_shape=(288, 432, 4)):

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(8, (3, 3), strides=(1, 1),
                                  activation="relu", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(
        16, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(
        64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(
        128, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization(axis=3))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(
        10, activation='softmax', name='fc10'))

    return model


model = GenreModel(input_shape=(288, 432, 4))

opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_generator, epochs=70,
          validation_data=vali_generator, verbose=1)
model.save('spectrogram-modelv4')

# spectrogram model 70
# spectrogram model v2 78
# v3 80
