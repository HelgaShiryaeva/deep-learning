from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from data import DataPreprocessor
from plot import plot_utils as plot
from train import train_utils as loader


def built_cnn():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_network(model, train_generator, validation_generator):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=["accuracy"]
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=2
    )

    return history


if __name__ == '__main__':
    preprocessor = DataPreprocessor.DataPreprocessor()
    model = built_cnn()
    #model = loader.load_model_from_json('3_conv_adam_1e-3_cat_cross_v2.json', '3_conv_adam_1e-3_cat_cross_v2.h5')
    train_generator, validation_generator = preprocessor.get_image_generators()
    history = train_network(model, train_generator, validation_generator)
    loader.save_model(model, '3_conv_adam_1e-3_v2.json', '3_conv_adam_1e-3_v2.h5')
    plot.plot_history(history)

