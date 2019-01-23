from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
import keras


TRAIN_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\train"
VALIDATION_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\validation"
TEST_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\test"
IMG_SHAPE = (224, 224)
IMG_SIZE = 224
train_batch = 64
validation_batch = 16
load_model = False

if __name__ == '__main__':

    if not load_model:
        mobile = keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
        mobile.summary()

        for layer in mobile.layers[:-6]:
            layer.trainable = False

        for layer in mobile.layers:
            print(layer, layer.trainable)

        model = Sequential()
        model.add(mobile)
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
    else:
        json_file = open(
            r'C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\mobilenet\mobile-model.json',
            'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(
            r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\mobilenet\mobile-model.h5")
        print("Loaded model from disk")
        print()
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-4),
                      metrics=['accuracy'])

    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)\
        .flow_from_directory(TRAIN_DIR, batch_size=train_batch, target_size=IMG_SHAPE)

    validation_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)\
        .flow_from_directory(VALIDATION_DIR, batch_size=validation_batch, target_size=IMG_SHAPE)

    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(TEST_DIR, batch_size=validation_batch, target_size=IMG_SHAPE)

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-4),
        metrics=["accuracy"]
    )

    history = model.fit_generator(
        train_batches,
        steps_per_epoch=train_batches.samples/train_batches.batch_size,
        epochs=20,
        validation_data=validation_batches,
        validation_steps=validation_batches.samples/validation_batches.batch_size,
        verbose=1
    )

    model_json = model.to_json()
    with open("mobile-model.json", "w") as json_file:
        json_file.write(model_json)
    model.save("mobile-model.h5")




