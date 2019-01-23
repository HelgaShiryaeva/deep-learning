import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import backend as K
from random import shuffle
from keras.models import model_from_json
import numpy as np
from tqdm import tqdm
import os
import cv2

IMG_SIZE = 28
DATASET_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset"
channels_first = True
load_model = False
new_model_train = True


def split_data(data):
    return np.array(data[:400]), np.array(data[400:436])


def load_data():
    data = []
    for img in tqdm(os.listdir(DATASET_DIR)):
        label = int(img.split("(")[0])
        if label > 10:
            label = int(label/10)
        if label == -1:
            continue
        path_to_image = os.path.join(DATASET_DIR, img)
        img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([img, label])
    shuffle(data)
    return split_data(data)


def get_data():
    img_label_train, img_label_test = load_data()
    new_train = [ilt[0] for ilt in img_label_train]
    new_test = [ilt[0] for ilt in img_label_test]
    new_train_labels = keras.utils.to_categorical([ilt[1] for ilt in img_label_train], 10)
    new_test_labels = keras.utils.to_categorical([ilt[1] for ilt in img_label_test], 10)

    return np.array(new_train), np.array(new_train_labels), np.array(new_test), np.array(new_test_labels)


def load_model_from_json():
    json_file = None
    if new_model_train:
        json_file = open('3_conv_adam_cat_cross_v2-model-vgg.json', 'r')
    else:
        json_file = open('model.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into 3_conv_adam_cat_cross_v2 model
    if new_model_train:
        model.load_weights("3_conv_adam_cat_cross_v2-model.h5")
    else:
        model.load_weights("model.h5")

    print("Loaded model from disk")
    return model


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("3_conv_adam_cat_cross_v2-model-vgg.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("3_conv_adam_cat_cross_v2-model.h5")
    print("Saved model to disk")


new_train, new_train_labels, new_test, new_test_labels = get_data()

if K.image_data_format() == 'channels_first':
    new_train = new_train.reshape(new_train.shape[0], 1, IMG_SIZE, IMG_SIZE)
    new_test = new_test.reshape(new_test.shape[0], 1, IMG_SIZE, IMG_SIZE)
    input_shape = (1, IMG_SIZE, IMG_SIZE)
else:
    channels_first = False
    print(new_train.shape)
    new_train = new_train.reshape(new_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
    new_test = new_test.reshape(new_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

if K.image_data_format() == 'channels_first':
    input_shape = (1, IMG_SIZE, IMG_SIZE)
else:
    input_shape = (IMG_SIZE, IMG_SIZE, 1)


data_format = 'channels_last'
if channels_first:
    data_format = 'channels_first'

model = load_model_from_json()
model.summary()

if not load_model:
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), data_format=data_format, padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), data_format=data_format, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5), data_format=data_format, padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (5, 5), data_format=data_format, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), data_format=data_format, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(new_train, new_train_labels,
                    batch_size=32,
                    epochs=30,
                    verbose=2,
                    validation_data=(new_test, new_test_labels))


score = model.evaluate(new_test, new_test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 
save_model(model)
