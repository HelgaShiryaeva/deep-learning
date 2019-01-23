from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop

TRAIN_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\train"
VALIDATION_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\validation"
IMG_SHAPE = (224, 224)
IMG_SIZE = 224


def predict(filename):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in vgg_model.layers[:-4]:
        layer.trainable = False

    for layer in vgg_model.layers:
        print(layer, layer.trainable)

    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.load_weights("vgg-digits.h5")
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=["accuracy"]
    )

    img = load_img(filename, target_size=IMG_SHAPE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    print(img.shape)
    img = preprocess_input(img)

    prediction = model.predict(img)
    print(prediction.argmax())


if __name__ == '__main__':
    predict(r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\util\7.jpg")