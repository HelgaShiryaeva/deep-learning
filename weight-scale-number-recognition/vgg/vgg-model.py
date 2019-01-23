from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop


TRAIN_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\train"
VALIDATION_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\validation"
IMG_SHAPE = (224, 224)
IMG_SIZE = 224
train_batch = 64
validation_batch = 16


def predict(filename):
    loaded_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    loaded_model.summary()

    img = load_img(filename, target_size=IMG_SHAPE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    prediction = loaded_model.predict(img)

    label = decode_predictions(prediction)
    label = label[0][0]

    print('%s (%.2f%%)' % (label[1], label[2] * 100))


if __name__ == '__main__':
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    print('********Summary**********')
    vgg_model.summary()

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

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SHAPE,
        batch_size=train_batch,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMG_SHAPE,
        class_mode='categorical',
        batch_size=validation_batch,
        shuffle=False
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=["accuracy"]
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        verbose=1
    )

    model.save("vgg-digits.h5")

