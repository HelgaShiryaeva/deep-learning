from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class DataPreprocessor:
    def __init__(self):
        self.TRAIN_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\classification\asl-alphabet\asl_alphabet_train\asl_alphabet_train\5"
        self.VALIDATION_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\classification\asl-alphabet\asl_alphabet_test\5"
        self.IMG_SHAPE = (64, 64)
        self.train_batch = 64
        self.validation_batch = 16

    def get_image_generators(self):
        train_datagen = ImageDataGenerator(
            zoom_range=0.5,
            rotation_range=60,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=self.IMG_SHAPE,
            batch_size=self.train_batch,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.VALIDATION_DIR,
            target_size=self.IMG_SHAPE,
            class_mode='categorical',
            batch_size=self.validation_batch,
            shuffle=False
        )

        return train_generator, validation_generator

    def preprocess_image(self, filename):
        img = load_img(filename, target_size=self.IMG_SHAPE)
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        print(img.shape)
        img = preprocess_input(img)

        return img

    def decode_prediction(self, prediction):
        if prediction == 0:
            return 'A'
        elif prediction == 1:
            return 'B'
        elif prediction == 2:
            return 'C'
        elif prediction == 3:
            return 'D'
        elif prediction == 4:
            return 'E'
