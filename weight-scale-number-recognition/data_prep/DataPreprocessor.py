from keras.preprocessing.image import ImageDataGenerator
import cv2


class DataPreprocessor:
    def __init__(self):
        self.TRAIN_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\train"
        self.VALIDATION_DIR = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\test"
        self.IMG_SHAPE = (64, 64)
        self.train_batch = 64
        self.validation_batch = 16

    def get_image_generators(self):
        train_datagen = ImageDataGenerator(
            zoom_range=0.5,
            rotation_range=60,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest',
            rescale=1./255
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

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

    def preprocess_image(self, path_to_img):
        img = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        return img
