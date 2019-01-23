import numpy as np
from tqdm import tqdm
import os
import cv2

from albumentations import (CLAHE, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, OneOf, Compose
)

IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
DATASET_TRAIN_DIR = r'C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\train'
DATASET_TEST_DIR = r'C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\dataset\test'


def load_image(path_to_image):
    image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def augment(aug, image):
    return aug(image=image.copy())['image']


def augment_flips_color(p=.5):
    return Compose([
        CLAHE(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        Blur(blur_limit=3),
        HueSaturationValue()
    ], p=p)


def strong_aug(p=.5):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def bright_contrast_aug():
    return Compose([
        RandomBrightnessContrast()
    ])


def motion_blur_aug():
    return MotionBlur(blur_limit=6, always_apply=True)


def augment_and_save(set_folder_path):
    for folder in tqdm(os.listdir(set_folder_path)):
        path_to_folder = os.path.join(set_folder_path, folder)
        for img in tqdm(os.listdir(path_to_folder)):
            path_to_image = os.path.join(path_to_folder, img)
            img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SHAPE)
            aug = motion_blur_aug()
            augmented = augment(aug, img)
            image = str(path_to_image.split('\\')[-1])
            image_name = 'blur-' + image
            cv2.imwrite(os.path.join(path_to_folder, image_name), augmented)
            aug = augment_flips_color(p=1)
            augmented = augment(aug, img)
            image_name = 'flips_color' + image
            cv2.imwrite(os.path.join(path_to_folder, image_name), augmented)
            aug = strong_aug(p=1)
            augmented = augment(aug, img)
            image_name = 'strong' + image
            cv2.imwrite(os.path.join(path_to_folder, image_name), augmented)
            aug = bright_contrast_aug()
            augmented = augment(aug, img)
            image_name = 'bright_contrast' + image
            cv2.imwrite(os.path.join(path_to_folder, image_name), augmented)


if __name__ == '__main__':
    augment_and_save(DATASET_TRAIN_DIR)
