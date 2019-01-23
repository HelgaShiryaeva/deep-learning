import numpy as np
import cv2
from matplotlib import pyplot as plt

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


def load_image(path_to_image):
    data = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return image


def augment_and_show(aug, image, title='no'):
    image = aug(image=image)['image']
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=30)
    plt.imshow(image)
    plt.show()


def augment(aug, image):
    return aug(image=image)['image']


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


def plot(image_plot):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Without augmentation', fontsize=30)
    plt.imshow(image_plot)
    plt.show()


if __name__ == '__main__':
    image = load_image('test_image.jpg')
    aug = motion_blur_aug()
    augment_and_show(aug, image, 'Motion Blur')
    aug = augment_flips_color(p=1)
    augment_and_show(aug, image, 'Flips Color')
    aug = strong_aug(p=1)
    augment_and_show(aug, image, 'Strong')
    aug = bright_contrast_aug()
    augment_and_show(aug, image, 'Brightness and contrast')