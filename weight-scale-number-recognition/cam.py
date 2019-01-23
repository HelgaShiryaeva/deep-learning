import cv2
import numpy as np
from keras.models import model_from_json
from keras import backend as K
from detection.BlueDetector import BlueDetector

is_ip_camera = False
ip_camera_url = 'http://admin:Password@192.168.115.135:8080/stream/video/mjpeg'

black_rgb_tuple = (0, 0, 0)
green_rgb_tuple = (0, 255, 0)
red_rgb_tuple = (0, 0, 255)
white_rgb_tuple = (255, 255, 255)
IMG_SIZE = (28, 28)
FRAME_IMG_SIZE = (680, 440)
DIGITS_MAX_NUM = 5
OFFSET = 10


def load_model():
    json_file = open('3_conv_adam_cat_cross_v2-model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("3_conv_adam_cat_cross_v2-model.h5")
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def calculate_dot_number(answers):
    number = 0.0
    max_power = 3
    for i, digit in enumerate(answers):
        if i != len(answers) - 2:
            if i != len(answers) - 1:
               number += digit * 10 ** (max_power - i)
            else:
                number += digit * 10 ** (max_power - i + 1)



    return number


def prepare_gray_image(img):
    image = cv2.resize(img, IMG_SIZE)
    image = np.array(image)
    image = image.astype('float32')
    # newImage_1 /= 255

    if K.image_data_format() == 'channels_first':
        image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1])
    else:
        image = image.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)
        image = np.expand_dims(image, axis=0)

    return image


def process_self_detected_rects():
    model = load_model()
    if is_ip_camera:
        cap = cv2.VideoCapture(ip_camera_url)
    else:
        cap = cv2.VideoCapture(0)

    detector = BlueDetector(FRAME_IMG_SIZE)

    while True:
        answers = []
        ret, img = cap.read()
        print(img)
        img = cv2.resize(img, FRAME_IMG_SIZE)
        img_color = img.copy()
        rectangles = detector.get_bounding_rects(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        number_size = len(rectangles)

        for i in range(0, number_size):
            bound = rectangles[i]
            x, y, w, h = cv2.boundingRect(bound)

            image = img[y - OFFSET: y + h + OFFSET, x - OFFSET: x + w + OFFSET]
            if len(image) > 4:
                image = prepare_gray_image(image)
                answer = model.predict(image).argmax()
                answers.append(answer)

        number = calculate_dot_number(answers)
        cv2.putText(img_color, str(number), (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, red_rgb_tuple, 3)
        cv2.imshow("Frame", img_color)

        k = cv2.waitKey(33)
        if k == 32:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break


def main():
    process_self_detected_rects()


if __name__ == '__main__':
    main()

