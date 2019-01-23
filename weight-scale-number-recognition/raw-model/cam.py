import cv2
import numpy as np
from keras.models import model_from_json
from keras import backend as K

is_ip_camera = False
ip_camera_url = 'http://admin:Password@192.168.115.135:8080/stream/video/mjpeg'

y_coord, width, height = 295, 145, 200
black_rgb_tuple = (0, 0, 0)
green_rgb_tuple = (0, 255, 0)
red_rgb_tuple = (0, 0, 255)
white_rgb_tuple = (255, 255, 255)
IMG_SIZE = (28, 28)
FRAME_IMG_SIZE = (680, 440)
DIGITS_MAX_NUM = 5
history_width = 450
history_height = 600
OFFSET = 7

boxes_x_coord = [310, 445, 570, 700, 895]


def get_img_contour_thresh_1(image, x):
    x, y, w, h = x, y_coord, width, height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gray = gray[y: y + h, x: x + w]
    thresh = thresh[y: y + h, x: x + w]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    return contours, gray


def load_model():
    json_file = open('3_conv_adam_cat_cross_v2-model-vgg.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("3_conv_adam_cat_cross_v2-model.h5")
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def add_answer_on_frame(answer, current_x, img):
    cv2.rectangle(img, (current_x, y_coord), (current_x + width, y_coord + height), green_rgb_tuple, 1)
    if answer == 10:
        answer = 'empty'
    cv2.putText(img, str(answer), (current_x + 70, y_coord - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, black_rgb_tuple, 2)


def calculate_number(answers):
    number = 0.0
    max_power = 3
    for i, digit in enumerate(answers):
        if digit == 10:
            digit = 0
        number += digit * 10 ** (max_power - i)

    return number


def show_camera(mirror=False):
    model = load_model()
    if is_ip_camera:
        cap = cv2.VideoCapture(ip_camera_url)
    else:
        cap = cv2.VideoCapture(0)

    history_count = 0
    history_img = np.zeros((history_height, history_width))
    offset_y = 20
    offset_x = 0

    while True:
        answers = []
        ret, img = cap.read()

        for i in range(0, len(boxes_x_coord)):
            current_x = boxes_x_coord[i]
            contour, gray_image = get_img_contour_thresh_1(img, current_x)

            if len(contour) > 0:
                contour = max(contour, key=cv2.contourArea)
                if cv2.contourArea(contour) > 500:
                    image = gray_image
                    image = cv2.resize(image, IMG_SIZE)
                    image = np.array(image)
                    image = image.astype('float32')
                    # newImage_1 /= 255

                    if K.image_data_format() == 'channels_first':
                        image = image.reshape(1, IMG_SIZE[0], IMG_SIZE[1])
                    else:
                        image = image.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)
                        image = np.expand_dims(image, axis=0)

                    answer = model.predict(image).argmax()
                    answers.append(answer)
            else:
                answers.append(10)

        for i, answer in enumerate(answers):
            current_x = boxes_x_coord[i]
            add_answer_on_frame(answer, current_x, img)

        number = calculate_number(answers)
        cv2.putText(img, str(number), (450, 630), cv2.FONT_HERSHEY_SIMPLEX, 4, red_rgb_tuple, 5)
        cv2.imshow("Frame", img)
        #cv2.imshow("History", history_img)
        if cv2.waitKey(33) == 32:
            print(number)
            history_count = history_count + 1
            y_coordinate = offset_y * history_count
            if y_coordinate >= history_height:
                offset_x += 90
                history_count = 1
                if offset_x >= history_width:
                    break
            cv2.putText(history_img, str(number), (offset_x, offset_y * history_count), cv2.QT_FONT_NORMAL,
                        0.7, white_rgb_tuple, 2)


def main():
    show_camera(mirror=True)


if __name__ == '__main__':
    main()

