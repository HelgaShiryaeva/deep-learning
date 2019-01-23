import cv2
import numpy as np
from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import RMSprop


class BlueDetector:

    def __init__(self, img_size):
        self.IMAGE_PATH = r"C:\Users\helga_sh\Desktop\CNN presentation\test.jpg"
        self.IMAGE_PATH_GRAY = r"gray1.jpg"
        self.IMAGE_SIZE = img_size
        self.OFFSET = 8
        self.black_rgb_tuple = (0, 0, 0)
        self.green_rgb_tuple = (0, 255, 0)
        self.red_rgb_tuple = (0, 0, 255)
        self.white_rgb_tuple = (255, 255, 255)

        self.blue = np.uint8([[[150, 120, 120]]])

    def get_bounding_rects(self, img):
        lower_blue = np.array([0, 75, 35])
        upper_blue = np.array([255, 255, 255])

        font = cv2.FONT_HERSHEY_SIMPLEX

        img = cv2.resize(img, self.IMAGE_SIZE)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

        kernelOpen = np.ones((4, 4))
        kernelClose = np.ones((6, 6))

        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose
        _, contours, hierarchy = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

        contours = sorted(contours, key=self.contour_comparator)
        #for i in range(len(contours)):
            #x, y, w, h = cv2.boundingRect(contours[i])
            #cv2.rectangle(img, (x - self.OFFSET, y - self.OFFSET), (x + w + self.OFFSET, y + h + self.OFFSET), self.green_rgb_tuple, 2)
            #cv2.putText(img, str(i + 1), (x - 2 * self.OFFSET, y + h + 2 * self.OFFSET), font, 1, self.black_rgb_tuple, 2)

        cv2.imshow("maskClose", maskClose)
        cv2.imshow("maskOpen", maskOpen)
        cv2.imshow("mask", mask)
        cv2.imshow("cam", img)

        #k = cv2.waitKey(0)
        #if k == 27:  # wait for ESC key to exit
            #cv2.destroyAllWindows()

        return contours

    @staticmethod
    def contour_comparator(c):
        return cv2.boundingRect(c)[0]

is_ip_camera = False
ip_camera_url = 'http://admin:Password@192.168.115.135:8080/stream/video/mjpeg'

black_rgb_tuple = (0, 0, 0)
green_rgb_tuple = (0, 255, 0)
red_rgb_tuple = (0, 0, 255)
white_rgb_tuple = (255, 255, 255)
IMG_SIZE = 64
FRAME_IMG_SIZE = (680, 440)
DIGITS_MAX_NUM = 5
OFFSET = 8


def load_model():
    json_file = open(r'C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\5_conv_adam_1e-5_cat_cross_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\digit-recognition\5_conv_adam_1e-5_cat_cross_v2.h5")
    print("Loaded model from disk")
    print()
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    return model


def calculate_dot_number(answers):
    number = 0.0
    max_power = len(answers) - 2
    for i, digit in enumerate(answers):
               number += digit * 10 ** (max_power - i)

    return number


def prepare_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(img)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image * 1./255

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
        img = cv2.resize(img, FRAME_IMG_SIZE)
        img_color = img.copy()
        rectangles = detector.get_bounding_rects(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        number_size = len(rectangles)

        for i in range(0, number_size):
            bound = rectangles[i]
            x, y, w, h = cv2.boundingRect(bound)

            image = img[y: y + h, x: x + w]
            if len(image) > 30:
                image = prepare_image(image)
                answer = model.predict(image).argmax()
                answers.append(answer)
                cv2.rectangle(img_color, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), green_rgb_tuple, 2)

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

