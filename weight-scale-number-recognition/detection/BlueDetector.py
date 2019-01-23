import numpy as np
import cv2


class BlueDetector:

    def __init__(self, img_size):
        self.IMAGE_PATH = r"C:\Users\helga_sh\Desktop\CNN presentation\test.jpg"
        self.IMAGE_PATH_GRAY = r"gray1.jpg"
        self.IMAGE_SIZE = img_size
        self.OFFSET = 10
        self.black_rgb_tuple = (0, 0, 0)
        self.green_rgb_tuple = (0, 255, 0)
        self.red_rgb_tuple = (0, 0, 255)
        self.white_rgb_tuple = (255, 255, 255)

        self.blue = np.uint8([[[150, 120, 120]]])

    def get_bounding_rects(self, img):
        lower_blue = np.array([20, 140, 60])
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
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(img, (x - self.OFFSET, y - self.OFFSET), (x + w + self.OFFSET, y + h + self.OFFSET), self.green_rgb_tuple, 2)
            cv2.putText(img, str(i + 1), (x - 2 * self.OFFSET, y + h + 2 * self.OFFSET), font, 1, self.black_rgb_tuple, 2)

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
