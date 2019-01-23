import cv2
import numpy as np

IMAGE_PATH = r"C:\Users\helga_sh\Desktop\CNNpresentation\test1.jpg"
IMAGE_PATH_GRAY = r"gray1.jpg"
IMAGE_SIZE = (680, 440)
OFFSET = 7
black_rgb_tuple = (0, 0, 0)
green_rgb_tuple = (0, 255, 0)
red_rgb_tuple = (0, 0, 255)
white_rgb_tuple = (255, 255, 255)



def contour_comparator(c):
    return cv2.boundingRect(c)[0]



lower_blue = np.array([90, 150, 0])
upper_blue = np.array([255, 255, 255])


font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, IMAGE_SIZE)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

kernelOpen = np.ones((4, 4))
kernelClose = np.ones((4, 4))

maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)


maskFinal = maskClose
_, contours, hierarchy = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

contours = sorted(contours, key=contour_comparator)
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(img, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), green_rgb_tuple, 2)
    cv2.putText(img, str(i+1), (x - 2*OFFSET, y + h + 2*OFFSET), font, 1, black_rgb_tuple, 2)

cv2.imshow("maskClose", maskClose)
cv2.imshow("maskOpen", maskOpen)
cv2.imshow("mask", mask)
cv2.imshow("cam", img)


k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()


