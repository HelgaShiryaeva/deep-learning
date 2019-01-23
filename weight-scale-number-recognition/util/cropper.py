import cv2
import os
import random as random

if __name__ == '__main__':
    path = r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\hd-14349\data\original_data\04_10_18"

    for num, image in enumerate(os.listdir(path)):
        im = cv2.imread(os.path.join(path, image))
        im = cv2.resize(im, (800, 500))
        r = cv2.selectROI(im)
        imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv2.imwrite('0({}).jpg'.format(num * random.randint(500, 100000) + random.randint(0, 200) * random.randint(700, 900)), imCrop)

    cv2.waitKey(0)
