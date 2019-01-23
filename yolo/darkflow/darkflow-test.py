from darkflow.net.build import TFNet
import cv2
import numpy as np


def calculate_dot_number(answers):
    number = 0.0
    max_power = len(answers) - 2
    for i, digit in enumerate(answers):
               number += digit * 10 ** (max_power - i)

    return number


def compare_key(s):
    return s['topleft']['x']


def boxing(original_img, predictions):
    newImage = np.copy(original_img)
    answers = []
    predictions = sorted(predictions, key=compare_key)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        answer = int(result['label'])
        if answer != 10:
            answers.append(int(result['label']))

        if confidence > 0.6:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                   (0, 230, 0), 1, cv2.LINE_AA)
    number = calculate_dot_number(answers)
    newImage = cv2.putText(newImage, str(number), (220, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    print(answers)

    return newImage


options = {"model": "cfg/tiny-yolo-voc-11c.cfg",
           "load": 3375,
           "threshold": 0.1
           }

tfnet = TFNet(options)

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('./sample_video/output-demo.avi', fourcc, 20.0, (int(width), int(height)))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = np.asarray(frame)
        results = tfnet.return_predict(frame)

        new_frame = boxing(frame, results)

        # Display the resulting frame
        out.write(new_frame)
        cv2.imshow('frame', new_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
