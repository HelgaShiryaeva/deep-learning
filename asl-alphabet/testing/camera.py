import cv2
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam

is_ip_camera = False

y_coord, width, height = 30, 400, 400
black_rgb_tuple = (0, 0, 0)
green_rgb_tuple = (0, 255, 0)
red_rgb_tuple = (0, 0, 255)
IMG_SIZE = 64

current_x = 0




def load_model():
    json_file = open(r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\classification\train\3_conv_adam_cat_cross_v2\adam\learning_rate\1e-4\3_conv_adam_1e-4_cat_cross_v2.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\classification\train\3_conv_adam_cat_cross_v2\adam\learning_rate\1e-4\3_conv_adam_1e-4_cat_cross_v2.h5")
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    return model


def decode_prediction(prediction):
    if prediction == 0:
        return 'A'
    elif prediction == 1:
         return 'B'
    elif prediction == 2:
        return 'C'
    elif prediction == 3:
        return 'D'
    elif prediction == 4:
        return 'E'


def add_answer_on_frame(answer, current_x, img):
    decoded_answer = decode_prediction(answer)
    cv2.putText(img, str(decoded_answer), (current_x + 70, y_coord + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, black_rgb_tuple, 2)


def prepare_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    image = img_to_array(img)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)

    return image


def show_camera():
    model = load_model()
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        image = img.copy()
        answer = model.predict(prepare_image(image)).argmax()
        add_answer_on_frame(answer, current_x, img)

        cv2.imshow("Frame", img)

        k = cv2.waitKey(33)
        if k == 32:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break


def main():
    show_camera()


if __name__ == '__main__':
    main()
