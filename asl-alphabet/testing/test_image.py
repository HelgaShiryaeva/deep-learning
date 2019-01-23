from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
import matplotlib.pyplot as plt
is_ip_camera = False

y_coord, width, height = 30, 400, 400
black_rgb_tuple = (0, 0, 0)
green_rgb_tuple = (0, 255, 0)
red_rgb_tuple = (0, 0, 255)
IMG_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
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


def predict(filename, model):
    img = load_img(filename, target_size=IMG_SHAPE)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    #img = preprocess_input(img)

    prediction = model.predict(img)

    return decode_prediction(prediction.argmax())


def plot_images(test_folder_path, row, col):
    fig = plt.figure()
    model = load_model()
    for num, img in enumerate(os.listdir(test_folder_path)):
        path = os.path.join(test_folder_path, img)
        img_label_true = img[0]
        label = predict(path, model)
        img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        y = fig.add_subplot(row, col, num + 1)
        y.imshow(img)
        plt.title('T : {}, N : {}'.format(img_label_true, label))
        y.get_xaxis().set_visible(False)
        y.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    plot_images(r"C:\Users\helga_sh\PycharmProjects\asl-alphabet\classification\test", 5, 5)
