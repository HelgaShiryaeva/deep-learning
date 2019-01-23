import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model


def plot_images(labeled_images, row, col, IMG_SIZE):
    fig = plt.figure()
    for num, data in enumerate(labeled_images):
        img_data = data[0]
        label = data[1]
        if label[0] == 1:
            label = 'A'
        else:
            label = 'Delete'
        y = fig.add_subplot(row, col, num + 1)
        y.imshow(np.reshape(img_data, (IMG_SIZE, IMG_SIZE)), cmap='gray')
        plt.title(label)
        y.get_xaxis().set_visible(False)
        y.get_yaxis().set_visible(False)
    plt.show()


def plot_model_to_png(model, filename):
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)


def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


