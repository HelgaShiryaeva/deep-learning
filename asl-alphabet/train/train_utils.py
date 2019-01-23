from keras.models import model_from_json


def save_model(model, json_filename, weights_filename):
    model_json = model.to_json()
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_filename)
    print("Saved model to disk")


def load_model_from_json(json_filename, weights_filename):
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_filename)
    print("Loaded model from disk")

    return model







