from keras.src.saving.saving_lib import load_model
from sklearn.preprocessing import MinMaxScaler

saved_model = load_model('games/pong/pong_model.keras')

import numpy as np


def predict_result(input_values):
    input_values = np.array(input_values)
    normalized_values = []

    with open("games/pong/min_max_values.txt", "r") as file:
        for v in input_values:
            min, max = file.readline().split(",")
            v = (v - float(min)) / (float(max) - float(min))
            normalized_values.append(round(v,1))

    print("normalized", normalized_values)

    prediction = saved_model.predict(np.array([normalized_values]))
    print(prediction)
    predicted_value = np.argmax(prediction, axis=1)

    return predicted_value[0] - 1

# predict_result([199.6,69.0,7.3,9.5,72])
