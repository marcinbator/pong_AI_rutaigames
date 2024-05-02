from keras.src.saving.saving_lib import load_model
from sklearn.preprocessing import MinMaxScaler

saved_model = load_model('games/pong/pong_model.keras')

import numpy as np


def predict_result(input_values):
    input_values = np.array(input_values)
    normalized_values = []

    with open("games/pong/min_max_values.txt", "r") as file:
        for v in input_values:
            min_val, max_val = map(float, file.readline().split(","))
            v = (2 * (v - min_val) / (max_val - min_val)) - 1
            normalized_values.append(round(v, 1))

    prediction = saved_model.predict(np.array([normalized_values]))
    print("prediction: ",prediction)
    predicted_value = np.argmax(prediction, axis=1)

    return -(predicted_value[0] - 1)

# predict_result([199.6,69.0,7.3,9.5,72])
