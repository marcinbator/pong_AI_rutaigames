import numpy as np
from keras.src.saving.saving_lib import load_model

saved_model = load_model('games/pong/normal/output/pong_model_normal.keras')


def predict_result(input_values):
    input_values = np.array(input_values)
    normalized_values = []

    with open("games/pong/normal/output/min_max_values_normal.txt", "r") as file:
        for v in input_values:
            v = round(v, 0)
            min_val, max_val = map(float, file.readline().split(","))
            v = (2 * (v - min_val) / (max_val - min_val)) - 1
            normalized_values.append(v)

    print("normalized:", normalized_values)
    prediction = saved_model.predict(np.array([normalized_values]))
    print("prediction: ", prediction)
    predicted_value = np.argmax(prediction, axis=1)

    return -(predicted_value[0] - 1)

# predict_result([199.6,69.0,7.3,9.5,72])
