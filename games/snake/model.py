from keras.src.saving import load_model

saved_model = load_model('games/snake/snake_model.keras')

import numpy as np


def predict_result(input_values):
    input_values = np.array(input_values) / 20.0
    prediction = saved_model.predict(np.array([input_values]))
    predicted_value = np.argmax(prediction,
                                axis=1) + 1
    print(predicted_value[0])

    return predicted_value[0]
