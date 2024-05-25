import numpy as np
from keras.src.utils import img_to_array
from tensorflow import keras
load_model = keras.models.load_model
image = keras.preprocessing.image
from PIL import ImageGrab

# Wczytanie zapisanego modelu
saved_model = load_model('games/pong/normal_cnn/output/pong_model_normal_cnn_2d.keras')

def predict_result(bbox):
    num_last_images = 7  # liczba ostatnich obrazów do użycia
    image_stack = []

    # Przechwytuje ostatnie X obrazów z ekranu o podanych koordynatach
    for _ in range(num_last_images):
        img = ImageGrab.grab(bbox)

        # Przekształć obraz w odpowiedni rozmiar
        img = img.resize((256, 144)).convert('L')

        # Przekształć obraz w tablicę numpy
        img_array = img_to_array(img)

        image_stack.append(img_array)

    # Stackowanie obrazów wzdłuż nowego wymiaru
    image_stack = np.stack(image_stack, axis=-1)

    # Usuń niepotrzebny wymiar i dodaj wymiar batch
    image_stack = np.squeeze(image_stack, axis=2)
    image_stack = np.expand_dims(image_stack, axis=0)

    # Normalizacja obrazów
    img_preprocessed = image_stack / 255.0

    # Użyj modelu do przewidzenia klasy obrazu
    predictions = saved_model.predict(img_preprocessed)

    # Wybierz klasę z największym prawdopodobieństwem
    predicted_class = np.argmax(predictions[0])

    predicted_class = predicted_class - 1  # przeskalowanie klasy, aby zaczynały się od -1, 0, 1

    print(predicted_class)

    return predicted_class

# predict_result((447, 340, 1453, 972))
