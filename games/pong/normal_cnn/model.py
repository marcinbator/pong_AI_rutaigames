import numpy as np
from tensorflow import keras
load_model = keras.models.load_model
image = keras.preprocessing.image
from PIL import ImageGrab

saved_model = load_model('games/pong/normal_cnn/output/pong_model_normal_cnn_2d.keras')

def predict_result(bbox):
    # Przechwytuje obraz z ekranu o podanych koordynatach
    img = ImageGrab.grab(bbox)

    # Przekształć obraz w odpowiedni rozmiar
    img = img.resize((256, 144)).convert('LA')

    # Wyświetl przechwycony obraz
    # img.show()

    # Przekształć obraz w tablicę numpy
    img_array = image.img_to_array(img)

    # Dodaj dodatkowy wymiar, aby stworzyć pojedynczy batch
    img_batch = np.expand_dims(img_array, axis=0)

    # Normalizuj obraz (jeśli to było robione podczas treningu)
    img_preprocessed = img_batch / 255.0

    # Użyj modelu do przewidzenia klasy obrazu
    predictions = saved_model.predict(img_preprocessed)

    # Wybierz klasę z największym prawdopodobieństwem
    predicted_class = np.argmax(predictions[0])

    predicted_class = (predicted_class - 1)

    print(predicted_class)

    return predicted_class

# predict_result((447, 340, 1453, 972))
