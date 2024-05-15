from tensorflow import keras

# Alternatywne importy
load_model = keras.models.load_model
image = keras.preprocessing.image

import numpy as np

# Załaduj model
model = load_model('output/pong_model_normal_cnn_2d.keras')

# Załaduj obraz
img = image.load_img('images/pong1018.png', target_size=(144, 256))

# Przekształć obraz w tablicę numpy
img_array = image.img_to_array(img)

# Dodaj dodatkowy wymiar, aby stworzyć pojedynczy batch
img_batch = np.expand_dims(img_array, axis=0)

# Normalizuj obraz (jeśli to było robione podczas treningu)
img_preprocessed = img_batch / 255.0

# Użyj modelu do przewidzenia klasy obrazu
predictions = model.predict(img_preprocessed)

# Wybierz klasę z największym prawdopodobieństwem
predicted_class = np.argmax(predictions[0])

predicted_class = (predicted_class - 1)

print("Przewidywana klasa to:", predicted_class)
