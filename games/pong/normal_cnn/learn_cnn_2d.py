import os
import numpy as np
from PIL import Image
from keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# folder z obrazami
images_dir = 'images/'

# lista plikow obrazow
image_files = os.listdir(images_dir)

# inicjalizacja list do przechowywania obrazów i etykiet
images = []
labels = []

# wczytanie obrazow i etykiet
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    try:
        image = Image.open(image_path)
        # print(f"Wczytano obraz: {image_file}")
        image = image.resize((96, 64))
        image = np.array(image)
        images.append(image)

        labels.append(0)

    except Exception as e:
        print(f"Błąd wczytywania obrazu {image_file}: {e}")

images = np.array(images)
labels = np.array(labels)

# podział danych na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# normalizacja obrazów (przeskalowanie wartości pikseli do zakresu 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# budowa modelu CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 96, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.save('output/pong_model_normal_cnn_2d.keras')