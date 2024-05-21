import os
import numpy as np
import pandas as pd
from PIL import Image
from keras import Sequential, Input
from keras.src.layers import AveragePooling2D
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# folder z obrazami
images_dir = 'images/'

# lista plikow obrazow
image_files = os.listdir(images_dir)

name = "output/prepared_pong_pong_normal.csv"
data = pd.read_csv(name, delimiter=',')

# inicjalizacja list do przechowywania obrazów i etykiet
images = []
labels = data.iloc[:, 5].values

# wczytanie obrazow i etykiet
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    try:
        image = Image.open(image_path).convert('L')
        # print(f"Wczytano obraz: {image_file}")
        image = np.array(image)
        images.append(image)

    except Exception as e:
        print(f"Błąd wczytywania obrazu {image_file}: {e}")

images = np.array(images)
labels = np.array(labels)

# podział danych na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# normalizacja obrazów (przeskalowanie wartości pikseli do zakresu 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = y_train + 1
y_test = y_test + 1

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# budowa modelu CNN
model = Sequential([
    Input(shape=(144, 256, 1)),
    Conv2D(64, (5, 5), (1, 1), 'same', activation='relu'),
    Conv2D(64, (5, 5), (2, 2), 'same', activation='relu'),
    AveragePooling2D(2, 2),
    Conv2D(64, (5, 5), (1, 1), 'same', activation='relu'),
    Conv2D(64, (5, 5), (2, 2), 'same', activation='relu'),
    AveragePooling2D(2, 2),
    # Conv2D(64, (5, 5), (1, 1), 'same', activation='relu'),
    # Conv2D(32, (5, 5), (3, 2), 'same', activation='relu'),
    Flatten(),
    # Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=32, batch_size=32, validation_data=(X_test, y_test))

model.save('output/pong_model_normal_cnn_2d.keras')