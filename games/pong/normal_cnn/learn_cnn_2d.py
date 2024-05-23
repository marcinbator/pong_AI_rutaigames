import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
from keras import Sequential, Input
from keras.src.layers import AveragePooling2D, Dropout
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# folder z obrazami
images_dir = 'images/'

# lista plikow obrazow
image_files = sorted(os.listdir(images_dir))

name = "output/prepared_pong_pong_normal.csv"
data = pd.read_csv(name, delimiter=',')

# inicjalizacja list do przechowywania obrazów i etykiet
images = []
labels = data.iloc[:, 5].values

# liczba ostatnich obrazów do użycia
num_last_images = 5

# wczytanie obrazow i etykiet
for i in range(len(image_files) - num_last_images + 1):
    image_stack = []
    for j in range(num_last_images):
        image_path = os.path.join(images_dir, image_files[i + j])
        try:
            image = Image.open(image_path).convert('L')
            image = np.array(image)
            image_stack.append(image)
        except Exception as e:
            print(f"Błąd wczytywania obrazu {image_files[i + j]}: {e}")
    images.append(np.stack(image_stack, axis=-1))

images = np.array(images)
labels = labels[num_last_images - 1:]

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
    Input(shape=(144, 256, num_last_images)),
    Conv2D(64, (5, 5), (2, 2), 'same', activation='relu'),
    Conv2D(32, (5, 5), (2, 2), 'same', activation='relu'),
    Dropout(0.5),
    AveragePooling2D(2, 2),
    Conv2D(32, (5, 5), (2, 2), 'same', activation='relu'),
    Conv2D(16, (5, 5), (2, 2), 'same', activation='relu'),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=160, batch_size=32, validation_data=(X_test, y_test))

model.save('output/pong_model_normal_cnn_2d.keras')


# Predykcje
predictions = model.predict(X_test)

# Sortowanie według rzeczywistych wartości etykiet
sort_index = np.argsort(np.argmax(y_test, axis=1))
y_test_sorted = y_test[sort_index]
predictions_sorted = predictions[sort_index]

# Wykresy
plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(y_test_sorted, label='True')
    plt.plot(predictions_sorted[:, i], label=f'Neuron {i + 1}')
    plt.title(f'Predictions for Neuron {i + 1}')
    plt.xlabel('Index')
    plt.ylabel('Output')
    plt.legend()
plt.tight_layout()
plt.show()

# Zapis predykcji do pliku CSV
with open('output/predictions_normal_cnn.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['True_Value', 'Predicted_Neuron_1', 'Predicted_Neuron_2', 'Predicted_Neuron_3'])
    for true_val, pred_vals in zip(y_test_sorted, predictions_sorted):
        writer.writerow([true_val] + list(pred_vals))

# Ocena modelu
test_loss, test_acc = model.evaluate(X_test, y_test)

predictions = np.argmax(model.predict(X_test), axis=1)
correct_predictions = np.sum(predictions == np.argmax(y_test, axis=1))
classification_accuracy = correct_predictions / len(y_test)

print('\nClassification accuracy:', classification_accuracy)

# Wykresy dokładności i straty modelu
fig, axs = plt.subplots(2)
fig.suptitle('Model Accuracy and Loss')
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_ylabel('Accuracy')
axs[0].legend(['Train', 'Validation'], loc='upper left')

axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Obliczanie dokładności klasyfikacji dla danych trenujących
train_predictions = np.argmax(model.predict(X_train), axis=1)
train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))

print('Train accuracy:', train_accuracy)
