import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Wczytanie danych
name = "output/prepared_pong_pong_normal.csv"
data = pd.read_csv(name, delimiter=',')

# Normalizacja danych
data_normalization = 'min_max'
activation = ''

input_data = data.iloc[:, :-1].values
output_data = data.iloc[:, 5].values

input_data_normalized = []
if data_normalization == 'min_max':
    with open("output/min_max_values_normal.txt", "w") as file:
        for column in range(input_data.shape[1]):
            column_data = input_data[:, column]
            min_val = column_data.min()
            max_val = column_data.max()
            file.write(f"{min_val}, {max_val}\n")
            normalized_column_data = (2 * (column_data - min_val) / (max_val - min_val)) - 1
            input_data_normalized.append(normalized_column_data.tolist())
    activation = 'tanh'
elif data_normalization == 'zero_one':
    with open("output/min_max_values_normal.txt", "w") as file:
        for column in range(input_data.shape[1]):
            column_data = input_data[:, column]
            min_val = column_data.min()
            max_val = column_data.max()
            file.write(f"{min_val}, {max_val}\n")
            normalized_column_data = (column_data - min_val) / (max_val - min_val)
            input_data_normalized.append(normalized_column_data.tolist())
    activation = 'relu'

input_data_normalized = np.array(input_data_normalized).T

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(input_data_normalized, output_data, test_size=0.2, random_state=42)
y_train = y_train + 1
y_test = y_test + 1

# Definicja modelu
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5,)),
    keras.layers.Dense(32, activation=activation),
    keras.layers.Dense(32, activation=activation),
    keras.layers.Dense(24, activation=activation),
    keras.layers.Dense(16, activation=activation),
    keras.layers.Dense(3, activation='softmax'),
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=6000, batch_size=1000, validation_split=0.2)

# Zapis modelu
model.save('output/pong_model_normal.keras')

# Predykcje
predictions = model.predict(X_test)

sort_index = np.argsort(y_test)
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
with open('output/predictions_normal.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['True_Value', 'Predicted_Neuron_1', 'Predicted_Neuron_2', 'Predicted_Neuron_3'])
    for true_val, pred_vals in zip(y_test_sorted, predictions_sorted):
        writer.writerow([true_val] + list(pred_vals))

# Ocena modelu
test_loss, test_acc = model.evaluate(X_test, y_test)

predictions = np.argmax(model.predict(X_test), axis=1)
correct_predictions = np.sum(predictions == y_test)
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
train_accuracy = np.mean(train_predictions == y_train)

print('Train accuracy:', train_accuracy)
