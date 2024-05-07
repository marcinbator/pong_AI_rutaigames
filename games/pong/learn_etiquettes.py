import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

name = "etiquettes_prepared_pong_pong.csv"

data = pd.read_csv(name, delimiter=',')

input_data = data.iloc[:, :-1].values
output_data = data.iloc[:, -1].values

input_data_normalized = (input_data - np.min(input_data, axis=0)) / (np.max(input_data, axis=0) - np.min(input_data, axis=0)) * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(input_data_normalized, output_data, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5,)),
    keras.layers.Dense(10, activation='tanh'),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(6, activation='tanh'),
    keras.layers.Dense(3),
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

model.save('etiquettes_pong_model.keras')

predictions = model.predict(X_test)

# Wykresy z odpowiedziami sieci dla danych testowych
plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(y_test, label='True')
    plt.plot(predictions[:, i], label=f'Neuron {i+1}')
    plt.title(f'Predictions for Neuron {i+1}')
    plt.xlabel('Index')
    plt.ylabel('Output')
    plt.legend()
plt.tight_layout()
plt.show()

# Zapis predykcji do pliku CSV
with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['True_Value', 'Predicted_Neuron_1', 'Predicted_Neuron_2', 'Predicted_Neuron_3'])
    for true_val, pred_vals in zip(y_test, predictions):
        writer.writerow([true_val] + list(pred_vals))

# Ocena modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

# Wykresy dokładności modelu
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Wykresy straty modelu
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Obliczanie dokładności klasyfikacji dla danych trenujących
train_predictions = np.argmax(model.predict(X_train), axis=1)
train_accuracy = np.mean(train_predictions == y_train)

# Obliczanie dokładności klasyfikacji dla danych testowych
val_predictions = np.argmax(model.predict(X_test), axis=1)
val_accuracy = np.mean(val_predictions == y_test)

print('Train accuracy:', train_accuracy)
print('Validation accuracy:', val_accuracy)

# Wykresy dokładności klasyfikacji
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(np.arange(len(history.history['val_accuracy'])), [train_accuracy]*len(history.history['val_accuracy']), linestyle='--')
plt.plot(np.arange(len(history.history['val_accuracy'])), [val_accuracy]*len(history.history['val_accuracy']), linestyle='--')
plt.title('Model classification accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation', 'Train Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()
