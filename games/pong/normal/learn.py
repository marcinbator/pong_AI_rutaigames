import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
name = "output/prepared_pong_pong_normal.csv"
data = pd.read_csv(name, delimiter=',')

input_data = data.iloc[:, :-1].values
output_data = data.iloc[:, 5].values

input_data_normalized = []
with open("output/min_max_values_normal.txt", "w") as file:
    for column in range(input_data.shape[1]):
        column_data = input_data[:, column]
        min_val = column_data.min()
        max_val = column_data.max()
        file.write(f"{min_val}, {max_val}\n")
        normalized_column_data = (2 * (column_data - min_val) / (max_val - min_val)) - 1
        input_data_normalized.append(normalized_column_data.tolist())


input_data_normalized = np.array(input_data_normalized).T


X_train, X_test, y_train, y_test = train_test_split(input_data_normalized, output_data, test_size=0.2, random_state=42)
y_train = y_train + 1
y_test = y_test + 1


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5,)),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dense(8, activation='tanh'),
    keras.layers.Dense(6, activation='tanh'),
    keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=500, batch_size=1000, validation_split=0.2)

model.save('output/pong_model_normal.keras')

########
predictions = model.predict(X_test)

sort_index = np.argsort(y_test)
y_test_sorted = y_test[sort_index]
predictions_sorted = predictions[sort_index]

plt.figure(figsize=(10, 8))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(y_test_sorted, label='True')
    plt.plot(predictions_sorted[:, i], label=f'Neuron {i+1}')
    plt.title(f'Predictions for Neuron {i+1}')
    plt.xlabel('Index')
    plt.ylabel('Output')
    plt.legend()
plt.tight_layout()
plt.show()

with open('output/predictions_normal.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['True_Value', 'Predicted_Neuron_1', 'Predicted_Neuron_2', 'Predicted_Neuron_3'])
    for true_val, pred_vals in zip(y_test_sorted, predictions_sorted):
        writer.writerow([true_val] + list(pred_vals))

########

test_loss, test_acc = model.evaluate(X_test, y_test)

predictions = np.argmax(model.predict(X_test), axis=1)
correct_predictions = np.sum(predictions == y_test)
classification_accuracy = correct_predictions / len(y_test)

print('\nClassification accuracy:', classification_accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
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

print('Train accuracy:', train_accuracy)


###showing stats

# Wczytanie danych z pliku CSV
df_normalized = pd.read_csv(name, header=None)
df_normalized.columns = ["x", "y", "vel_x", "vel_y", "posY", "move"]

# Normalizacja danych
def normalize_data(df, normalization_type):
    for column in df.columns:
        if normalization_type == "min_max":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (2 * (df[column] - min_val) / (max_val - min_val)) - 1
        elif normalization_type == "zero_one":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

# Kopie danych do normalizacji dla różnych typów normalizacji
df_min_max = df_normalized.copy()
df_zero_one = df_normalized.copy()

# Normalizacja danych
df_min_max = normalize_data(df_min_max, "min_max")
df_zero_one = normalize_data(df_zero_one, "zero_one")

# Wykresy pudełkowe dla każdej kolumny po znormalizowaniu
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
df_min_max.boxplot()
plt.title("Wykresy pudełkowe po znormalizowaniu (min_max)")

plt.subplot(2, 2, 2)
df_zero_one.boxplot()
plt.title("Wykresy pudełkowe po znormalizowaniu (zero_one)")

plt.tight_layout()
plt.show()

# Wykresy liczby wystąpień wartości dla każdej kolumny po znormalizowaniu
for column in df_min_max.columns:
    plt.figure(figsize=(8, 6))
    df_min_max[column].hist(bins=20, alpha=0.7, color='blue')
    plt.title(f"Rozkład wartości dla kolumny {column} po znormalizowaniu (min_max)")
    plt.xlabel("Znormalizowane wartości")
    plt.ylabel("Liczba wystąpień")
    plt.show()

for column in df_zero_one.columns:
    plt.figure(figsize=(8, 6))
    df_zero_one[column].hist(bins=20, alpha=0.7, color='green')
    plt.title(f"Rozkład wartości dla kolumny {column} po znormalizowaniu (zero_one)")
    plt.xlabel("Znormalizowane wartości")
    plt.ylabel("Liczba wystąpień")
    plt.show()
