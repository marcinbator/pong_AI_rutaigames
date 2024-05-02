import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras

data = pd.read_csv('prepared_pong_pong.csv', delimiter=',')

input_data = data.iloc[:, :-1].values
output_data = data.iloc[:, 5].values

input_data_normalized = []
with open("min_max_values.txt", "w") as file:
    for column in range(input_data.shape[1]):
        column_data = input_data[:, column]
        min_val = column_data.min()
        max_val = column_data.max()
        file.write(f"{min_val}, {max_val}\n")
        normalized_column_data = (column_data - min_val) / (max_val - min_val)
        input_data_normalized.append(normalized_column_data.tolist())

input_data_normalized = np.array(input_data_normalized).T


X_train, X_test, y_train, y_test = train_test_split(input_data_normalized, output_data, test_size=0.2, random_state=42)
y_train = y_train + 1
y_test = y_test + 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(5,)),
    keras.layers.Dense(16, activation='relu', ),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer=Adam(learning_rate=0.005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

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

model.save('pong_model.keras')
