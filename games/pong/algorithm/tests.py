import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm


def train_model(layer_config, learning_rate, data_normalization):
    input_data_normalized = []
    data = pd.read_csv('output/prepared_pong_pong_algorithm.csv', delimiter=',')
    input_data = data.iloc[:, :-1].values
    output_data = data.iloc[:, 5].values

    if data_normalization == 'min_max':
        input_data_normalized = (input_data - input_data.min(axis=0)) / (
                input_data.max(axis=0) - input_data.min(axis=0))
        activation = 'tanh'
    elif data_normalization == 'zero_one':
        input_data_normalized = (input_data - input_data.mean(axis=0)) / input_data.std(axis=0)
        activation = 'relu'

    X_train, X_test, y_train, y_test = train_test_split(input_data_normalized, output_data, test_size=0.2,
                                                        random_state=42)
    y_train = (y_train + 1)/2
    y_test = (y_test + 1)/2

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(5,)))

    for units in layer_config:
        model.add(keras.layers.Dense(units, activation=activation))

    model.add(keras.layers.Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=1000, batch_size=500, validation_split=0.2)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    predictions = np.argmax(model.predict(X_test), axis=1)
    correct_predictions = np.sum(predictions == y_test)
    classification_accuracy = correct_predictions / len(y_test)

    return test_acc, test_loss, classification_accuracy


layer_configs = [(8, 8), (12, 8), (16, 8), (10, 8, 6), (12, 8, 6), (16, 8, 6)]
learning_rates = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
normalization_methods = ['min_max', 'zero_one']

total_combinations = len(layer_configs) * len(learning_rates) * len(normalization_methods)
progress_bar = tqdm(total=total_combinations, desc='Training Models')

with open("output/results_algorithm.csv", "w") as file:
    file.write("architecture,lr,normalization,accuracy,loss,classification_accuracy\n")
for layer_config in layer_configs:
    for learning_rate in learning_rates:
        for normalization_method in normalization_methods:
            acc, loss, classification_accuracy = train_model(layer_config, learning_rate, normalization_method)
            progress_bar.update(1)
            with open("output/results_algorithm.csv", "a") as file:
                file.write(f'"{";".join(str(x) for x in layer_config)}", {learning_rate}, {normalization_method},{acc}, {loss}, {classification_accuracy}\n')

progress_bar.close()


