import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, delimiter=',')
    input_data = data.drop(data.columns[0], axis=1).values
    output_data = data.iloc[:, 0].values

    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)

    label_encoder = LabelEncoder()
    output_data = label_encoder.fit_transform(output_data)

    return input_data, output_data

X_train, y_train = load_and_preprocess_data('prepared_snake_snake.csv')
X_test, y_test = load_and_preprocess_data('prepared_snake_snake_test.csv')

# Define model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Save model
model.save('snake_model.keras')
