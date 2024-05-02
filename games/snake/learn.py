import keras
import pandas as pd
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
    keras.layers.Flatten(input_shape=(400,)),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(32, activation='tanh'),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=300, batch_size=8, verbose=1, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Save model
model.save('snake_model.keras')
