from tensorflow import keras

model = keras.models.load_model('output/pong_model_normal_cnn_2d.keras')

# Zapisanie modelu w formacie .h5
model.save('output/pong_model_normal_cnn_2d.h5')
