import tensorflow as tf
from tensorflow import keras

# Załaduj swój model Keras
model = keras.models.load_model('output/pong_model_normal_cnn_2d.keras')

# Zapisz model w formacie SavedModel
tf.saved_model.save(model, 'output/pong_model_normal_cnn_2d')
