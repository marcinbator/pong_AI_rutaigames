import tensorflow as tf
import tf2onnx
import onnx

# Załaduj model h5
model = tf.keras.models.load_model('output/pong_model_normal_cnn_2d.h5')

# Konwertuj model do formatu ONNX
spec = (tf.TensorSpec((None,) + model.input_shape[1:], tf.float32, name="input"),)
output_path = 'output/pong_model_normal_cnn_2d_5images.onnx'

# Użyj from_function zamiast from_keras
model_proto, external_tensor_storage = tf2onnx.convert.from_function(
    tf.function(model.call), input_signature=spec, opset=13
)

# Zapisz model ONNX do pliku
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
