import tensorflow as tf
from tensorflowjs.converters import save_keras_model

# Load your trained model
model = tf.keras.models.load_model("model.keras")

# Convert and save to TensorFlow.js format
save_keras_model(model, "tfjs_model")

print("✅ Model converted to TensorFlow.js format successfully!")
print("Folder 'tfjs_model' created with model.json and weights")