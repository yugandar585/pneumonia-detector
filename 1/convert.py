import tensorflow as tf
import tensorflowjs as tfjs
import os

# Fix for Keras 3 compatibility (important for newer TensorFlow)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

print("Loading your model... This may take a moment.")

model = tf.keras.models.load_model("model.keras")

print("Converting to TensorFlow.js format...")

# This creates the tfjs_model folder
tfjs.converters.save_keras_model(model, "tfjs_model")

print("✅ SUCCESS! The folder 'tfjs_model' has been created.")
print("It contains model.json and weight files.")
print("You can now use it in the web app.")