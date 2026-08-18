import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("model.keras")

img = cv2.imread("test.jpg")

if img is None:
    print("❌ Image not found")
    exit()

img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

pred = model.predict(img)[0][0]

print("Prediction value:", pred)

if pred > 0.5:
    print("🟥 PNEUMONIA")
else:
    print("🟩 NORMAL")