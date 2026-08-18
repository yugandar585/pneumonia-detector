from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

model = tf.keras.models.load_model("model.keras")


@app.get("/")
def home():
    return {"message": "Pneumonia API running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # 🔥 FIXED preprocessing
    img = preprocess_input(img)

    pred = model.predict(img)[0][0]

    result = "NORMAL" if pred > 0.5 else "PNEUMONIA"

    return {
        "result": result,
        "confidence": float(pred)
    }