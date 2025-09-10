from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = "skinlesionmodel.h5"
model = load_model(MODEL_PATH)

# Define your class labels (change this according to your dataset)
CLASS_NAMES = [
    "Melanocytic nevi", 
    "Melanoma", 
    "Benign keratosis-like lesions", 
    "Basal cell carcinoma", 
    "Actinic keratoses", 
    "Vascular lesions", 
    "Dermatofibroma"
]

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        # Preprocess the image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))  # ResNet input size
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = round(float(np.max(preds)) * 100, 2)

        return jsonify({
            "class": CLASS_NAMES[predicted_class],
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

