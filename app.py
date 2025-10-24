import os
import numpy as np
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === CONFIG ===
IMG_SIZE = 224
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === LOAD MODEL ===
MODEL_PATH = "skinlesionmodel.h5"
model = load_model(MODEL_PATH)

# === LOAD CLASS NAMES ===
with open("label1_classes.pkl", "rb") as f:
    class_names = pickle.load(f)  # This should be a list of class names

# === FLASK APP ===
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Make sure index.html exists in "templates" folder

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    # Save uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if model was trained with 0-1 scaling

    # Predict
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    confidence = float(np.max(preds) * 100)

    result = f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f}%)"

    return render_template("index.html", prediction=result, uploaded_image=file.filename)

if __name__ == "__main__":
    app.run(debug=False)


