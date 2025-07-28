import streamlit as st
import numpy as np
import time
import math
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib  # For loading the label encoder

# Set up the Streamlit page
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🧬", layout="wide")

st.title("🧬 Skin Lesion Classification App")
st.write("Upload a skin lesion image and the model will classify its type using deep learning.")

# Sidebar developer info
st.sidebar.write("Developed by:\n**Subhasri P**")
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 [GitHub](https://github.com/subhasri-03)")
st.sidebar.markdown("📧 23am064@kpriet.ac.in")

# Load model and label encoder
model = load_model("skinlesionmodel.h5")
le = joblib.load("label_encoder.pkl")  # Save using: joblib.dump(le, 'label_encoder.pkl')

# File uploader
uploaded_file = st.file_uploader("📂 Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)

    img = image.resize((224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Classifying the lesion... Please wait"):
        time.sleep(2)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(predictions[0][predicted_class])
        class_name = le.classes_[predicted_class]

    st.success(f"### 🏷️ Predicted Lesion Type: **{class_name}**")
    st.write(f"Confidence Level: **{confidence * 100:.2f}%**")

    # Optional: Add basic guidance or interpretation
    st.info("📝 This is a deep learning-based prediction. Please consult a dermatologist for medical confirmation.")
