import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
MODEL_PATH = "brain_tumor_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------------------------------------------
# Class Names
# ---------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ---------------------------------------------------
# Predict Function
# ---------------------------------------------------
def predict(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    class_index = np.argmax(pred)
    confidence = pred[class_index] * 100

    return CLASS_NAMES[class_index], confidence

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image below to predict the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing MRI image..."):
            label, conf = predict(image)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf:.2f}%")
