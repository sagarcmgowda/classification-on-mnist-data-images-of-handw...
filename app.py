import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("MNIST Digit Recognizer")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img.reshape(1, 784).astype("float32") / 255.0

    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(uploaded_file, caption="Uploaded Image", width=150)
    st.success(f"Prediction: **{digit}**")
    st.write(f"Confidence: **{confidence:.2%}**")
