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

# ---------------------------
# Input method selector
# ---------------------------
option = st.radio(
    "Select input method:",
    ("Upload Image", "Capture from Camera")
)

image = None

# ---------------------------
# Upload image
# ---------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

# ---------------------------
# Camera input
# ---------------------------
elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture handwritten digit")
    if camera_image:
        image = Image.open(camera_image)

# ---------------------------
# Prediction logic
# ---------------------------
if image:
    # Preprocess
    img = image.convert("L").resize((28, 28))
    img = np.array(img)

    # Invert colors (important for MNIST)
    img = 255 - img

    img = img.reshape(1, 784).astype("float32") / 255.0

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display
    st.image(image, caption="Input Image", width=200)
    st.success(f"Prediction: **{digit}**")
    st.write(f"Confidence: **{confidence:.2%}**")
