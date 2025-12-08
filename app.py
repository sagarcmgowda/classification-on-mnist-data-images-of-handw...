import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("mnist_model.h5")

st.title("Handwritten Digit Recognition")

file = st.file_uploader("Upload digit image", type=["png","jpg"])

if file:
    img = Image.open(file).convert("L").resize((28,28))
    img = np.array(img)/255.0
    img = img.reshape(1,784)

    pred = model.predict(img)
    st.write("Prediction:", np.argmax(pred))
