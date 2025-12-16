import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Best weights load karna
best_model1 = load_model("best_model1.h5")


# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .title {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        color: #00FFFF;
    }
    .subtitle {
        font-size: 35px;
        text-align: center;
        color: #00FFFF;
        margin-bottom: 20px;
    }
    hr {
        border: 1px solid #ccc;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- APP TITLE & SUBTITLE ---
st.markdown('<div class="title">Smart Mask Detector</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your image here and predict</div>', unsafe_allow_html=True)

file_upload = st.file_uploader("upload an image" , type = ["jpg" , "png","jpeg"],    label_visibility="collapsed")

if file_upload is not None:
    img = Image.open(file_upload)
    st.image(img , use_container_width=True)

def process_img(img):
    img = Image.open(file_upload)
    img = img.resize((224,224))
    img = np.array(img) / 225.0
    img = np.expand_dims(img , axis=0)
    return img

if st.button("Predict"):
    if file_upload is not None:
        image = process_img(file_upload)
        pred = best_model1.predict(image)
        if pred[0][0] > 0.5:
            st.success("With Mask")
        else:
            st.success("Without Mask")
    else:

        st.warning("Please upload an image first!")

