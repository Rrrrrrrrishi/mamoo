import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('my_model.keras')

# Custom CSS for the pink color theme and styling
st.markdown("""
    <style>
    body {
        background-color: #ffe6f0;  /* Light pink background */
    }
    .css-18e3th9 {
        background-color: #ffccda !important;  /* Button background */
    }
    h1 {
        color: #e60073;  /* Dark pink title color */
        text-align: center;
        font-size: 3rem;
    }
    .stButton > button {
        background-color: #ff66b2;  /* Button color */
        color: white;
        font-size: 1.5rem;
        border-radius: 10px;
        width: 100%;
    }
    .stFileUploader {
        background-color: #ffccda;  /* Uploader background */
        border-radius: 15px;
        padding: 10px;
    }
    .stFileUploader label {
        color: #e60073;  /* Label color */
        font-size: 1.5rem;
    }
    .stButton > button:hover {
        background-color: #ff4d94;  /* Button hover color */
    }
    .result-text {
        color: #cc0066;  /* Result text color */
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the web app
st.title("MAMOGRAM")

# File uploader for breast scan images
uploaded_file = st.file_uploader("Upload your breast scan image", type=["jpg", "png", "jpeg"])

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (100, 100))  # Resize to match model input size
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to validate if the uploaded image resembles a breast scan
def is_valid_breast_scan(image):
    # Basic validation to check if the image is in an acceptable size range
    width, height = image.size
    return width >= 50 and height >= 50  # Adjust these numbers based on your needs

# If an image is uploaded, show it and provide a button to predict
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Check if the uploaded image is a valid breast scan
    if is_valid_breast_scan(image):
        # "PREDICT CANCER" button
        if st.button("PREDICT CANCER"):
            # Preprocess the image and make prediction
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = 'MALIGNANT' if prediction[0][0] > 0.5 else 'BENIGN'

            # Show the result in bold text
            st.markdown(f"<div class='result-text'>{predicted_class}</div>", unsafe_allow_html=True)
    else:
        # Display a message if the uploaded image is not a breast scan
        st.markdown("<div class='result-text'>THIS IMAGE IS NOT A BREAST SCAN</div>", unsafe_allow_html=True)
