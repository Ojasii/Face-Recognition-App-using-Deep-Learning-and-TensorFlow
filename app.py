import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="Facial Recognition System", page_icon="👁️", layout="centered")

st.title("👁️ Facial Recognition and Verification System")
st.markdown("Upload an image to verify whether it matches an existing face in the dataset.")

model = siamese_model  # or whatever your model variable name is

def preprocess_image(image):
    image = image.resize((100, 100))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    anchor_path = st.text_input("Enter anchor image path (from data/anchor):", "data/anchor/example.jpg")

    if st.button("Run Verification"):
        if not os.path.exists(anchor_path):
            st.error("Anchor image not found. Please check the path.")
        else:
            anchor_img = Image.open(anchor_path)
            st.image(anchor_img, caption="Anchor Image", use_column_width=True)

            input_img = preprocess_image(image)
            anchor_input = preprocess_image(anchor_img)

            result = model.predict([anchor_input, input_img])

            if result < 0.5:
                st.success("✅ Match Found: Same Person")
            else:
                st.error("❌ No Match Found: Different Person")
