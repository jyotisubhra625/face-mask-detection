import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load Model
model = load_model("mask_detection_model.h5")

st.title("Face Mask Detection App")
st.write("Upload an image and click **Predict** to see the result.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display image preview
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        img_array = np.array(image)

        # Preprocess
        img_resized = cv2.resize(img_array, (128, 128))
        img_scaled = img_resized / 255.0
        img_reshaped = np.reshape(img_scaled, (1, 128, 128, 3))

        # Prediction
        prediction = model.predict(img_reshaped)[0][0]

        # Output
        if prediction > 0.5:
            st.error("🚫 The person is NOT wearing a mask")
        else:
            st.success("😷 The person IS wearing a mask")
