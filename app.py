import streamlit as st
from fastai.vision.all import *
import pathlib
import os
from src.config import EXPORT_PATH


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.set_page_config(page_title="Pneumonia AI", page_icon="🫁")

st.title("Pneumonia Detection AI")
st.markdown("""
This tool uses a **Convolutional Neural Network (ResNet34)** to analyze chest X-rays.
Upload an image below to check for signs of Pneumonia.
""")

@st.cache_resource
def load_model():
    if not os.path.exists(EXPORT_PATH):
        return None
    return load_learner(EXPORT_PATH)

learn = load_model()

if learn is None:
    st.error(f"Model not found at `{EXPORT_PATH}`.")
    st.info("Please run `python main.py` first to train and save the model.")
    st.stop()

# ==========================================
# PREDICTION LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = PILImage.create(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_column_width=True)
    
    if st.button('Analyze Image', type="primary"):
        with st.spinner('Analyzing patterns...'):
            # Get prediction
            pred_class, pred_idx, probs = learn.predict(img)
            confidence = probs[pred_idx] * 100
            
            # Display Result
            st.divider()
            if pred_class == 'PNEUMONIA':
                st.error(f"Result: PNEUMONIA DETECTED")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.warning("Recommendation: Please consult a radiologist immediately.")
            else:
                st.success(f"### Result: NORMAL")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write("No signs of pneumonia detected.")