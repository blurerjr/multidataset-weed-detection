import streamlit as st
from PIL import Image
import io
import numpy as np
import time

# Move st.set_page_config() to the very top
st.set_page_config(page_title="Weed Detection with YOLOv8", layout="wide", icon="🌿")

# Ensure necessary libraries are available
try:
    import cv2 # headless should be installed via requirements.txt
except ImportError:
    st.error("OpenCV is not installed. Please ensure 'opencv-python-headless' is in your requirements.txt.")
    st.stop()

# --- Configuration ---
# Your specific Hugging Face model ID and filename
YOLOV8_MODEL_ID = "blurerjr/yolov8_cd"
MODEL_FILENAME = "best.pt"

# Initial values for sliders (matching your HTML defaults)
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5

# --- Model Loading ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_yolov8_model(model_id: str, filename: str):
    """Loads a YOLOv8 model from Hugging Face Hub."""
    try:
        # No Streamlit widgets directly inside this cached function.
        # Messages about loading should be displayed OUTSIDE this function call.
        model_path = hf_hub_download(repo_id=model_id, filename=filename)
        model = YOLO(model_path)
        return model
    except Exception as e:
        # If an error happens here, the app would likely already be stopped by a preceding st.error
        raise e # Re-raise the error so it can be caught and displayed by Streamlit

# Display model loading messages before calling the cached function
with st.spinner(f"Downloading model from Hugging Face Hub: {YOLOV8_MODEL_ID}/{MODEL_FILENAME}"):
    try:
        model = load_yolov8_model(YOLOV8_MODEL_ID, MODEL_FILENAME)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


class_names = model.names # Get class names from the model (e.g., 'crop', 'weed')


# --- Streamlit UI ---
# Your custom CSS and markdown should also come after set_page_config
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc; /* bg-gray-50 */
}
# ... (rest of your CSS and UI code) ...
""", unsafe_allow_html=True)

st.header("Weed Detection with YOLOv8", divider="gray")
st.markdown("Upload images or use your camera with our fine-tuned YOLOv8 weed detection model")

# ... (rest of your Streamlit app code) ...
