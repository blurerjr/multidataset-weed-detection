import streamlit as st
from PIL import Image
import io
import torch
import numpy as np

# Assuming you're using Ultralytics for YOLOv8
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --- Model Loading ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_yolov8_model(model_id: str, filename: str = "best.pt"):
    """Loads a YOLOv8 model from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(repo_id=model_id, filename=filename)
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.stop() # Stop the app if model loading fails

# Your specific Hugging Face model ID and filename
YOLOV8_MODEL_ID = "Rohankumar31/Yolo-weed-detection"
MODEL_FILENAME = "best.pt"

# Load the model
model = load_yolov8_model(YOLOV8_MODEL_ID, MODEL_FILENAME)
#blurerjr/yolov8_cd
# Get class names from the model
class_names = model.names
# The model summary shows two classes: 'crop' and 'weed'
# You can verify this by printing model.names locally

# --- Streamlit UI ---
st.set_page_config(page_title="YOLOv8 Weed Detection", layout="wide")
st.title("🌿 YOLOv8 Weed Detection App")
st.write("Upload an image to detect crops and weeds using your custom YOLOv8 model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting crops and weeds...")

    # Convert PIL Image to a format YOLOv8 can use (e.g., numpy array)
    img_array = np.array(image)

    # Perform inference
    # You might want to adjust confidence threshold (conf) or IoU threshold (iou)
    # based on your model's performance.
    results = model(img_array, conf=0.25, iou=0.45) # Example thresholds

    # Process and display results
    # The 'results' object from Ultralytics YOLO has various methods for visualization
    detection_made = False
    for r in results:
        # Plotting results directly on the image
        im_bgr = r.plot()  # plot a BGR numpy array of predictions
        im_rgb = Image.fromarray(im_bgr[..., ::-1]) # Convert BGR to RGB for Streamlit
        st.image(im_rgb, caption='Detection Results', use_column_width=True)

        # Display text details of detections
        if r.boxes:
            detection_made = True
            st.subheader("Detected Objects:")
            for box in r.boxes:
                conf = box.conf[0].item() # Confidence score
                cls = int(box.cls[0].item()) # Class ID
                name = class_names[cls] # Class name (e.g., 'crop' or 'weed')
                
                # Get bounding box coordinates for potential further use
                # x1, y1, x2, y2 = box.xyxy[0].tolist() 
                
                st.write(f"- **{name.capitalize()}**: Confidence: {conf:.2f}")
    
    if not detection_made:
        st.write("No crops or weeds detected with the current confidence threshold.")


else:
    st.info("Please upload an image to get started with weed detection.")

st.markdown("""
---
*Disclaimer: This app is for demonstration purposes only and may not be 100% accurate for all scenarios. The model detects 'crop' and 'weed' categories.*
""")
