import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import requests
from PIL import Image
import os

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(url):
    """
    Downloads the YOLOv8 model from the provided URL and loads it.
    Uses Streamlit's caching to avoid re-downloading on every run.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the model to a temporary file
        model_path = "best.pt"
        with open(model_path, "wb") as f:
            f.write(response.content)
            
        model = YOLO(model_path)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the model: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- UI and Main Logic ---
st.set_page_config(page_title="Weed & Crop Detection", layout="wide")
st.title("🌿 Weed & Crop Detection using YOLOv8")
st.write("Upload an image or a video to detect weeds and crops. The YOLOv8 model is trained to identify these two classes.")

# URL to the raw model file on GitHub
MODEL_URL = "https://raw.githubusercontent.com/blurerjr/multidataset-weed-detection/master/best.pt"

# Load the model
with st.spinner("Downloading and loading the model..."):
    model = load_yolo_model(MODEL_URL)

if model is None:
    st.stop()

# Class names - assuming these are the classes the model was trained on.
# You can get the exact class names from the model's .yaml file if available.
CLASS_NAMES = ['crop', 'weed'] 
# Colors for bounding boxes (Crop: Green, Weed: Red)
COLORS = {
    'crop': (0, 255, 0),
    'weed': (0, 0, 255)
}

# Sidebar for options
st.sidebar.title("Options")
detection_mode = st.sidebar.radio("Select Detection Mode", ["Image", "Video"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# --- Image Detection Logic ---
if detection_mode == "Image":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Weeds & Crops"):
            with st.spinner("Processing image..."):
                # Perform detection
                results = model.predict(img_array, conf=confidence_threshold)
                
                annotated_img_array = img_array.copy()
                
                detection_count = {'crop': 0, 'weed': 0}

                # Draw bounding boxes on the image
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls_id = int(box.cls[0])
                        class_name = CLASS_NAMES[cls_id]
                        
                        detection_count[class_name] += 1
                        
                        label = f"{class_name} {conf:.2f}"
                        color = COLORS.get(class_name, (255, 255, 255)) # Default to white
                        
                        # Draw rectangle
                        cv2.rectangle(annotated_img_array, (x1, y1), (x2, y2), color, 2)
                        
                        # Put label
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_img_array, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_img_array, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                st.image(annotated_img_array, caption="Processed Image", use_column_width=True)
                st.success(f"Detection complete! Found {detection_count['crop']} crops and {detection_count['weed']} weeds.")

# --- Video Detection Logic ---
elif detection_mode == "Video":
    st.header("Video Detection")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection on the frame
                results = model.predict(frame, conf=confidence_threshold, verbose=False)
                
                annotated_frame = frame.copy()
                
                # Draw bounding boxes
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls_id = int(box.cls[0])
                        class_name = CLASS_NAMES[cls_id]
                        
                        label = f"{class_name} {conf:.2f}"
                        color = COLORS.get(class_name, (255, 255, 255))
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display the annotated frame
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
            
            cap.release()
            tfile.close()
            os.remove(video_path) # Clean up the temp file
            st.success("Video processing complete.")
