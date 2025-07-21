import streamlit as st
from PIL import Image
import io
import numpy as np
import time

# Assuming you're using Ultralytics for YOLOv8
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

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
        # Status message for user
        with st.spinner(f"Downloading model from Hugging Face Hub: {model_id}/{filename}"):
            model_path = hf_hub_download(repo_id=model_id, filename=filename)
            model = YOLO(model_path)
            st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub. Please check the model ID and file name. Error: {e}")
        st.stop() # Stop the app if model loading fails

# Load the model once
model = load_yolov8_model(YOLOV8_MODEL_ID, MODEL_FILENAME)
class_names = model.names # Get class names from the model (e.g., 'crop', 'weed')


# --- Streamlit UI ---
st.set_page_config(page_title="Weed Detection with YOLOv8", layout="wide", icon="🌿")

# Custom CSS for styling (similar to your Tailwind ideas)
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc; /* bg-gray-50 */
}
.stTabs [data-baseweb="tab-list"] button {
    background-color: #fff;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    color: #4f46e5; /* indigo-600 */
    border: 1px solid #e2e8f0; /* gray-200 */
    transition: all 0.3s ease;
}
.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #eef2ff; /* indigo-50 */
    border-color: #4f46e5;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #4f46e5; /* indigo-600 */
    color: white;
    border-color: #4f46e5;
}
.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
}

/* Header styling */
h1 {
    color: #4338ca; /* indigo-700 */
    font-size: 2.5rem; /* 4xl */
    font-weight: 700; /* bold */
    text-align: center;
    margin-bottom: 0.5rem;
}
.stMarkdown p {
    color: #4b5563; /* gray-600 */
    text-align: center;
    margin-bottom: 2.5rem;
}

/* Card-like containers for sections */
.stContainer {
    background-color: white;
    border-radius: 0.75rem; /* rounded-xl */
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-md */
    padding: 1.5rem; /* p-6 */
    margin-bottom: 2rem;
}

/* Button styling */
.stButton>button {
    background-color: #4f46e5; /* indigo-600 */
    color: white;
    padding: 0.75rem 1.5rem; /* px-4 py-2 */
    border-radius: 0.5rem; /* rounded-lg */
    font-weight: 500; /* font-medium */
    transition: background-color 0.3s ease;
    border: none;
    width: 100%; /* runModelBtn was w-full */
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem; /* space-x-2 */
}
.stButton>button:hover:enabled {
    background-color: #3730a3; /* indigo-700 */
}
.stButton>button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

/* Sliders */
.stSlider label {
    font-weight: 500;
    color: #374151; /* gray-700 */
}

/* Image placeholders */
.image-placeholder {
    background-color: #f3f4f6; /* gray-100 */
    border-radius: 0.5rem; /* rounded-lg */
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    padding: 2.5rem; /* py-10 */
    color: #9ca3af; /* gray-400 */
    text-align: center;
    min-height: 200px; /* To give it some height */
}
.image-placeholder i {
    font-size: 3rem; /* 4xl / 5xl */
    margin-bottom: 1rem;
}

/* Statistics cards */
.metric-card {
    background-color: white;
    border-radius: 0.5rem;
    padding: 0.75rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
}
.metric-label {
    font-size: 0.875rem; /* text-sm */
    color: #6b7280; /* gray-500 */
}
.metric-value {
    font-size: 1.5rem; /* text-2xl */
    font-weight: 700; /* bold */
    color: #4f46e5; /* indigo-600 */
}

/* Table styling */
.stDataFrame table, .stTable table {
    min-width: 100%;
    background-color: white;
    border-radius: 0.5rem;
    overflow: hidden; /* For rounded corners */
}
.stDataFrame th, .stTable th {
    background-color: #f9fafb; /* gray-50 */
    padding: 0.5rem 1rem;
    text-align: left;
    font-size: 0.75rem; /* text-xs */
    font-weight: 500;
    color: #6b7280; /* gray-500 */
    text-transform: uppercase;
    border-bottom: 1px solid #e5e7eb; /* gray-200 */
}
.stDataFrame td, .stTable td {
    padding: 1rem;
    font-size: 0.875rem; /* text-sm */
    color: #374151; /* gray-800 */
    border-bottom: 1px solid #e5e7eb; /* gray-200 */
}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)


st.header("Weed Detection with YOLOv8", divider="gray")
st.markdown("Upload images or use your camera with our fine-tuned YOLOv8 weed detection model")

# Create two columns for input and results
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 class='text-2xl font-semibold text-gray-800 mb-6'>Input Options</h2>", unsafe_allow_html=True)

    # Use tabs for Upload Image and Camera
    tab1, tab2 = st.tabs(["<i class='fas fa-file-upload'></i> Upload Image", "<i class='fas fa-camera'></i> Use Camera"])

    uploaded_file = None
    with tab1:
        st.markdown("<h3 class='text-lg font-medium text-gray-700 mb-3'>Upload Image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],
                                           help="Drag & drop your image here or browse files.")

    camera_image = None
    with tab2:
        st.markdown("<h3 class='text-lg font-medium text-gray-700 mb-3'>Use Camera</h3>", unsafe_allow_html=True)
        # Streamlit's camera_input handles preview and capture
        camera_image = st.camera_input("Take a picture")

    st.markdown("<h3 class='text-lg font-medium text-gray-700 mb-3'>Model Settings</h3>", unsafe_allow_html=True)

    # Sliders for Confidence and IOU Threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONF_THRESHOLD,
        step=0.05,
        format="%.2f",
        help="Adjust the minimum confidence score for displaying detections."
    )
    st.markdown(f"<div class='flex justify-between text-xs text-gray-500'><span>0%</span><span id='confidenceValue'>{int(confidence*100)}%</span><span>100%</span></div>", unsafe_allow_html=True)

    iou = st.slider(
        "IOU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_IOU_THRESHOLD,
        step=0.05,
        format="%.2f",
        help="Adjust the Intersection Over Union (IOU) threshold for Non-Maximum Suppression (NMS)."
    )
    st.markdown(f"<div class='flex justify-between text-xs text-gray-500'><span>0%</span><span id='iouValue'>{int(iou*100)}%</span><span>100%</span></div>", unsafe_allow_html=True)


    # Determine which image to process
    selected_image = None
    if uploaded_file is not None:
        selected_image = uploaded_file
    elif camera_image is not None:
        selected_image = camera_image

    # Run Model Button
    run_model_button = st.button("🚀 Detect Weeds",
                                disabled=selected_image is None,
                                use_container_width=True)


with col2:
    st.markdown("<h2 class='text-2xl font-semibold text-gray-800 mb-6'>Detection Results</h2>", unsafe_allow_html=True)

    # Placeholders for images
    original_image_placeholder = st.empty()
    processed_image_placeholder = st.empty()
    
    # Placeholders for stats and details
    stats_placeholder = st.empty()
    details_placeholder = st.empty()

    if selected_image is None:
        original_image_placeholder.markdown("""
        <div class="image-placeholder">
            <i class="fas fa-image"></i>
            <p>No image selected</p>
        </div>
        """, unsafe_allow_html=True)
        processed_image_placeholder.markdown("""
        <div class="image-placeholder">
            <i class="fas fa-search"></i>
            <p>Run the model to see detections</p>
        </div>
        """, unsafe_allow_html=True)
        stats_placeholder.markdown("""
        <div class="bg-indigo-50 rounded-lg p-4 mb-6">
            <h3 class="text-lg font-medium text-indigo-700 mb-3">Detection Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
                <div class="metric-card">
                    <div class="metric-label">Objects Detected</div>
                    <div class="metric-value">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Inference Time</div>
                    <div class="metric-value">0 ms</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        details_placeholder.markdown("""
        <div>
            <h3 class="text-lg font-medium text-gray-700 mb-3">Detection Details</h3>
            <table class="min-w-full bg-white rounded-lg overflow-hidden">
                <thead class="bg-gray-50">
                    <tr>
                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Class</th>
                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                        <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Box Coordinates</th>
                    </tr>
                </thead>
                <tbody id="detectionDetails" class="divide-y divide-gray-200">
                    <tr>
                        <td colspan="3" class="px-4 py-4 text-center text-sm text-gray-500">No detections yet</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Display the selected original image
        original_image = Image.open(selected_image)
        original_image_placeholder.image(original_image, caption='Original Image', use_column_width=True)

        if run_model_button:
            # Convert PIL Image to numpy array for YOLOv8
            img_array = np.array(original_image)

            with st.spinner("Detecting objects..."):
                start_time = time.time()
                # Perform inference with user-defined thresholds
                results = model(img_array, conf=confidence, iou=iou)
                inference_time_ms = (time.time() - start_time) * 1000

            detection_made = False
            all_detections_data = []

            for r in results:
                im_bgr = r.plot()  # plot a BGR numpy array of predictions
                im_rgb = Image.fromarray(im_bgr[..., ::-1]) # Convert BGR to RGB for Streamlit
                processed_image_placeholder.image(im_rgb, caption='Detection Results', use_column_width=True)

                if r.boxes:
                    detection_made = True
                    for box in r.boxes:
                        conf_score = box.conf[0].item()
                        cls_id = int(box.cls[0].item())
                        name = class_names[cls_id]
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                        
                        all_detections_data.append({
                            "Class": name.capitalize(),
                            "Confidence": f"{conf_score:.2f}",
                            "Box Coordinates": f"[{x1}, {y1}, {x2}, {y2}]"
                        })
            
            # Update stats
            stats_placeholder.markdown(f"""
            <div class="bg-indigo-50 rounded-lg p-4 mb-6">
                <h3 class="text-lg font-medium text-indigo-700 mb-3">Detection Statistics</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div class="metric-card">
                        <div class="metric-label">Objects Detected</div>
                        <div class="metric-value">{len(all_detections_data)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Inference Time</div>
                        <div class="metric-value">{inference_time_ms:.2f} ms</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Update detection details table
            if all_detections_data:
                import pandas as pd
                df_detections = pd.DataFrame(all_detections_data)
                details_placeholder.dataframe(df_detections, use_container_width=True, hide_index=True)
            else:
                details_placeholder.markdown("""
                <div>
                    <h3 class="text-lg font-medium text-gray-700 mb-3">Detection Details</h3>
                    <table class="min-w-full bg-white rounded-lg overflow-hidden">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Class</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                                <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Box Coordinates</th>
                            </tr>
                        </thead>
                        <tbody id="detectionDetails" class="divide-y divide-gray-200">
                            <tr>
                                <td colspan="3" class="px-4 py-4 text-center text-sm text-gray-500">No detections found with current thresholds.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Initial state or after file/camera input but before running model
            processed_image_placeholder.markdown("""
            <div class="image-placeholder">
                <i class="fas fa-search"></i>
                <p>Run the model to see detections</p>
            </div>
            """, unsafe_allow_html=True)
            stats_placeholder.markdown("""
            <div class="bg-indigo-50 rounded-lg p-4 mb-6">
                <h3 class="text-lg font-medium text-indigo-700 mb-3">Detection Statistics</h3>
                <div class="grid grid-cols-2 gap-4">
                    <div class="metric-card">
                        <div class="metric-label">Objects Detected</div>
                        <div class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Inference Time</div>
                        <div class="metric-value">0 ms</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            details_placeholder.markdown("""
            <div>
                <h3 class="text-lg font-medium text-gray-700 mb-3">Detection Details</h3>
                <table class="min-w-full bg-white rounded-lg overflow-hidden">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Class</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                            <th class="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Box Coordinates</th>
                        </tr>
                    </thead>
                    <tbody id="detectionDetails" class="divide-y divide-gray-200">
                        <tr>
                            <td colspan="3" class="px-4 py-4 text-center text-sm text-gray-500">No detections yet</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
---
<p class="text-gray-500 text-sm text-center">
*Disclaimer: This app is for demonstration purposes only and may not be 100% accurate for all scenarios. The model detects 'crop' and 'weed' categories.*
</p>
""", unsafe_allow_html=True)
