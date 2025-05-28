import streamlit as st
import io
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_train_test_loaders
from utils.model import CustomVGG

# Constants
MODEL_PATH = os.path.join("weights", "leather_model.h5")
DATA_PATH = os.path.join("data", "leather")
OVERVIEW_IMAGE_PATH = os.path.join("docs", "overview_dataset.jpg")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="AnomalyInspection", page_icon=":camera:")

st.title("AnomalyInspection")
st.caption("Boost Your Quality Control with AnomalyInspection - The Ultimate AI-Powered Inspection App")
st.write("Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly.")

# Sidebar
with st.sidebar:
    try:
        if os.path.exists(OVERVIEW_IMAGE_PATH):
            img = Image.open(OVERVIEW_IMAGE_PATH)
            st.image(img, caption="Dataset Overview")
        else:
            st.warning("Dataset overview image not found")
    except Exception as e:
        st.error(f"Error loading overview image: {str(e)}")
    
    st.subheader("About AnomalyInspection")
    st.write("AnomalyInspection is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With AnomalyInspection, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency.")
    st.write("This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. AnomalyInspection is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Leather Product Images.")

# Define the functions to load images
def load_uploaded_image(file):
    try:
        img = Image.open(file).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

# Initialize image variables
uploaded_file_img = None
camera_file_img = None

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        if uploaded_file_img:
            st.image(uploaded_file_img, caption="Uploaded Image", width=300)
            st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        if camera_file_img:
            st.image(camera_file_img, caption="Camera Input Image", width=300)
            st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

@st.cache_resource
def load_model_and_classes():
    """Load the model and class names with caching"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}")
            return None, None
            
        if not os.path.exists(DATA_PATH):
            st.error(f"Data directory not found at {DATA_PATH}")
            return None, None
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(MODEL_PATH, map_location=device)
        model.eval()
        
        _, test_loader = get_train_test_loaders(DATA_PATH, batch_size=1)
        return model, test_loader.dataset.classes
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def detect_anomaly(image):
    """Detect anomalies in the given image"""
    try:
        model, class_names = load_model_and_classes()
        if model is None or class_names is None:
            return "Error: Model not properly loaded"
            
        # Preprocess the image
        input_tensor = preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.sigmoid(output).squeeze().numpy()
        
        if probs.ndim == 0:
            probs = np.array([probs])
            
        # Get prediction and confidence
        predicted_idx = int(np.argmax(probs))
        predicted_label = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        # Format the prediction message
        if predicted_label == "Good":
            return f"✅ Product classified as 'Good' with {confidence:.2%} confidence."
        else:
            return f"⚠️ Anomaly detected with {confidence:.2%} confidence."
            
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return "Error during inference"

# Prediction trigger
submit = st.button(label="Submit a Leather Product Image")
if submit:
    st.subheader("Inspection Result")
    if input_method == "File Uploader" and uploaded_file_img is not None:
        image_to_process = uploaded_file_img
    elif input_method == "Camera Input" and camera_file_img is not None:
        image_to_process = camera_file_img
    else:
        st.warning("Please upload or capture an image first.")
        image_to_process = None
        
    if image_to_process is not None:
        with st.spinner("Analyzing image..."):
            result = detect_anomaly(image_to_process)
            st.success(result)
