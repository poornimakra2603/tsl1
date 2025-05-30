import numpy as np
from PIL import Image
import pickle
import streamlit as st

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match training size
        image = image.resize((64, 64))
        
        # Convert to array and flatten for scikit-learn
        img_array = np.array(image).flatten()
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        return img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def load_model(model_path):
    """Load trained model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def validate_image(image):
    """Validate uploaded image"""
    try:
        # Check if image is valid
        if image is None:
            return False, "No image provided"
        
        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image too small (minimum 50x50 pixels)"
        
        # Check image mode
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False, "Unsupported image format"
        
        return True, "Image is valid"
    
    except Exception as e:
        return False, f"Error validating image: {e}"

def get_image_stats(image):
    """Get statistics about the image"""
    try:
        stats = {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
            'width': image.width,
            'height': image.height
        }
        
        # Get pixel statistics for RGB images
        if image.mode == 'RGB':
            img_array = np.array(image)
            stats['mean_rgb'] = np.mean(img_array, axis=(0, 1))
            stats['std_rgb'] = np.std(img_array, axis=(0, 1))
        
        return stats
    
    except Exception as e:
        st.error(f"Error getting image statistics: {e}")
        return {}

def format_confidence(confidence):
    """Format confidence score for display"""
    return f"{confidence:.1%}"

def create_confidence_color(confidence):
    """Create color based on confidence level"""
    if confidence >= 0.8:
        return "green"
    elif confidence >= 0.6:
        return "orange"
    else:
        return "red"
