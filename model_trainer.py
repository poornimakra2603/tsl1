import numpy as np
import os
from PIL import Image
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

def load_dataset(dataset_path):
    """Load images and labels from dataset directory"""
    images = []
    labels = []
    
    # Get all category folders
    categories = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    categories.sort()  # Ensure consistent ordering
    
    st.write(f"Found categories: {categories}")
    
    total_images = 0
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        category_images = 0
        
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(category_path, filename)
                    img = Image.open(img_path)
                    
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image for better feature extraction
                    img = img.resize((64, 64))
                    
                    # Keep as RGB for better gesture recognition
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Convert to array and flatten for traditional ML
                    img_array = np.array(img).flatten()
                    
                    # Normalize pixel values
                    img_array = img_array / 255.0
                    
                    images.append(img_array)
                    labels.append(category)
                    category_images += 1
                    total_images += 1
                    
                except Exception as e:
                    st.warning(f"Error loading image {filename}: {e}")
        
        st.write(f"Loaded {category_images} images from {category}")
    
    st.write(f"Total images loaded: {total_images}")
    
    return np.array(images), np.array(labels), categories

def create_model(num_classes):
    """Create a Random Forest model for Tamil sign language recognition"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    return model

def train_model(dataset_path):
    """Train the Tamil sign language recognition model"""
    
    # Load dataset
    st.write("Loading dataset...")
    images, labels, categories = load_dataset(dataset_path)
    
    if len(images) == 0:
        raise ValueError("No images found in the dataset!")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    
    st.write(f"Training set: {len(X_train)} images")
    st.write(f"Testing set: {len(X_test)} images")
    
    # Create model
    num_classes = len(categories)
    model = create_model(num_classes)
    
    # Train model
    st.write("Training model...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Training in progress...")
    progress_bar.progress(0.5)
    
    model.fit(X_train, y_train)
    
    progress_bar.progress(1.0)
    status_text.text("Training completed!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Show classification report
    report = classification_report(y_test, y_pred, target_names=categories)
    st.text("Classification Report:")
    st.code(report)
    
    # Save model and label encoder
    with open('tsl_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    st.write("Model and label encoder saved successfully!")
    
    return None, test_accuracy

def evaluate_model():
    """Evaluate the trained model"""
    if not os.path.exists('tsl_model.pkl'):
        st.error("Model not found! Please train the model first.")
        return
    
    with open('tsl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    st.write("Model loaded successfully!")
    st.write(f"Number of classes: {len(label_encoder.classes_)}")
    st.write(f"Classes: {list(label_encoder.classes_)}")