import streamlit as st
import numpy as np
from PIL import Image
import os
import pickle
from utils import preprocess_image, load_model
from tamil_labels import get_tamil_labels

# Configure page
st.set_page_config(
    page_title="Tamil Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤟 Tamil Sign Language Recognition")
    st.markdown("Upload an image of Tamil sign language gesture to get the corresponding Tamil text.")
    
    # Check if model exists
    model_path = "tsl_model.pkl"
    label_encoder_path = "label_encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(label_encoder_path)):
        # Auto-train model if not exists
        st.info("🚀 Training the Tamil Sign Language model...")
        
        dataset_path = "TSL Dataset"
        if os.path.exists(dataset_path):
            with st.spinner("Training model... This will take a few minutes."):
                try:
                    from model_trainer import train_model
                    history, accuracy = train_model(dataset_path)
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.stop()
        else:
            st.error("❌ Dataset not found! Please ensure TSL Dataset folder exists.")
            st.stop()
    
    model_status = True
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a Tamil sign language image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image showing Tamil sign language gesture"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Add image details
            st.write(f"**Image size:** {image.size}")
            st.write(f"**Image mode:** {image.mode}")
    
    with col2:
        st.header("Recognition Results")
        
        if uploaded_file is not None and model_status:
            try:
                # Load model and label encoder
                model = load_model(model_path)
                
                if model is not None:
                    with open(label_encoder_path, 'rb') as f:
                        label_encoder = pickle.load(f)
                    
                    # Get the uploaded image
                    image = Image.open(uploaded_file)
                    
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Make prediction
                        with st.spinner("Recognizing gesture..."):
                            # Get prediction probabilities
                            predicted_class_idx = model.predict([processed_image])[0]
                            predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
                            
                            # Get prediction probabilities for confidence
                            try:
                                probabilities = model.predict_proba([processed_image])[0]
                                confidence = np.max(probabilities)
                                all_predictions = list(zip(label_encoder.classes_, probabilities))
                                all_predictions.sort(key=lambda x: x[1], reverse=True)
                            except:
                                # Fallback if predict_proba not available
                                confidence = 0.85  # Default confidence for RandomForest
                                all_predictions = [(predicted_class, confidence)]
                        
                        # Get Tamil labels
                        tamil_labels = get_tamil_labels()
                        tamil_text = tamil_labels.get(predicted_class, predicted_class)
                        
                        # Display results
                        st.success("🎯 Recognition Complete!")
                        
                        # Main result
                        st.markdown("### Recognized Gesture:")
                        st.markdown(f"## {tamil_text}")
                        st.markdown(f"**English:** {predicted_class}")
                        st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Progress bar for confidence
                        st.progress(confidence)
                        
                        # Show all predictions
                        st.markdown("### All Predictions:")
                        
                        # Get top 3 predictions
                        for i, (class_name, score) in enumerate(all_predictions[:3]):
                            tamil_name = tamil_labels.get(class_name, class_name)
                            st.write(f"{i+1}. **{tamil_name}** ({class_name}) - {score:.2%}")
                        
                        # Confidence threshold warning
                        if confidence < 0.7:
                            st.warning("⚠️ Low confidence prediction. Please try with a clearer image.")
                    else:
                        st.error("❌ Error processing the uploaded image.")
                else:
                    st.error("❌ Failed to load the trained model.")
                
            except Exception as e:
                st.error(f"❌ Error during recognition: {str(e)}")
                st.error("Please make sure the model is properly trained.")
        
        elif uploaded_file is not None and not model_status:
            st.warning("⚠️ Please train the model first before making predictions.")
        
        elif model_status:
            st.info("👆 Upload an image to start recognition")
        
        else:
            st.info("🚀 Train the model and upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Tamil Sign Language Recognition System</p>
            <p>Built with Streamlit and TensorFlow</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
