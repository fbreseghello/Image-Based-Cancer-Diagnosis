"""
Enhanced Streamlit application for cancer diagnosis using histopathology images.
Features include Grad-CAM visualization, confidence metrics, and detailed model info.
"""

import streamlit as st
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

from model_utils import (
    load_image,
    load_model,
    predict_with_confidence,
    get_sample_image_files,
    generate_gradcam,
    overlay_heatmap,
    get_model_info,
    ModelLoadError,
    ImageProcessingError
)
from config import CLASS_NAMES, THRESHOLD

# Page configuration
st.set_page_config(
    page_title="Cancer Diagnosis AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.title("🔬 Navigation Panel")
    
    st.markdown("""
    ### How to use:
    1. **Upload** your image or use a sample
    2. View the **prediction** and confidence
    3. Check **Grad-CAM** visualization for interpretability
    """)
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_confidence = st.checkbox("Show Confidence Metrics", value=True)
    show_model_info = st.checkbox("Show Model Information", value=False)
    
    st.divider()
    
    # Training history
    if st.checkbox("📊 Show Training History"):
        try:
            with open("treinamento_history.json") as f:
                hist = json.load(f)
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy
            ax[0].plot(hist['accuracy'], label='Training', linewidth=2, marker='o')
            ax[0].plot(hist['val_accuracy'], label='Validation', linewidth=2, marker='s')
            ax[0].set_title('Model Accuracy', fontweight='bold')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Accuracy')
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            
            # Loss
            ax[1].plot(hist['loss'], label='Training', linewidth=2, marker='o')
            ax[1].plot(hist['val_loss'], label='Validation', linewidth=2, marker='s')
            ax[1].set_title('Model Loss', fontweight='bold')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Loss')
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best metrics
            best_acc = max(hist['val_accuracy'])
            best_epoch = hist['val_accuracy'].index(best_acc) + 1
            st.success(f"Best Validation Accuracy: **{best_acc:.4f}** (Epoch {best_epoch})")
            
        except FileNotFoundError:
            st.warning("Training history not found. Train the model first.")
        except Exception as e:
            st.error(f"Error loading training history: {e}")


# --- Header ---
st.markdown('<div class="main-header">🔬 AI-Powered Cancer Diagnosis</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2rem;'>
    Predict whether tumor tissue samples are <b>benign</b> or <b>malignant (cancerous)</b> 
    using deep learning analysis of histopathology images.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load model and samples ---
@st.cache_resource
def cached_load_model():
    """Cache model loading for performance."""
    try:
        return load_model()
    except ModelLoadError as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = cached_load_model()

# Model information
if show_model_info:
    with st.expander("📋 Model Information"):
        model_info = get_model_info(model)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{model_info.get('total_params', 0):,}")
        with col2:
            st.metric("Trainable Parameters", f"{model_info.get('trainable_params', 0):,}")
        with col3:
            st.metric("Total Layers", model_info.get('layers', 0))
        
        st.json(model_info)

# Load sample images
try:
    sample_images = get_sample_image_files()
except Exception as e:
    st.warning(f"Could not load sample images: {e}")
    sample_images = {}

# --- Main Interface ---
tab1, tab2 = st.tabs(["📤 Upload Image", "🖼️ Sample Images"])

def display_prediction(img_array, prob, predicted_class, confidence):
    """Display prediction results with visualizations."""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image((img_array * 255).astype("uint8"), caption="Input Image", use_container_width=True)
    
    with col2:
        # Prediction result
        if predicted_class == "benign":
            st.success(f"### ✅ Prediction: BENIGN")
            st.markdown(f"**Probability:** {prob:.4f}")
        else:
            st.error(f"### ⚠️ Prediction: MALIGNANT")
            st.markdown(f"**Probability:** {prob:.4f}")
        
        # Confidence metrics
        if show_confidence:
            st.markdown("---")
            st.markdown("### 📊 Confidence Metrics")
            
            # Progress bar
            st.progress(confidence / 100)
            st.markdown(f"**Confidence Level:** {confidence:.2f}%")
            
            # Classification threshold
            threshold_diff = abs(prob - THRESHOLD)
            st.markdown(f"**Distance from Threshold ({THRESHOLD}):** {threshold_diff:.4f}")
            
            # Interpretation
            if confidence > 90:
                st.info("🔵 Very High Confidence")
            elif confidence > 75:
                st.info("🟢 High Confidence")
            elif confidence > 60:
                st.warning("🟡 Moderate Confidence")
            else:
                st.warning("🟠 Low Confidence - Consider additional tests")
        
        # Grad-CAM visualization
        if show_gradcam:
            with st.spinner("Generating Grad-CAM..."):
                try:
                    heatmap = generate_gradcam(model, img_array)
                    
                    if heatmap is not None:
                        st.markdown("---")
                        st.markdown("### 🔥 Grad-CAM Visualization")
                        st.caption("Highlights regions the model focused on for prediction")
                        
                        overlayed = overlay_heatmap(img_array, heatmap)
                        st.image(overlayed, caption="Grad-CAM Overlay", use_container_width=True)
                    else:
                        st.warning("Grad-CAM not available for this model")
                except Exception as e:
                    st.error(f"Grad-CAM error: {e}")


# Tab 1: Upload Image
with tab1:
    st.markdown("### Upload a histopathology image for analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a histopathology image in JPG, PNG, or BMP format"
    )
    
    if uploaded_file:
        try:
            img_array = load_image(uploaded_file, resize=True)
            predicted_class, prob, confidence = predict_with_confidence(model, img_array)
            
            st.divider()
            display_prediction(img_array, prob, predicted_class, confidence)
            
        except ImageProcessingError as e:
            st.error(f"❌ Error processing image: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
    else:
        st.info("👆 Please upload an image to begin analysis")


# Tab 2: Sample Images
with tab2:
    st.markdown("### Test with sample images from the dataset")
    
    if not sample_images:
        st.warning("No sample images found. Please add images to the sample_images directory.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_class = st.selectbox(
                "Select class:",
                options=list(sample_images.keys()),
                help="Choose between benign or malignant samples"
            )
        
        with col2:
            if st.button("🎲 Show Random Sample", use_container_width=True):
                st.session_state['show_sample'] = True
        
        if st.session_state.get('show_sample', False):
            try:
                image_list = sample_images[selected_class]
                random_idx = random.randint(0, len(image_list) - 1)
                img_path = image_list[random_idx]
                
                img_array = load_image(img_path, resize=True)
                predicted_class, prob, confidence = predict_with_confidence(model, img_array)
                
                st.divider()
                st.markdown(f"**True Label:** {selected_class.upper()}")
                st.markdown(f"**Sample:** {img_path.name}")
                
                # Check if prediction matches true label
                is_correct = (
                    (selected_class == "benign" and predicted_class == "benign") or
                    (selected_class == "malignant" and predicted_class == "malignant")
                )
                
                if is_correct:
                    st.success("✅ Correct Prediction!")
                else:
                    st.error("❌ Incorrect Prediction")
                
                display_prediction(img_array, prob, predicted_class, confidence)
                
            except Exception as e:
                st.error(f"❌ Error loading sample: {e}")


# --- Footer Warning ---
st.divider()
st.markdown(
    """
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p>
        <strong>This is an unvalidated AI project for educational and research purposes only.</strong><br>
        Do not use these results for actual medical diagnosis. Always consult a qualified healthcare 
        professional for proper medical evaluation and treatment.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Made with ❤️ using Streamlit and TensorFlow</div>",
    unsafe_allow_html=True
)

