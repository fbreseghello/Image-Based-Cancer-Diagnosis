"""
Enhanced utilities for model operations and image processing.
Includes error handling, validation, and evaluation metrics.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

from config import IMG_SIZE, SAMPLE_IMG_DIR, DEFAULT_MODEL_PATH, THRESHOLD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model fails to load."""
    pass


class ImageProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass


def load_image(image, resize: bool = False) -> np.ndarray:
    """
    Open and process image for model prediction.
    
    Args:
        image: File path, file-like object, or PIL Image
        resize: Whether to resize to model input size
    
    Returns:
        Processed image as numpy array
    
    Raises:
        ImageProcessingError: If image processing fails
    """
    try:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
        
        if resize:
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            return img_array
        
        return np.array(img)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ImageProcessingError(f"Failed to process image: {str(e)}")


def get_sample_image_files() -> Dict[str, List[Path]]:
    """
    Collect sample images organized by label.
    
    Returns:
        Dictionary mapping labels to lists of image paths
    
    Raises:
        FileNotFoundError: If sample directory doesn't exist
    """
    if not SAMPLE_IMG_DIR.exists():
        raise FileNotFoundError(f"Sample directory not found: {SAMPLE_IMG_DIR}")
    
    sample_images = {}
    
    for label_dir in SAMPLE_IMG_DIR.iterdir():
        if label_dir.is_dir():
            # Support multiple image formats
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(label_dir.glob(ext))
            
            if images:
                sample_images[label_dir.name] = sorted(images)
                logger.info(f"Found {len(images)} images in {label_dir.name}")
    
    return sample_images


def load_model(model_path: Optional[Path] = None) -> keras.Model:
    """
    Load trained Keras model.
    
    Args:
        model_path: Path to model file (uses default if None)
    
    Returns:
        Loaded Keras model
    
    Raises:
        ModelLoadError: If model fails to load
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ModelLoadError(f"Failed to load model: {str(e)}")


def predict(model: keras.Model, image: np.ndarray) -> float:
    """
    Execute prediction with model.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array
    
    Returns:
        Prediction probability (0-1)
    """
    try:
        # Ensure correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        prediction = model.predict(image, verbose=0)[0][0]
        return float(prediction)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


def predict_with_confidence(
    model: keras.Model, 
    image: np.ndarray
) -> Tuple[str, float, float]:
    """
    Make prediction with confidence level.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array
    
    Returns:
        Tuple of (predicted_class, probability, confidence)
    """
    prob = predict(model, image)
    
    # Determine class and confidence
    if prob < THRESHOLD:
        predicted_class = "benign"
        confidence = (1 - prob) * 100  # Distance from threshold
    else:
        predicted_class = "malignant"
        confidence = prob * 100
    
    return predicted_class, prob, confidence


def generate_gradcam(
    model: keras.Model,
    image: np.ndarray,
    layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model interpretability.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array
        layer_name: Name of convolutional layer (auto-detected if None)
    
    Returns:
        Heatmap array
    """
    try:
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            logger.warning("No convolutional layer found for Grad-CAM")
            return None
        
        # Ensure correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        # Create gradient model
        grad_model = keras.models.Model(
            inputs=[model.input],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, 0]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        
        # Resize to match input image
        heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
        
        return heatmap
    
    except Exception as e:
        logger.error(f"Grad-CAM generation error: {str(e)}")
        return None


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (0-1 range)
        heatmap: Grad-CAM heatmap
        alpha: Transparency of overlay
        colormap: OpenCV colormap
    
    Returns:
        Overlayed image
    """
    try:
        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply colormap to heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    except Exception as e:
        logger.error(f"Heatmap overlay error: {str(e)}")
        return image


def get_model_info(model: keras.Model) -> Dict[str, any]:
    """
    Extract model information.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model information
    """
    try:
        total_params = model.count_params()
        trainable_params = sum([
            tf.size(w).numpy() for w in model.trainable_weights
        ])
        non_trainable_params = total_params - trainable_params
        
        return {
            'name': model.name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
        }
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {}
