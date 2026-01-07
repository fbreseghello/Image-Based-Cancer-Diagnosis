"""
Configuration file for the Image-Based Cancer Diagnosis project.
Centralizes all constants and hyperparameters.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_IMG_DIR = PROJECT_ROOT / "sample_images"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Image parameters
IMG_SIZE = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# Training parameters
BATCH_SIZE = 16
EPOCHS = 30
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Model parameters
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.001

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2
FILL_MODE = "nearest"

# Model paths
DEFAULT_MODEL_PATH = MODEL_DIR / "cnn_model.h5"
HISTORY_PATH = PROJECT_ROOT / "treinamento_history.json"

# Class labels
CLASS_NAMES = ["benign", "malignant"]
THRESHOLD = 0.5

# Callbacks
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
