"""
Enhanced training script for cancer diagnosis CNN model.
Includes data augmentation, callbacks, and comprehensive logging.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
from config import *


def create_data_augmentation():
    """Create data augmentation layer."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(ROTATION_RANGE / 360.0),
        layers.RandomZoom(ZOOM_RANGE),
        layers.RandomTranslation(HEIGHT_SHIFT_RANGE, WIDTH_SHIFT_RANGE),
    ], name="data_augmentation")


def build_cnn_model(use_transfer_learning=False):
    """
    Build CNN model for binary classification.
    
    Args:
        use_transfer_learning: If True, use MobileNetV2 as base model
    
    Returns:
        Compiled Keras model
    """
    if use_transfer_learning:
        # Use transfer learning with MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=INPUT_SHAPE,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base model
        
        model = models.Sequential([
            layers.Rescaling(1./127.5, offset=-1, input_shape=INPUT_SHAPE),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(128, activation='relu', 
                        kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
            layers.Dropout(DROPOUT_RATE / 2),
            layers.Dense(1, activation='sigmoid'),
        ], name="transfer_learning_model")
    else:
        # Custom CNN architecture
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=INPUT_SHAPE),
            create_data_augmentation(),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
            layers.BatchNormalization(),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(128, activation='relu'),
            layers.Dropout(DROPOUT_RATE / 2),
            layers.Dense(1, activation='sigmoid'),
        ], name="custom_cnn_model")
    
    return model


def load_datasets():
    """Load and prepare training and validation datasets."""
    print(f"Loading datasets from {SAMPLE_IMG_DIR}...")
    
    train_dataset = image_dataset_from_directory(
        SAMPLE_IMG_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
        validation_split=VALIDATION_SPLIT,
        subset="training"
    )
    
    val_dataset = image_dataset_from_directory(
        SAMPLE_IMG_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
        validation_split=VALIDATION_SPLIT,
        subset="validation"
    )
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset


def get_callbacks(model_path):
    """Create training callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=str(LOGS_DIR / f"fit_{timestamp}"),
            histogram_freq=1
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path=None):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_history(history, filepath=HISTORY_PATH):
    """Save training history to JSON."""
    history_dict = {key: [float(val) for val in values] 
                   for key, values in history.history.items()}
    
    with open(filepath, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to {filepath}")


def main(args):
    """Main training function."""
    print("=" * 70)
    print("Cancer Diagnosis CNN Training")
    print("=" * 70)
    
    # Load datasets
    train_dataset, val_dataset = load_datasets()
    
    # Build model
    print(f"\nBuilding model (transfer_learning={args.transfer_learning})...")
    model = build_cnn_model(use_transfer_learning=args.transfer_learning)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]
    )
    
    # Print model summary
    model.summary()
    
    # Get callbacks
    model_path = MODEL_DIR / args.model_name
    callbacks = get_callbacks(model_path)
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("-" * 70)
    print("\nTraining completed!")
    
    # Save history
    save_history(history)
    
    # Plot training history
    plot_path = MODEL_DIR / f"{Path(args.model_name).stem}_history.png"
    plot_training_history(history, save_path=plot_path)
    
    # Final evaluation
    print("\nFinal Model Evaluation:")
    print("-" * 70)
    results = model.evaluate(val_dataset, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize():.<30} {value:.4f}")
    
    print("\n" + "=" * 70)
    print(f"Model saved to: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN model for cancer diagnosis"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--model-name", type=str, default="cnn_model.h5",
        help="Name for saved model file (default: cnn_model.h5)"
    )
    parser.add_argument(
        "--transfer-learning", action="store_true",
        help="Use transfer learning with MobileNetV2"
    )
    
    args = parser.parse_args()
    
    main(args)
