"""
Model evaluation script with comprehensive metrics and visualizations.
Generates confusion matrix, ROC curve, and classification report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from config import *


def load_test_dataset(data_dir: Path):
    """Load test dataset."""
    print(f"Loading test dataset from {data_dir}...")
    
    dataset = image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,  # Important for evaluation
        seed=RANDOM_SEED
    )
    
    return dataset


def get_predictions(model, dataset):
    """Get predictions and true labels from dataset."""
    print("Generating predictions...")
    
    y_true = []
    y_pred_probs = []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_probs.extend(predictions.flatten())
        y_true.extend(labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= THRESHOLD).astype(int)
    
    return y_true, y_pred, y_pred_probs


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_probs, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_pred_probs, save_path=None):
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
    avg_precision = average_precision_score(y_true, y_pred_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()
    
    return avg_precision


def print_classification_report(y_true, y_pred):
    """Print detailed classification report."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)


def plot_prediction_distribution(y_pred_probs, y_true, save_path=None):
    """Plot distribution of prediction probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate probabilities by true class
    benign_probs = y_pred_probs[y_true == 0]
    malignant_probs = y_pred_probs[y_true == 1]
    
    # Plot histograms
    ax.hist(benign_probs, bins=30, alpha=0.6, label='True Benign', color='green', edgecolor='black')
    ax.hist(malignant_probs, bins=30, alpha=0.6, label='True Malignant', color='red', edgecolor='black')
    
    # Add threshold line
    ax.axvline(x=THRESHOLD, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold ({THRESHOLD})')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution saved to {save_path}")
    
    plt.show()


def main(args):
    """Main evaluation function."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    model_path = MODEL_DIR / args.model_name
    print(f"\nLoading model from {model_path}...")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load dataset
    dataset = load_test_dataset(SAMPLE_IMG_DIR)
    
    # Get predictions
    y_true, y_pred, y_pred_probs = get_predictions(model, dataset)
    
    print(f"\nTotal samples: {len(y_true)}")
    print(f"Benign samples: {np.sum(y_true == 0)}")
    print(f"Malignant samples: {np.sum(y_true == 1)}")
    
    # Create output directory
    eval_dir = MODEL_DIR / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating evaluation plots...")
    print("-" * 70)
    
    # Confusion Matrix
    cm_path = eval_dir / f"{Path(args.model_name).stem}_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    
    # ROC Curve
    roc_path = eval_dir / f"{Path(args.model_name).stem}_roc_curve.png"
    roc_auc = plot_roc_curve(y_true, y_pred_probs, save_path=roc_path)
    
    # Precision-Recall Curve
    pr_path = eval_dir / f"{Path(args.model_name).stem}_pr_curve.png"
    avg_precision = plot_precision_recall_curve(y_true, y_pred_probs, save_path=pr_path)
    
    # Prediction Distribution
    dist_path = eval_dir / f"{Path(args.model_name).stem}_distribution.png"
    plot_prediction_distribution(y_pred_probs, y_true, save_path=dist_path)
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print(f"Accuracy: {np.mean(y_true == y_pred):.4f}")
    
    # Save results
    results_path = eval_dir / f"{Path(args.model_name).stem}_results.txt"
    with open(results_path, 'w') as f:
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Total samples: {len(y_true)}\n")
        f.write(f"Benign samples: {np.sum(y_true == 0)}\n")
        f.write(f"Malignant samples: {np.sum(y_true == 1)}\n\n")
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        f.write(f"Average Precision Score: {avg_precision:.4f}\n")
        f.write(f"Accuracy: {np.mean(y_true == y_pred):.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    print(f"\n✓ Results saved to {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model with comprehensive metrics"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cnn_model.h5",
        help="Name of the model file to evaluate (default: cnn_model.h5)"
    )
    
    args = parser.parse_args()
    main(args)
