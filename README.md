# Image-Based Cancer Diagnosis

An AI-powered system for diagnosing cancer through histopathology image analysis using deep learning. This project utilizes Convolutional Neural Networks (CNNs) to classify tumor tissue samples as **benign** or **malignant**, with the goal of assisting healthcare professionals in early cancer detection.

## Important Disclaimer

**This is an unvalidated project for AI study and educational purposes only.** 

Do not use the results presented here for actual medical diagnosis. Always consult a specialized healthcare professional for proper medical evaluation and treatment.

---

## Features

- **Advanced CNN Architecture**: Custom deep learning model with:
  - Batch normalization for stable training
  - Dropout layers for regularization
  - Data augmentation for better generalization
  - Optional transfer learning with MobileNetV2
  
- **Interactive Web Interface**: Built with Streamlit featuring:
  - Image upload and analysis
  - Real-time predictions with confidence scores
  - Grad-CAM visualization for model interpretability
  - Sample image testing
  - Training history visualization
  
- **Comprehensive Evaluation**: Tools for model assessment including:
  - Confusion matrix
  - ROC curve and AUC score
  - Precision-Recall curve
  - Detailed classification reports
  - Prediction distribution analysis

- **Production-Ready Code**:
  - Error handling and logging
  - Configurable hyperparameters
  - Model checkpointing and early stopping
  - TensorBoard integration
  - Command-line interface

---

## Project Structure

```
Image-Based-Cancer-Diagnosis/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Streamlit web application
│   ├── config.py                # Configuration and constants
│   └── model_utils.py           # Utility functions for model operations
├── models/                      # Saved models directory
│   ├── cnn_model.h5            # Trained model (generated)
│   └── evaluation/             # Evaluation results (generated)
├── logs/                        # TensorBoard logs (generated)
├── sample_images/              # Dataset directory
│   ├── benign/                 # Benign samples
│   └── malignant/              # Malignant samples
├── train_model.py              # Model training script
├── evaluate_model.py           # Model evaluation script
├── requirements.txt            # Python dependencies
├── parameters.json             # Azure ML configuration (optional)
├── template.json               # Azure deployment template (optional)
└── README.md                   # This file
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/fbreseghello/Image-Based-Cancer-Diagnosis.git
   cd Image-Based-Cancer-Diagnosis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   
   Organize your histopathology images in the following structure:
   ```
   sample_images/
   ├── benign/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── malignant/
       ├── image1.jpg
       ├── image2.jpg
       └── ...
   ```

---

## 💻 Usage

### Training the Model

**Basic training:**
```bash
python train_model.py
```

**Advanced options:**
```bash
python train_model.py --epochs 50 --learning-rate 0.0001 --model-name my_model.h5
```

**Using transfer learning:**
```bash
python train_model.py --transfer-learning --epochs 30
```

**Available arguments:**
- `--epochs`: Number of training epochs (default: 30)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--model-name`: Name for the saved model file (default: cnn_model.h5)
- `--transfer-learning`: Use MobileNetV2 for transfer learning

### Evaluating the Model

```bash
python evaluate_model.py --model-name cnn_model.h5
```

This generates:
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Classification report
- Prediction distribution analysis

All results are saved in `models/evaluation/`

### Running the Web Application

```bash
streamlit run src/app.py
```

Then open your browser to `http://localhost:8501`

**Features of the web app:**
- Upload histopathology images for instant diagnosis
- Test with random sample images
- View prediction confidence and probabilities
- Visualize Grad-CAM heatmaps showing model attention
- Review training history and metrics
- Access model information and architecture details

### Monitoring Training (TensorBoard)

```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006` to view training metrics in real-time.

---

## 🔧 Configuration

All hyperparameters and settings can be modified in [src/config.py](src/config.py):

```python
# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 16

# Training parameters
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model parameters
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.001

# Data augmentation
ROTATION_RANGE = 20
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# And more...
```

---

## Model Architecture

### Custom CNN (Default)

- **Input Layer**: 224×224×3 RGB images
- **Data Augmentation**: Random flips, rotations, zoom, and shifts
- **3 Convolutional Blocks**:
  - Conv2D (32, 64, 128 filters) → BatchNorm → MaxPooling → Dropout
- **Dense Layers**: 256 → 128 → 1 (sigmoid)
- **Regularization**: L2 regularization + Dropout (0.5)
- **Total Parameters**: ~1.5M (varies by architecture)

### Transfer Learning (Optional)

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Head**: GlobalAvgPool → Dense(128) → Dropout → Dense(1)
- **Training Strategy**: Feature extraction with frozen base

---

## 📈 Performance Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive predictions that are correct
- **Recall (Sensitivity)**: Proportion of actual positives identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Average Precision**: Area under the Precision-Recall curve

*Note: Actual performance depends on your dataset quality and size.*

---

## Understanding Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which regions of the image the model focused on when making its prediction. This helps:

- Validate that the model is looking at relevant tissue features
- Build trust in model predictions
- Identify potential biases or artifacts
- Assist pathologists in understanding AI decisions

---

## 🛠️ Development

### Code Quality

The project includes tools for maintaining code quality:

```bash
# Format code with Black
black .

# Check code style with flake8
flake8 .

# Run tests (if implemented)
pytest
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Resources

- [Histopathology](https://en.wikipedia.org/wiki/Histopathology)
- [CNN for Medical Imaging](https://www.tensorflow.org/tutorials/images/classification)
- [Transfer Learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)

---

**Remember**: This is a research and educational project. Always consult qualified medical professionals for health-related decisions.

