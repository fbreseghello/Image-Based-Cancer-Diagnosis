from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
SAMPLE_IMG_DIR = Path("sample_images")

def load_image(image, resize: bool = False):
    """Abre e processa a imagem para o modelo."""
    img = Image.open(image).convert("RGB")
    if resize:
        img = img.resize((IMG_SIZE, IMG_SIZE))
        return np.array(img) / 255.0
    return img

def get_sample_image_files() -> dict:
    """Coleta imagens de exemplo, separadas por label."""
    return {
        dir.name: list(dir.glob("*.jpg"))
        for dir in SAMPLE_IMG_DIR.iterdir() if dir.is_dir()
    }

def load_model(model_path="models/cnn_model.h5") -> tf.keras.Model:
    """Carrega o modelo Keras salvo."""
    return tf.keras.models.load_model(model_path)

def predict(model, image: np.ndarray) -> float:
    """Executa a predição com o modelo."""
    img = np.expand_dims(image, 0)
    pred = model.predict(img, verbose=0)[0][0]
    return float(pred)
