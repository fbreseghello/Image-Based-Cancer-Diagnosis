import streamlit as st
import numpy as np
import random
import json
import matplotlib.pyplot as plt

from model_utils import load_image, load_model, predict, get_sample_image_files

# --- Sidebar ---
st.sidebar.title("Painel de Navegação")
st.sidebar.info(
    """
    - **1.** Faça upload de uma imagem _ou_ use um exemplo.
    - **2.** Veja a predição e o resultado.
    - **3.** Marque abaixo para ver o gráfico do treinamento.
    """
)

if st.sidebar.checkbox("Mostrar gráfico de treinamento"):
    try:
        with open("treinamento_history.json") as f:
            hist = json.load(f)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist['accuracy'], label='Treino')
        ax[0].plot(hist['val_accuracy'], label='Validação')
        ax[0].set_title('Acurácia')
        ax[0].legend()
        ax[1].plot(hist['loss'], label='Treino')
        ax[1].plot(hist['val_loss'], label='Validação')
        ax[1].set_title('Loss')
        ax[1].legend()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Erro ao carregar histórico: {e}")

# --- Header & Instructions ---
st.header("Image diagnosis of cancer using histopathology")
st.markdown(
    "Predict whether samples of tumour tissue are *benign* or "
    "*malignant (cancerous)*.\n\n"
    "[hp]: https://en.wikipedia.org/wiki/Histopathology"
)

# --- Load model and sample images ---
with st.spinner("Carregando modelo..."):
    model = load_model()
sample_images = get_sample_image_files()

# --- Main tabs: Upload & Sample ---
upload_tab, sample_tab = st.tabs(["Upload an image", "Use a sample image"])

with upload_tab:
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if file:
        img = load_image(file, resize=True)
        st.image((img * 255).astype("uint8"))
        prob = predict(model, img)
        if prob < 0.5:
            st.success(f"Result: {prob:.5f} — :green['benign']")
        else:
            st.warning(f"Result: {prob:.5f} — :orange['malignant']")
    else:
        st.info("Faça upload de uma imagem para análise.")

with sample_tab:
    label = st.selectbox("Escolha a classe", ["benign", "malignant"])
    image_list = sample_images[label]
    if st.button("Mostrar amostra aleatória"):
        idx = random.randint(0, len(image_list) - 1)
        img = load_image(image_list[idx], resize=True)
        st.image((img * 255).astype("uint8"), caption=f"{label} sample")
        prob = predict(model, img)
        if prob < 0.5:
            st.success(f"Result: {prob:.5f} — :green['benign']")
        else:
            st.warning(f"Result: {prob:.5f} — :orange['malignant']")

st.caption("**This is an unvalidated project for AI study purposes. Do not consider the results presented. Please consult a specialized doctor.**")

