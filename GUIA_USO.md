# Guia de Uso Rápido - Image-Based Cancer Diagnosis

## 🚀 Início Rápido

### 1. Configuração Inicial

```bash
# Clone o repositório
git clone https://github.com/fbreseghello/Image-Based-Cancer-Diagnosis.git
cd Image-Based-Cancer-Diagnosis

# Execute o script de setup
python setup.py
```

Este script irá:
- Criar as pastas necessárias
- Instalar todas as dependências
- Verificar a instalação

### 2. Preparar Dataset

Organize suas imagens histopatológicas:

```
sample_images/
├── benign/          <- Imagens de tecido benigno
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── malignant/       <- Imagens de tecido maligno
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

**Requisitos das imagens:**
- Formatos suportados: JPG, JPEG, PNG, BMP
- Tamanho: Qualquer (será redimensionado para 224x224)
- Recomendado: Mínimo 100 imagens por classe para bons resultados

### 3. Treinar o Modelo

**Treinamento básico:**
```bash
python train_model.py
```

**Treinamento avançado:**
```bash
# Com mais épocas
python train_model.py --epochs 50

# Com transfer learning (mais rápido e geralmente melhor)
python train_model.py --transfer-learning --epochs 30

# Personalizando taxa de aprendizado
python train_model.py --learning-rate 0.0001 --epochs 40
```

**Durante o treinamento:**
- O progresso será exibido em tempo real
- Modelos são salvos automaticamente em `models/`
- Histórico de treinamento salvo em `treinamento_history.json`
- Logs do TensorBoard em `logs/`

**Monitorar com TensorBoard:**
```bash
tensorboard --logdir logs/
# Abra http://localhost:6006 no navegador
```

### 4. Avaliar o Modelo

```bash
python evaluate_model.py
```

Isso irá gerar:
- `models/evaluation/cnn_model_confusion_matrix.png` - Matriz de confusão
- `models/evaluation/cnn_model_roc_curve.png` - Curva ROC
- `models/evaluation/cnn_model_pr_curve.png` - Curva Precision-Recall
- `models/evaluation/cnn_model_distribution.png` - Distribuição de predições
- `models/evaluation/cnn_model_results.txt` - Relatório completo

### 5. Executar Aplicação Web

```bash
python run_app.py
```

Ou diretamente:
```bash
streamlit run src/app.py
```

A aplicação abrirá em `http://localhost:8501`

---

## 📊 Interpretando os Resultados

### Métricas do Modelo

- **Accuracy (Acurácia)**: Porcentagem de predições corretas
  - > 0.90: Excelente
  - 0.80-0.90: Bom
  - < 0.80: Precisa melhorar

- **Precision (Precisão)**: Das predições positivas, quantas estavam corretas
  - Importante quando queremos evitar falsos positivos

- **Recall (Sensibilidade)**: Dos casos positivos reais, quantos foram detectados
  - Crítico em medicina - queremos detectar todos os casos de câncer

- **AUC-ROC**: Área sob a curva ROC (0.5 a 1.0)
  - > 0.95: Excelente
  - 0.90-0.95: Muito bom
  - 0.80-0.90: Bom
  - < 0.80: Precisa melhorar

### Grad-CAM (Mapa de Ativação)

O Grad-CAM mostra quais regiões da imagem o modelo focou:
- **Vermelho/Amarelo**: Regiões mais importantes para a decisão
- **Azul/Verde**: Regiões menos importantes
- **Validação**: Verifique se o modelo está olhando para as características corretas do tecido

---

## 🔧 Solução de Problemas

### Erro: "Model file not found"
```bash
# Você precisa treinar o modelo primeiro
python train_model.py
```

### Erro: "No sample images found"
```bash
# Verifique se as imagens estão nas pastas corretas
ls sample_images/benign/
ls sample_images/malignant/
```

### Erro de memória durante treinamento
```python
# Em src/config.py, reduza o BATCH_SIZE
BATCH_SIZE = 8  # ou menor
```

### Modelo com baixa acurácia
Possíveis soluções:
1. **Mais dados**: Adicione mais imagens de treinamento
2. **Data augmentation**: Já está ativado por padrão
3. **Transfer learning**: Use `--transfer-learning`
4. **Mais épocas**: Aumente `--epochs 50` ou mais
5. **Ajuste de hiperparâmetros**: Modifique `src/config.py`

### App Streamlit lento
```bash
# Use modelo mais leve ou desative Grad-CAM na sidebar
# Ou reduza o tamanho das imagens de entrada
```

---

## 🎯 Melhores Práticas

### Para Treinamento

1. **Dataset balanceado**: Número similar de imagens benign/malignant
2. **Validação separada**: Use 20% dos dados para validação (já configurado)
3. **Early stopping**: Evita overfitting (já implementado)
4. **Monitoramento**: Sempre use TensorBoard para visualizar métricas
5. **Checkpoints**: Modelos são salvos automaticamente

### Para Predições

1. **Confiança**: Sempre verifique o nível de confiança
   - > 90%: Alta confiança
   - 60-90%: Confiança moderada
   - < 60%: Baixa confiança - considere testes adicionais

2. **Grad-CAM**: Use para validar que o modelo está olhando regiões relevantes

3. **Nunca use sozinho**: Este é um sistema de apoio à decisão, não substitui médicos

---

## ⚙️ Configurações Avançadas

### Modificar Hiperparâmetros

Edite `src/config.py`:

```python
# Aumentar tamanho da imagem (melhora qualidade, mas usa mais memória)
IMG_SIZE = 299

# Ajustar data augmentation
ROTATION_RANGE = 30  # Mais rotação
ZOOM_RANGE = 0.3     # Mais zoom

# Regularização
DROPOUT_RATE = 0.6   # Mais dropout = menos overfitting
L2_REGULARIZATION = 0.01  # Mais regularização

# Early stopping
EARLY_STOPPING_PATIENCE = 10  # Mais paciência
```

### Usar GPU

Se você tem GPU NVIDIA com CUDA:

```bash
# Verifique se o TensorFlow detecta a GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Se não detectar, instale tensorflow-gpu
pip install tensorflow-gpu
```

### Exportar Modelo

```python
# Para TensorFlow Lite (mobile)
import tensorflow as tf
model = tf.keras.models.load_model('models/cnn_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 📚 Recursos Adicionais

- **TensorFlow Tutoriais**: https://www.tensorflow.org/tutorials
- **Streamlit Docs**: https://docs.streamlit.io
- **Grad-CAM Paper**: https://arxiv.org/abs/1610.02391
- **Medical Imaging ML**: https://www.tensorflow.org/tutorials/images/classification

---

## 💡 Dicas

1. **Comece pequeno**: Teste com poucos dados primeiro
2. **Transfer learning**: Geralmente melhor que treinar do zero
3. **Validação**: Sempre avalie com `evaluate_model.py`
4. **Experimente**: Teste diferentes hiperparâmetros
5. **Documente**: Anote os resultados de diferentes configurações

---

## ⚠️ Lembrete Importante

**Este projeto é apenas para fins educacionais e de pesquisa.**

Nunca use os resultados para diagnósticos médicos reais. Sempre consulte profissionais de saúde qualificados.

---

Desenvolvido com ❤️ por Felipe Breseghello
