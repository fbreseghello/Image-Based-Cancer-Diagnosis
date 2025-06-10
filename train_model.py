import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = 224
BATCH_SIZE = 8

# Carrega as imagens das pastas sample_images/benign e sample_images/malignant
train_dataset = image_dataset_from_directory(
    "sample_images",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training"
)
val_dataset = image_dataset_from_directory(
    "sample_images",
    labels="inferred",
    label_mode="binary",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"
)

# Define o modelo CNN simples
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(train_dataset, validation_data=val_dataset, epochs=8)

# Salva o modelo treinado
model.save('cnn_model.h5')
print("Modelo treinado e salvo como cnn_model.h5!")
