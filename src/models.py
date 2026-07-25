"""Arquiteturas dos dois modelos do projeto (CNN do zero e transfer learning
com MobileNetV2), compartilhadas entre o notebook da Parte 2 e `treino.py`.

A normalização de cada modelo fica embutida na própria arquitetura (Rescaling
ou `preprocess_input`), então `dataset.load_image` sempre entrega imagens cruas
em [0, 255] e qualquer modelo aqui sabe lidar com isso sozinho.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .dataset import CLASS_NAMES, IMG_SIZE

NUM_CLASSES = len(CLASS_NAMES)


def _augmentation_block() -> tf.keras.Sequential:
    """Augmentation só ativa em `training=True` (dentro de model.fit)."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def build_cnn_scratch(img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """CNN convolucional simples treinada do zero."""
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = _augmentation_block()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="cnn_scratch")


def build_mobilenetv2(
    img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES, fine_tune_at: int | None = None
) -> tf.keras.Model:
    """Transfer learning com MobileNetV2 pré-treinada na ImageNet.

    `fine_tune_at`: se informado, descongela as camadas da base a partir desse
    índice (usado na segunda etapa de fine-tuning); se None, a base fica toda
    congelada (etapa 1, treino só do head).
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = _augmentation_block()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = fine_tune_at is not None
    if fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    # Referência direta usada por `train_mobilenetv2` (src/treino.py) para
    # descongelar a base na etapa de fine-tuning. Não depender disso para o
    # Grad-CAM (src/gradcam.py): atributos Python customizados como este não
    # sobrevivem a um `model.save()`/`load_model()`, então o Grad-CAM localiza
    # a base aninhada estruturalmente em vez de usar este atalho.
    model.base_model = base_model
    return model


def default_callbacks(patience: int = 6) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6
        ),
    ]
