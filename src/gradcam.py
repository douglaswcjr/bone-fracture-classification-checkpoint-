"""Grad-CAM (Selvaraju et al., 2017) implementado manualmente com
`tf.GradientTape`, reutilizado pelo notebook de avaliação e pelo app Streamlit.

Suporta dois formatos de modelo:
- "achatado": todas as camadas (incluindo a última Conv2D) pertencem direto ao
  modelo (caso da CNN treinada do zero).
- "base aninhada": o modelo tem uma base pré-treinada (ex.: MobileNetV2) chamada
  como um submodelo — em Keras 3 não é possível religar o grafo através dela, então
  recompomos a passagem "base -> head" manualmente.

A detecção do caso "base aninhada" é estrutural (procura por uma camada que seja
ela própria um `tf.keras.Model` dentro de `model.layers`), não depende de atributos
Python customizados — esses NÃO sobrevivem a um `model.save()`/`load_model()`, que é
exatamente o caminho usado pelo app Streamlit e por `predicao.py`.
"""

import numpy as np
import tensorflow as tf


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Acha o nome da última camada convolucional de um modelo achatado."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Nenhuma camada Conv2D encontrada no modelo.")


def _find_nested_model_layer(model: tf.keras.Model) -> tf.keras.Model | None:
    """Acha a camada que seja ela própria um `tf.keras.Model` E contenha
    camadas convolucionais (ex.: a base MobileNetV2 embutida) — não basta
    checar `isinstance(_, tf.keras.Model)`, porque o bloco de augmentation
    (`tf.keras.Sequential`) também é um `tf.keras.Model` e viria antes na
    lista de camadas. Retorna None se o modelo for "achatado"."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(
            isinstance(inner, tf.keras.layers.Conv2D) for inner in layer.layers
        ):
            return layer
    return None


def make_gradcam_heatmap(
    image_batch: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    class_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """Gera o mapa de calor Grad-CAM para uma imagem (batch de 1, em [0,255]).

    Retorna (heatmap normalizado em [0,1] no tamanho da última conv, classe usada).
    """
    base_model = _find_nested_model_layer(model)
    if base_model is not None:
        return _gradcam_nested_base(image_batch, model, base_model, last_conv_layer_name, class_index)
    return _gradcam_flat(image_batch, model, last_conv_layer_name, class_index)


def _gradcam_flat(image_batch, model, last_conv_layer_name, class_index):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    return _pool_and_normalize(conv_outputs, grads), class_index


def _gradcam_nested_base(image_batch, model, base_model, last_conv_layer_name, class_index):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(base_model)

    activation_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(tf.identity(image_batch))

    # As camadas do "head" (tudo que vem depois da base no grafo do modelo
    # externo — GlobalAveragePooling2D, Dropout, Dense) são recuperadas pela
    # posição na lista `model.layers`, que o Keras preserva ao salvar/carregar.
    base_index = model.layers.index(base_model)
    head_layers = model.layers[base_index + 1 :]

    with tf.GradientTape() as tape:
        conv_outputs, base_output = activation_model(preprocessed, training=False)
        x = base_output
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    return _pool_and_normalize(conv_outputs, grads), class_index


def _pool_and_normalize(conv_outputs: tf.Tensor, grads: tf.Tensor) -> np.ndarray:
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Sobrepõe o heatmap (colormap 'jet') em cima da imagem original (RGB uint8)."""
    import matplotlib as mpl

    h, w = image_uint8.shape[:2]
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (h, w)).numpy().squeeze()
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    overlaid = jet_heatmap * alpha + image_uint8 * (1 - alpha)
    return np.uint8(overlaid)
