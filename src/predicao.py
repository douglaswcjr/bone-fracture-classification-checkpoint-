"""Inferência: carrega um modelo treinado e classifica uma imagem de raio-X.

Responsabilidade deste módulo: SÓ inferência (carrega um `.keras` já salvo,
classifica uma imagem por vez). Não importa nem depende de `src/treino.py`
(que cuida do treino em si) — só precisa do arquivo `.keras` produzido por
ele. É o único módulo de classificação importado por `app.py`.

Usado tanto pelo app Streamlit (`app.py`) quanto para testes manuais via CLI:
    python -m src.predicao models/mobilenetv2_transfer.keras caminho/para/imagem.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from .dataset import IMG_SIZE, decode_image, load_class_names, load_image


def load_model(model_path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def _predict_from_image(
    image: tf.Tensor, model: tf.keras.Model, class_names: list[str]
) -> tuple[str, dict[str, float], np.ndarray]:
    """Classifica uma imagem já decodificada/redimensionada (H, W, 3) float32.

    Retorna (classe_prevista, {classe: probabilidade}, batch_de_entrada_do_modelo).
    O batch é devolvido para poder ser reaproveitado direto no Grad-CAM sem
    reler/redecodificar a imagem.
    """
    batch = tf.expand_dims(image, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    predicted_class = class_names[int(np.argmax(probs))]
    probabilities = {name: float(p) for name, p in zip(class_names, probs)}
    return predicted_class, probabilities, batch.numpy()


def predict_image(
    image_path: str | Path,
    model: tf.keras.Model,
    class_names: list[str],
    img_size: int = IMG_SIZE,
) -> tuple[str, dict[str, float], np.ndarray]:
    """Classifica uma imagem a partir de um caminho no disco."""
    image = load_image(tf.constant(str(image_path)), img_size=img_size)
    return _predict_from_image(image, model, class_names)


def predict_bytes(
    image_bytes: bytes,
    model: tf.keras.Model,
    class_names: list[str],
    img_size: int = IMG_SIZE,
) -> tuple[str, dict[str, float], np.ndarray]:
    """Classifica uma imagem a partir de bytes crus (ex.: upload no Streamlit)."""
    image = decode_image(tf.constant(image_bytes), img_size=img_size)
    return _predict_from_image(image, model, class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica uma imagem de raio-X.")
    parser.add_argument("model_path", help="Caminho do modelo .keras salvo")
    parser.add_argument("image_path", help="Caminho da imagem a classificar")
    parser.add_argument("--class-names", default="models/class_names.json")
    args = parser.parse_args()

    model = load_model(args.model_path)
    class_names = load_class_names(args.class_names)
    predicted_class, probabilities, _ = predict_image(args.image_path, model, class_names)

    print(f"Classe prevista: {predicted_class}\n")
    for name, prob in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {prob:6.2%}")


if __name__ == "__main__":
    sys.exit(main())
