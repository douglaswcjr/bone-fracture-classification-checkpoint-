"""Monta um dataset de classificação de imagem inteira a partir das anotações
YOLOv8 (bounding boxes) do HBFMID.

Cada imagem pode ter mais de uma bounding box (às vezes de classes diferentes).
Como o projeto pede classificação da imagem inteira (não detecção), o rótulo de
cada imagem é a classe majoritária entre suas boxes — critério documentado no
notebook da Parte 1. Imagens sem nenhuma box são descartadas (sem rótulo confiável).
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Precisa bater com a ordem de `data/data.yaml` (índice = class_id do YOLO).
CLASS_NAMES = [
    "Comminuted",
    "Greenstick",
    "Healthy",
    "Linear",
    "Oblique Displaced",
    "Oblique",
    "Segmental",
    "Spiral",
    "Transverse Displaced",
    "Transverse",
]

IMG_SIZE = 224
SPLITS = ("train", "valid", "test")


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Lê um .txt YOLO e retorna a lista de boxes (class_id, x, y, w, h)."""
    boxes = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        class_id = int(parts[0])
        x, y, w, h = (float(v) for v in parts[1:5])
        boxes.append((class_id, x, y, w, h))
    return boxes


def majority_class(boxes: list[tuple[int, float, float, float, float]]) -> int | None:
    """Classe majoritária entre as boxes de uma imagem.

    Empate é resolvido pelo menor class_id (regra determinística e documentada).
    Retorna None se não houver nenhuma box.
    """
    if not boxes:
        return None
    counts = Counter(b[0] for b in boxes)
    max_count = max(counts.values())
    winners = sorted(cls for cls, n in counts.items() if n == max_count)
    return winners[0]


def build_dataframe(data_dir: str | Path = "data", splits: tuple[str, ...] = SPLITS) -> pd.DataFrame:
    """Varre `data/{split}/{images,labels}` e monta um DataFrame com um rótulo
    (classe majoritária) por imagem. Imagens sem nenhuma box são excluídas.
    """
    data_dir = Path(data_dir)
    rows = []
    for split in splits:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = parse_yolo_label(label_path) if label_path.exists() else []
            class_id = majority_class(boxes)
            if class_id is None:
                continue
            rows.append(
                {
                    "filepath": str(image_path),
                    "label_path": str(label_path),
                    "split": split,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "n_boxes": len(boxes),
                    "n_unique_classes": len({b[0] for b in boxes}),
                }
            )
    return pd.DataFrame(rows)


def decode_image(raw_bytes: tf.Tensor, img_size: int = IMG_SIZE) -> tf.Tensor:
    """Decodifica bytes de imagem (jpg/png) e redimensiona. Retorna float32 em
    [0, 255] — a normalização de cada modelo (Rescaling, preprocess_input do
    MobileNetV2, etc.) fica embutida na própria arquitetura do modelo."""
    image = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size], method="bilinear")
    return tf.cast(image, tf.float32)


def load_image(path: tf.Tensor, img_size: int = IMG_SIZE) -> tf.Tensor:
    """Lê um arquivo de imagem do disco e delega a decodificação a `decode_image`."""
    raw = tf.io.read_file(path)
    return decode_image(raw, img_size)


def make_dataset(
    df: pd.DataFrame,
    split: str,
    batch_size: int = 32,
    shuffle: bool = False,
    img_size: int = IMG_SIZE,
) -> tf.data.Dataset:
    """Constrói um tf.data.Dataset (imagem, rótulo inteiro) para um split."""
    subset = df[df["split"] == split]
    paths = subset["filepath"].tolist()
    labels = subset["class_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: (load_image(path, img_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(df: pd.DataFrame, split: str = "train") -> dict[int, float]:
    """Pesos por classe (sklearn `balanced`) calculados só no split de treino,
    para compensar o desbalanceamento discutido na Parte 1."""
    subset = df[df["split"] == split]
    classes = np.array(sorted(subset["class_id"].unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=subset["class_id"])
    return dict(zip(classes.tolist(), weights))


def save_class_names(path: str | Path = "models/class_names.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2), encoding="utf-8")


def load_class_names(path: str | Path = "models/class_names.json") -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
