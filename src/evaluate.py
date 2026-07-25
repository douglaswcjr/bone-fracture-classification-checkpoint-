"""Avaliação compartilhada entre o notebook da Parte 2 e `treino.py`: coleta
previsões de um split, gera classification_report, matriz de confusão e
curvas de aprendizado.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def collect_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(probs, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path | None = None,
):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_learning_curves(history: dict, title: str, save_path: str | Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="treino")
    axes[0].plot(history["val_loss"], label="validação")
    axes[0].set_title(f"{title} — loss")
    axes[0].set_xlabel("época")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="treino")
    axes[1].plot(history["val_accuracy"], label="validação")
    axes[1].set_title(f"{title} — acurácia")
    axes[1].set_xlabel("época")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    return fig


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
    name: str,
    figures_dir: str | Path = "reports/figures",
) -> dict:
    """Roda o modelo no split de avaliação e salva relatório + matriz de confusão.

    Retorna um resumo (accuracy, f1 macro) usado na tabela comparativa final.
    Observação: classes ausentes no split (support=0) recebem 0.0 nas métricas
    (zero_division=0) — é o caso de Linear/Segmental no conjunto de teste oficial.
    """
    y_true, y_pred = collect_predictions(model, dataset)
    report_text = classification_report(
        y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, labels=range(len(class_names)), target_names=class_names, zero_division=0, output_dict=True
    )
    print(f"\n=== {name} ===")
    print(report_text)

    figures_dir = Path(figures_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, f"Matriz de confusão — {name}", figures_dir / f"cm_{name}.png")

    return {
        "name": name,
        "accuracy": report_dict["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": report_dict,
    }
