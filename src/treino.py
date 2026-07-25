"""CLI de treino: constrói o dataset de classificação a partir das anotações
YOLO, treina a CNN do zero e/ou o MobileNetV2 (transfer learning) e salva
modelos + métricas + figuras.

Exemplos:
    python -m src.treino --model both
    python -m src.treino --model mobilenet --epochs-head 10 --epochs-finetune 8
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from .dataset import build_dataframe, compute_class_weights, load_class_names, make_dataset, save_class_names
from .evaluate import evaluate_model, plot_learning_curves
from .models import build_cnn_scratch, build_mobilenetv2, default_callbacks


def _merge_histories(*histories: dict) -> dict:
    merged = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def train_cnn_scratch(train_ds, val_ds, class_weight, epochs, figures_dir, verbose=2):
    model = build_cnn_scratch()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=default_callbacks(),
        verbose=verbose,
    )
    plot_learning_curves(history.history, "CNN do zero", Path(figures_dir) / "curvas_cnn_scratch.png")
    return model, history.history


def train_mobilenetv2(train_ds, val_ds, class_weight, epochs_head, epochs_finetune, figures_dir, verbose=2):
    model = build_mobilenetv2(fine_tune_at=None)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        class_weight=class_weight,
        callbacks=default_callbacks(),
        verbose=verbose,
    )

    # etapa 2: descongela a base inteira e faz fine-tuning com LR baixo
    model.base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        class_weight=class_weight,
        callbacks=default_callbacks(patience=4),
        verbose=verbose,
    )

    merged_history = _merge_histories(history_head.history, history_ft.history)
    plot_learning_curves(merged_history, "MobileNetV2 (transfer learning)", Path(figures_dir) / "curvas_mobilenetv2.png")
    return model, merged_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina os classificadores de fratura óssea.")
    parser.add_argument("--model", choices=["cnn", "mobilenet", "both"], default="both")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-cnn", type=int, default=30)
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.data_dir)
    print(f"Total de imagens rotuladas: {len(df)}")
    save_class_names(output_dir / "class_names.json")
    class_names = load_class_names(output_dir / "class_names.json")
    class_weight = compute_class_weights(df, "train")

    train_ds = make_dataset(df, "train", batch_size=args.batch_size, shuffle=True)
    val_ds = make_dataset(df, "valid", batch_size=args.batch_size)
    test_ds = make_dataset(df, "test", batch_size=args.batch_size)

    summaries = []

    if args.model in ("cnn", "both"):
        model, _ = train_cnn_scratch(train_ds, val_ds, class_weight, args.epochs_cnn, args.figures_dir)
        model.save(output_dir / "cnn_scratch.keras")
        summaries.append(evaluate_model(model, test_ds, class_names, "cnn_scratch", args.figures_dir))

    if args.model in ("mobilenet", "both"):
        model, _ = train_mobilenetv2(
            train_ds, val_ds, class_weight, args.epochs_head, args.epochs_finetune, args.figures_dir
        )
        model.save(output_dir / "mobilenetv2_transfer.keras")
        summaries.append(evaluate_model(model, test_ds, class_names, "mobilenetv2_transfer", args.figures_dir))

    summary_path = Path(args.figures_dir) / "resumo_modelos.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps([{k: v for k, v in s.items() if k != "report"} for s in summaries], indent=2), encoding="utf-8"
    )
    print("\nResumo final:")
    for s in summaries:
        print(f"  {s['name']:22s} accuracy={s['accuracy']:.3f}  macro_f1={s['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
