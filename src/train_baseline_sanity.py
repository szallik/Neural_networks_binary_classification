from __future__ import annotations
import sys
from pathlib import Path
import tensorflow as tf
tf.keras.utils.set_random_seed(42)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_loader import get_datasets  # noqa: E402
from models import build_baseline_cnn  # noqa: E402

def main() -> None:
    print("== SANITY TRAIN RUN: Baseline CNN ==")

    # 1) Dane
    train_ds, val_ds, _, class_names = get_datasets()
    print(f"[INFO] class_names: {class_names}")

    # 2) Model
    model = build_baseline_cnn(
        input_shape=(224, 224, 3),
        dense_units=64,
        dropout_rate=0.4,
        name="baseline_cnn_flatten",
    )
    model.summary()

    # 3) Kompilacja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # 4) Trening (krotki sanity run tylko tak o)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2,
        steps_per_epoch=10,
        validation_steps=3,
        verbose=1
    )

    # 5) Szybkie podsumowanie
    final = {k: v[-1] for k, v in history.history.items()}
    print("\n== SANITY RUN DONE ==")
    for k, v in final.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
