from __future__ import annotations

from pathlib import Path
import tensorflow as tf

from models import build_baseline_cnn, build_transfer_model


def _print_model_info(model: tf.keras.Model, title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    model.summary()
    print(f"\n[INFO] Liczba parametrow (trainable):   {model.count_params():,}")
    trainable = sum(int(tf.size(v)) for v in model.trainable_variables)
    non_trainable = sum(int(tf.size(v)) for v in model.non_trainable_variables)
    print(f"[INFO] Trainable vars:     {trainable:,}")
    print(f"[INFO] Non-trainable vars: {non_trainable:,}")


def _forward_pass_smoke_test(
    model: tf.keras.Model,
    batch_size: int = 4,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> None:
    """
    Szybki test: tworzymy sztuczny batch (0..255) i sprawdzamy
    czy model daje wyjście o poprawnym ksztalcie (batch_size, 1).
    """
    x = tf.random.uniform(
        shape=(batch_size, *input_shape),
        minval=0.0,
        maxval=255.0,
        dtype=tf.float32,
    )
    y = model(x, training=False)
    print("\n[SMOKE TEST] Forward pass OK")
    print(f"[SMOKE TEST] Input shape : {x.shape}")
    print(f"[SMOKE TEST] Output shape: {y.shape}")
    print(f"[SMOKE TEST] Output dtype: {y.dtype}")
    # Dla sigmoid spodziewamy się wartości w [0,1]
    y_min = float(tf.reduce_min(y).numpy())
    y_max = float(tf.reduce_max(y).numpy())
    print(f"[SMOKE TEST] Output range ~ [{y_min:.4f}, {y_max:.4f}]")


def main() -> None:
    # (Opcjonalnie) upewniamy sie ze uruchamiamy z katalogu projektu bo to wazne
    project_root = Path(__file__).resolve().parent.parent
    print(f"[INFO] Project root: {project_root}")

    # 1) BASELINE CNN 
    baseline = build_baseline_cnn(
        input_shape=(224, 224, 3),
        dense_units=64,
        dropout_rate=0.4,  # zgodnie z ustaleniami
        name="baseline_cnn_flatten",
    )
    _print_model_info(baseline, "MODEL 1: Baseline CNN (Flatten)")
    _forward_pass_smoke_test(baseline, batch_size=4, input_shape=(224, 224, 3))

if __name__ == "__main__":
    main()
