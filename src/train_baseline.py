from __future__ import annotations
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import tensorflow as tf

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_loader import get_datasets  # noqa: E402
from models import build_baseline_cnn  # noqa: E402



# Konfiguracja treningu

@dataclass(frozen=True)
class TrainConfig:
    # Model
    input_shape: tuple[int, int, int] = (224, 224, 3)
    dense_units: int = 64
    dropout_rate: float = 0.4

    # Trening
    batch_size: int = 32  # pamietamy o tym ze batch_size jest ustawiany w data_loader.py (domyslnie 32)
    epochs: int = 20
    learning_rate: float = 1e-4

    # Callbacki
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 5
    restore_best_weights: bool = True

    checkpoint_monitor: str = "val_loss"
    save_best_only: bool = True

    seed: int = 42

    # Runs dir
    runs_dir: str = "runs"

    # TensorBoard
    enable_tensorboard: bool = True


class EpochTimeLogger(tf.keras.callbacks.Callback):
    """Prosty callback do pomiaru czasu epok"""
    def on_train_begin(self, logs=None):
        self.epoch_times: list[float] = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.perf_counter() - self._t0
        self.epoch_times.append(dt)
        print(f"[TIME] Epoch {epoch+1} took {dt:.2f}s")


def ensure_dirs(run_root: Path) -> Dict[str, Path]:
    paths = {
        "run_root": run_root,
        "models": run_root / "models",
        "logs": run_root / "logs",
        "history": run_root / "history",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_history_json(history: Dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_history_csv(history: Dict[str, Any], out_path: Path) -> None:
    # Zapisujemy w formie kolumny = metryki i wiersze = epoki
    keys = list(history.keys())
    n_epochs = len(history[keys[0]]) if keys else 0

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(n_epochs):
            row = [i + 1] + [history[k][i] for k in keys]
            writer.writerow(row)


def main() -> None:
    cfg = TrainConfig()

    # Seed
    tf.keras.utils.set_random_seed(cfg.seed)

    # Tworzymy folder run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(cfg.runs_dir) / f"baseline_{run_id}"
    paths = ensure_dirs(run_root)

    print("== TRAIN BASELINE CNN ==")
    print(f"[INFO] run_id: {run_id}")
    print(f"[INFO] run_root: {paths['run_root'].resolve()}")
    print(f"[INFO] seed: {cfg.seed}")
    print(f"[INFO] epochs(max): {cfg.epochs}")
    print(f"[INFO] learning_rate: {cfg.learning_rate}")
    print(f"[INFO] dropout_rate: {cfg.dropout_rate}")
    print(f"[INFO] dense_units: {cfg.dense_units}\n")

    # Zapis konfiguracji do pliku (tak juz raportowo)
    cfg_path = paths["run_root"] / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    # 1) Wczytanie danych przez loader
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print(f"[INFO] class_names: {class_names}")

    # Diagnostyka liczności (len(ds) bedzie dzialac bo dataset jest batched)
    try:
        train_batches = len(train_ds)
        val_batches = len(val_ds)
        test_batches = len(test_ds)
        print(f"[INFO] batches: train={train_batches}, val={val_batches}, test={test_batches}")
        print(f"[INFO] approx samples (batches * batch_size): "
              f"train~{train_batches*cfg.batch_size}, val~{val_batches*cfg.batch_size}, test~{test_batches*cfg.batch_size}\n")
    except TypeError:
        print("[WARN] Nie mozna policzyc len(dataset). Pomijamy diagnostyke licznosci.\n")

    # 2) Zbudowanie modelu zgodnie z ustaleniami
    model = build_baseline_cnn(
        input_shape=cfg.input_shape,
        dense_units=cfg.dense_units,
        dropout_rate=cfg.dropout_rate,
        name="baseline_cnn_flatten",
    )
    model.summary()

    # 3) model.compile() z ustawieniami
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    # 4) Callbacki: EarlyStopping + ModelCheckpoint (+ ten nice-to-have TensorBoard)
    callbacks: list[tf.keras.callbacks.Callback] = []

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=cfg.early_stopping_monitor,
            patience=cfg.early_stopping_patience,
            restore_best_weights=cfg.restore_best_weights,
            verbose=1,
        )
    )

    best_model_path = paths["models"] / "best_model.keras"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=cfg.checkpoint_monitor,
            save_best_only=cfg.save_best_only,
            verbose=1,
        )
    )

    epoch_time_logger = EpochTimeLogger()
    callbacks.append(epoch_time_logger)

    if cfg.enable_tensorboard:
        tb_log_dir = paths["logs"] / "tensorboard"
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tb_log_dir),
                histogram_freq=0,
                write_graph=True,
                update_freq="epoch",
            )
        )
        print(f"[INFO] TensorBoard logs: {tb_log_dir.resolve()}")
        print("       uruchom: tensorboard --logdir runs\n")

    # 5) Trening model.fit(...) juz na pelnych epokach
    t0 = time.perf_counter()
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    total_time = time.perf_counter() - t0
    print(f"\n[TIME] Total training time: {total_time:.2f}s")

    # 6) Ewaluacja po treningu
    print("\n== EVALUATION ==")
    val_metrics = model.evaluate(val_ds, verbose=1, return_dict=True)
    test_metrics = model.evaluate(test_ds, verbose=1, return_dict=True)

    print("\n[RESULT] Validation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[RESULT] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 7) Zapis historii i wyników
    history = history_obj.history
    history_json_path = paths["history"] / "history.json"
    history_csv_path = paths["history"] / "history.csv"
    save_history_json(history, history_json_path)
    save_history_csv(history, history_csv_path)

    results_path = paths["history"] / "final_metrics.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "val": val_metrics,
                "test": test_metrics,
                "total_training_time_sec": total_time,
                "epoch_times_sec": getattr(epoch_time_logger, "epoch_times", []),
                "best_model_path": str(best_model_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Zapis modelu koncowego (po EarlyStopping restore_best_weights)
    final_model_path = paths["models"] / "final_model.keras"
    model.save(final_model_path)
    print(f"\n[INFO] Saved best model to:  {best_model_path}")
    print(f"[INFO] Saved final model to: {final_model_path}")
    print(f"[INFO] Saved history to:     {history_json_path} and {history_csv_path}")
    print(f"[INFO] Saved metrics to:     {results_path}")

    # Print koncowych metryk w skrocie
    def _last(metric_name: str) -> float | None:
        arr = history.get(metric_name)
        return float(arr[-1]) if arr else None

    print("\n== FINAL (last epoch) ==")
    for key in ["loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc"]:
        val = _last(key)
        if val is not None:
            print(f"{key}: {val:.4f}")

    print("\n== DONE ==")


if __name__ == "__main__":
    main()
