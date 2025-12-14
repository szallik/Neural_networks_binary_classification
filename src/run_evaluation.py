from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
PLOTS_DIR = PROJECT_ROOT / "plots"


def _find_latest_run(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Brak katalogu runs/: {runs_dir}")

    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Brak runow w {runs_dir}")

    # Najnowszy po czasie modyfikacji katalogu
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_run_dir(run_arg: Optional[str]) -> Path:
    if run_arg:
        run_dir = RUNS_DIR / run_arg
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Nie znaleziono runa: {run_dir}")
        return run_dir
    return _find_latest_run(RUNS_DIR)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_history(run_dir: Path) -> pd.DataFrame:
    """
    Wczytuje historie treningu jako DataFrame.
    Preferuje history.csv, a jesli brak to history.json.
    """
    hist_dir = run_dir / "history"
    csv_path = hist_dir / "history.csv"
    json_path = hist_dir / "history.json"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # history.csv ma kolumne epoch — upewniamy sie ze jest
        if "epoch" not in df.columns:
            df.insert(0, "epoch", range(1, len(df) + 1))
        return df

    if json_path.exists():
        hist = _load_json(json_path)
        df = pd.DataFrame(hist)
        df.insert(0, "epoch", range(1, len(df) + 1))
        return df

    raise FileNotFoundError(f"Brak history.csv i history.json w {hist_dir}")


def _best_epoch_by_val_loss(history_df: pd.DataFrame) -> Tuple[int, float]:
    if "val_loss" not in history_df.columns:
        return -1, float("nan")
    idx = history_df["val_loss"].idxmin()
    best_epoch = int(history_df.loc[idx, "epoch"])
    best_val_loss = float(history_df.loc[idx, "val_loss"])
    return best_epoch, best_val_loss


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "N/A"


def _get_metric(d: Dict[str, Any], key: str) -> Any:
    return d.get(key, "N/A")


def _plot_two_curves(
    history_df: pd.DataFrame,
    train_col: str,
    val_col: str,
    title: str,
    out_path: Path,
) -> None:
    if train_col not in history_df.columns or val_col not in history_df.columns:
        print(f"[WARN] Brak kolumn do wykresu: {train_col} / {val_col}. Skip {title}.")
        return

    epochs = history_df["epoch"].to_list()
    train_vals = history_df[train_col].to_list()
    val_vals = history_df[val_col].to_list()

    plt.figure()
    plt.plot(epochs, train_vals, label=train_col)
    plt.plot(epochs, val_vals, label=val_col)
    plt.xlabel("Epoch")
    plt.ylabel(train_col.replace("val_", ""))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic run evaluation.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Nazwa katalogu runa w runs/, np. baseline_20251213_192118. Jesli brak -> najnowszy run.",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run)
    run_id = run_dir.name

    # Artefakty
    config_path = run_dir / "config.json"
    final_metrics_path = run_dir / "history" / "final_metrics.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Brak config.json: {config_path}")
    if not final_metrics_path.exists():
        raise FileNotFoundError(f"Brak final_metrics.json: {final_metrics_path}")

    config = _load_json(config_path)
    history_df = _load_history(run_dir)
    final_metrics = _load_json(final_metrics_path)

    best_epoch, best_val_loss = _best_epoch_by_val_loss(history_df)

    # Czasy
    total_time = final_metrics.get("total_training_time_sec", None)
    epoch_times = final_metrics.get("epoch_times_sec", [])
    avg_epoch = (sum(epoch_times) / len(epoch_times)) if epoch_times else None

    # Metryki końcowe
    val = final_metrics.get("val", {})
    test = final_metrics.get("test", {})


    # RAPORT
    print("\n" + "=" * 72)
    print("RUN EVALUATION (basic)")
    print("=" * 72)
    print(f"Run:              {run_id}")
    print(f"Run path:         {run_dir.resolve()}")
    print(f"Model name:       {config.get('input_shape', 'N/A')} | baseline_cnn_flatten")
    print("-" * 72)
    print("Config (kluczowe):")
    print(f"  epochs(max):    {config.get('epochs', 'N/A')}")
    print(f"  lr:            {config.get('learning_rate', 'N/A')}")
    print(f"  dropout:       {config.get('dropout_rate', 'N/A')}")
    print(f"  dense_units:   {config.get('dense_units', 'N/A')}")
    print(f"  seed:          {config.get('seed', 'N/A')}")
    print("-" * 72)
    print("Best epoch wg val_loss:")
    print(f"  best_epoch:    {best_epoch}")
    print(f"  best_val_loss: {_fmt(best_val_loss)}")
    print("-" * 72)
    print("Final metrics (VAL):")
    print(f"  loss:          {_fmt(_get_metric(val, 'loss'))}")
    print(f"  accuracy:      {_fmt(_get_metric(val, 'accuracy'))}")
    print(f"  auc:           {_fmt(_get_metric(val, 'auc'))}")
    print("Final metrics (TEST):")
    print(f"  loss:          {_fmt(_get_metric(test, 'loss'))}")
    print(f"  accuracy:      {_fmt(_get_metric(test, 'accuracy'))}")
    print(f"  auc:           {_fmt(_get_metric(test, 'auc'))}")
    print("-" * 72)
    print("Timing:")
    print(f"  total_train_s: {_fmt(total_time)}")
    print(f"  avg_epoch_s:   {_fmt(avg_epoch)}")
    print(f"  epochs_ran:    {len(history_df)}")
    print("=" * 72 + "\n")


    # WYKRESY (do globalnego plots/)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    loss_path = PLOTS_DIR / f"{run_id}_loss.png"
    auc_path = PLOTS_DIR / f"{run_id}_auc.png"

    _plot_two_curves(
        history_df,
        train_col="loss",
        val_col="val_loss",
        title=f"{run_id} | Loss (train vs val)",
        out_path=loss_path,
    )
    _plot_two_curves(
        history_df,
        train_col="auc",
        val_col="val_auc",
        title=f"{run_id} | AUC (train vs val)",
        out_path=auc_path,
    )

    print("[INFO] Saved plots:")
    print(f"  {loss_path.resolve()}")
    print(f"  {auc_path.resolve()}")
    print("")


if __name__ == "__main__":
    main()
