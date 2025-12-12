from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IMG_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
SEED: int = 42


def _load_split(
    split_dir: Path,
    img_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Wczytuje pojedynczy podzbior danych (train/val/test) jako tf.data.Dataset
    z katalogu o strukturze:
        split_dir/
            NORMAL/
            PNEUMONIA/
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",       # bo mamy dwie klasy
        color_mode="rgb",          # obrazy RGB
        batch_size=batch_size,
        image_size=img_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )
    return dataset


def get_datasets(
    data_dir: Path = DATA_DIR,
    img_size: Tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Zwraca trzy zbiory danych (train, val, test) jako tf.data.Dataset
    + liste nazw klas w kolejnosci uzywanej przez TensorFlow.

    Kazdy z nich:
    - ma obrazy w kształcie (batch_size, img_size[0], img_size[1], 3)
    - ma etykiety binarne (0. lub 1.)
    """

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
        raise FileNotFoundError(
            f"Nie znaleziono katalogow train/val/test w {data_dir}. "
            f"Sprawdz sciezki do danych."
        )

    # Wczytanie trzech podzbiorow
    train_ds = _load_split(
        train_dir, img_size=img_size, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_ds = _load_split(
        val_dir, img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
    )
    test_ds = _load_split(
        test_dir, img_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
    )

    # Sprawdzenie spojnosci nazw klas
    class_names = train_ds.class_names
    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError(
            "Niespojne nazwy klas miedzy zbiorem train/val/test. "
            f"train: {train_ds.class_names}, "
            f"val: {val_ds.class_names}, "
            f"test: {test_ds.class_names}"
        )

    # Usprawnienia wydajności (prefetch)
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names
