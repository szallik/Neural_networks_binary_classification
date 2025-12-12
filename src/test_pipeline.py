from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf

# Pozwala uruchamiac ten plik z poziomu katalogu glownego projektu
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_loader import get_datasets


def main() -> None:
    print("TEST PIPELINE DANYCH")

    # Wczytanie zbiorow
    train_ds, val_ds, test_ds, class_names = get_datasets()

    print("\n[INFO] Wczytano zbiory danych przez data_loader.get_datasets().")
    print(f"[INFO] Nazwy klas: {class_names}")
    print("       (kolejnosc ta sama, ktorej uzywa TensorFlow przy etykietowaniu)\n")

    # Liczba batchy w kazdym zbiorze
    try:
        num_train_batches = len(train_ds)
        num_val_batches = len(val_ds)
        num_test_batches = len(test_ds)

        print("[INFO] Liczba batchy:")
        print(f"       train: {num_train_batches}")
        print(f"       val:   {num_val_batches}")
        print(f"       test:  {num_test_batches}\n")
    except TypeError:
        print("[UWAGA] Nie mozna obliczyc len(dataset) dla ktoregos ze zbiorow.")
        print("        To nie jest blad krytyczny ale warto wiedziec.\n")

    # Pobranie jednego batcha ze zbioru treningowego
    print("[INFO] Sprawdzam pojedynczy batch ze zbioru treningowego...")
    train_batch = next(iter(train_ds))
    images, labels = train_batch

    print(f"       Ksztalt obrazow: {images.shape}")   # (batch_size, 224, 224, 3)
    print(f"       Ksztalt etykiet: {labels.shape}")   # (batch_size, 1)
    print(f"       Typ danych obrazow: {images.dtype}")
    print(f"       Typ danych etykiet: {labels.dtype}")

    # Sprawdzenie przykładowych wartości
    images_np = images.numpy()
    labels_np = labels.numpy()

    print(f"       Zakres pikseli w batchu: "
          f"min={images_np.min():.1f}, max={images_np.max():.1f}")
    print(f"       Przykładowe etykiety (pierwsze 10): {labels_np[:10].reshape(-1)}\n")

    # Szybkie podsumowanie liczby próbek:
    # (bazuje na liczbie batchy i wielkosci batcha, ostatni batch może byc mniejszy)
    batch_size = images.shape[0]
    print("[INFO] Szacunkowa liczba probek (batchy * batch_size):")
    print(f"       train ~= {len(train_ds) * batch_size}")
    print(f"       val   ~= {len(val_ds) * batch_size}")
    print(f"       test  ~= {len(test_ds) * batch_size}\n")

    print("TEST PIPELINE ZAKONCZONY POMYSLNIE")


if __name__ == "__main__":
    main()
