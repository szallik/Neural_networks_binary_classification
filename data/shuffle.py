
import os
import shutil
import random
from collections import defaultdict

# Sciezki do ustawienia
ORIGINAL_DATASET_PATH = ""
NEW_DATASET_PATH = ""

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]


def count_images_per_split_and_class(root_path):
    counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}

    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(root_path, split, cls)
            if os.path.exists(folder):
                files = [
                    f for f in os.listdir(folder)
                    if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))
                ]
                counts[split][cls] = len(files)

    return counts


def sum_over_splits(counts):
    total = defaultdict(int)
    for split in counts:
        for cls in counts[split]:
            total[cls] += counts[split][cls]
    return dict(total)


def print_counts(title, counts):
    print(f"\n=== {title} ===")
    for split in SPLITS:
        print(split.upper() + ":")
        for cls in CLASSES:
            print(f"  {cls}: {counts[split][cls]}")
    total = sum_over_splits(counts)
    print("SUMA (po wszystkich splitach):")
    for cls in CLASSES:
        print(f"  {cls}: {total[cls]}")
    print(f"  Razem: {sum(total.values())}")


def main():
    # 1. Liczymy dane w oryginalnym zbiorze
    print("Liczenie obrazow w ORYGINALNYM zbiorze...")
    original_counts = count_images_per_split_and_class(ORIGINAL_DATASET_PATH)
    print_counts("ORYGINALNY ZBIOR", original_counts)

    # 2. Tworzymy nowy zbior tylko jesli jeszcze nie istnieje
    random.seed(42)  # powtarzalny podział

    if os.path.exists(NEW_DATASET_PATH):
        print(f"\nFolder '{NEW_DATASET_PATH}' juz istnieje.")
        return

    # tworzymy strukture katalogow
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(NEW_DATASET_PATH, split, cls), exist_ok=True)

    print("\nTworzenie nowego podziału 70/15/15...")

    # 3. Dla kazdej klasy osobno zbieramy wszystkie pliki z train/val/test i dzielimy
    for cls in CLASSES:
        print(f"\nPrzetwarzanie klasy: {cls}")
        all_files = []

        for split in SPLITS:
            source_folder = os.path.join(ORIGINAL_DATASET_PATH, split, cls)
            if not os.path.exists(source_folder):
                continue

            files = [
                f for f in os.listdir(source_folder)
                if not f.startswith(".") and os.path.isfile(os.path.join(source_folder, f))
            ]
            all_files.extend((f, source_folder) for f in files)

        n = len(all_files)
        print(f"Liczba wszystkich obrazow klasy {cls}: {n}")

        random.shuffle(all_files)

        # 70% train 15% val 15% test
        train_split = int(n * 0.7)
        val_split = int(n * 0.85)

        train_files = all_files[:train_split]
        val_files = all_files[train_split:val_split]
        test_files = all_files[val_split:]

        print(f"  -> train: {len(train_files)}")
        print(f"  -> val:   {len(val_files)}")
        print(f"  -> test:  {len(test_files)}")

        for file, source_folder in train_files:
            src = os.path.join(source_folder, file)
            dst = os.path.join(NEW_DATASET_PATH, "train", cls, file)
            shutil.copy(src, dst)

        for file, source_folder in val_files:
            src = os.path.join(source_folder, file)
            dst = os.path.join(NEW_DATASET_PATH, "val", cls, file)
            shutil.copy(src, dst)

        for file, source_folder in test_files:
            src = os.path.join(source_folder, file)
            dst = os.path.join(NEW_DATASET_PATH, "test", cls, file)
            shutil.copy(src, dst)

    print("\nNowy podział utworzony w:", NEW_DATASET_PATH)

    # 4. Liczymy dane w nowym zbiorze
    new_counts = count_images_per_split_and_class(NEW_DATASET_PATH)
    print_counts("NOWY ZBIOR", new_counts)


if __name__ == "__main__":
    main()
