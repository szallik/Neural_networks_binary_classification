import os
from pathlib import Path
from PIL import Image

# SCIEZKI 
SOURCE = ""
DEST   = ""

# PARAMETRY 
TARGET_SIZE = (224, 224)   # docelowy rozmiar obrazów
COLOR_MODE = "RGB"        

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]


def preprocess_image(src: Path, dst: Path):

    try:
        with Image.open(src) as img:
            # 1. Ujednolicenie kanałów
            img = img.convert(COLOR_MODE)

            # 2. Przeskalowanie
            img = img.resize(TARGET_SIZE, Image.BILINEAR)

            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst)
    except Exception as e:
        print(f"Blad przetwarzania {src}: {e}")


def preprocess_dataset():
    source_root = Path(SOURCE)
    dest_root = Path(DEST)

    print("PREPROCESSING OBRAZOW ")
    print(f"Zrodlo: {source_root}")
    print(f"Cel:    {dest_root}")
    print(f"Rozmiar docelowy: {TARGET_SIZE}, tryb: {COLOR_MODE}\n")

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = source_root / split / cls
            dst_dir = dest_root / split / cls

            if not src_dir.exists():
                print(f"Pomijam brakujacy katalog: {src_dir}")
                continue

            print(f"Przetwarzanie: {split}/{cls}")

            files = [
                f for f in src_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]

            for i, src_img in enumerate(files, start=1):
                dst_img = dst_dir / src_img.name
                preprocess_image(src_img, dst_img)

                if i % 500 == 0:
                    print(f"  ...przetworzono {i} obrazow")

    print("\nZakonczono preprocessing.")
    print("Nowy przeskalowany zbior dostepny w folderze:")
    print(dest_root)


if __name__ == "__main__":
    preprocess_dataset()
