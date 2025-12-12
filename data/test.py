from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = Path("data")
EXPECTED_SIZE = (224, 224)
EXPECTED_MODE = "RGB"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def main():
    subsets = ["train", "val", "test"]

    print("\nWERYFIKACJA ZBIORU DANYCH\n")

    for subset in subsets:
        subset_dir = DATA_DIR / subset
        if not subset_dir.exists():
            print(f"Zbior {subset} nie istnieje ({subset_dir})")
            continue

        print(f"--- {subset.upper()} ---")

        for class_dir in sorted(d for d in subset_dir.iterdir() if d.is_dir()):
            cls = class_dir.name

            total_count = 0
            wrong_size = []
            wrong_mode = []

            print(f"\nSprawdzanie klasy {cls}...")

            for img_path in class_dir.iterdir():
                if not img_path.is_file() or not is_image_file(img_path):
                    continue

                total_count += 1

                if total_count % 100 == 0:
                    print(f"  [POSTEP] sprawdzono {total_count} obrazow w '{cls}'")

                try:
                    with Image.open(img_path) as img:
                        size_ok = img.size == EXPECTED_SIZE
                        mode_ok = img.mode == EXPECTED_MODE
                except Exception:
                    size_ok = False
                    mode_ok = False

                if not size_ok:
                    wrong_size.append(img_path)
                if not mode_ok:
                    wrong_mode.append(img_path)

            # Podsumowanie
            print(f"\n{cls}: {total_count} obrazow")

            if wrong_size or wrong_mode:
                print("NIEPOPRAWNE OBRAZY:")
                if wrong_size:
                    print(f"    - zly rozmiar (nie 224x224): {len(wrong_size)}")
                    for p in wrong_size[:3]:
                        print(f"      * {p.name}")
                if wrong_mode:
                    print(f"    - zly tryb (nie RGB): {len(wrong_mode)}")
                    for p in wrong_mode[:3]:
                        print(f"      * {p.name}")
            else:
                print("Wszystkie obrazy są 224x224 i w RGB")

        print()

    print("KONIEC WERYFIKACJI\n")

if __name__ == "__main__":
    main()
