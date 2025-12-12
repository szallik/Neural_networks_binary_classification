import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# SCIEZKI DO ZBIOROW
# 1) ORYGINALNY zbiór wykres "przed"
# 2) NOWY zbiór wykres "po"
DATASET_PATH = ""

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]

plt.style.use("ggplot")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def count_images_per_split_and_class(root_path):
    counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}

    for split in SPLITS:
        for cls in CLASSES:
            folder = os.path.join(root_path, split, cls)
            if os.path.exists(folder):
                files = [
                    f for f in os.listdir(folder)
                    if not f.startswith(".")
                    and os.path.isfile(os.path.join(folder, f))
                ]
                counts[split][cls] = len(files)

    return counts


def sum_over_splits(counts):
    total = defaultdict(int)
    for split in counts:
        for cls in counts[split]:
            total[cls] += counts[split][cls]
    return dict(total)


def make_plots(root_path):
    counts = count_images_per_split_and_class(root_path)
    total_counts = sum_over_splits(counts)

    # maksymalna liczba obrazów w jakimkolwiek splicie do ujednolicenia osi Y
    max_y = 0
    for split in SPLITS:
        split_sum = counts[split]["NORMAL"] + counts[split]["PNEUMONIA"]
        max_y = max(max_y, split_sum)
    max_y = max(max_y, total_counts["NORMAL"] + total_counts["PNEUMONIA"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.ravel()

    colors = ["#4C72B0", "#55A868"]

    dataset_name = os.path.basename(os.path.normpath(root_path))
    fig.suptitle(f"Class distribution – {dataset_name}",
                 fontsize=14, fontweight="bold")

    # 3 pierwsze wykresy train / val / test 
    split_titles = [("train", "Train"),
                    ("val", "Validation"),
                    ("test", "Test")]

    for idx, (split, title) in enumerate(split_titles):
        ax = axes[idx]

        normal = counts[split]["NORMAL"]
        pneu = counts[split]["PNEUMONIA"]
        values = [normal, pneu]
        total_split = normal + pneu if (normal + pneu) > 0 else 1

        percents = [v / total_split * 100 for v in values]
        x = np.arange(2)

        bars = ax.bar(x, values, color=colors)
        ax.set_title(title)

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"Normal\n{percents[0]:.1f}%",
            f"Pneumonia\n{percents[1]:.1f}%"
        ])

        ax.set_ylim(0, max_y * 1.1)

        # liczby nad słupkami
        for bar, v in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    height + max_y * 0.02,
                    f"{v}",
                    ha="center", va="bottom", fontsize=9)

    # 4. wykres suma po wszystkich splitach 
    ax = axes[3]

    normal = total_counts["NORMAL"]
    pneu = total_counts["PNEUMONIA"]
    values = [normal, pneu]
    total_all = normal + pneu if (normal + pneu) > 0 else 1
    percents = [v / total_all * 100 for v in values]
    x = np.arange(2)

    bars = ax.bar(x, values, color=colors)
    ax.set_title("Suma")

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Normal\n{percents[0]:.1f}%",
        f"Pneumonia\n{percents[1]:.1f}%"
    ])

    ax.set_ylim(0, max_y * 1.1)

    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                height + max_y * 0.02,
                f"{v}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88) 
    plt.show()


if __name__ == "__main__":
    make_plots(DATASET_PATH)
