"""
Finds the actual class folders inside a messy/nested raw dataset download
(e.g. Kaggle's "archive" folder) and splits them 80/20 into:

    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg

Usage:
    python split_dataset.py --source "C:\\Users\\user\\Downloads\\archive" --dest data --val-ratio 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_folders(source_dir: Path):
    """
    A 'class folder' is any directory that directly contains image files.
    This handles nested layouts like archive/PlantVillage/color/<class>/*.jpg
    without needing to know the exact structure in advance.
    """
    class_folders = []
    for path in source_dir.rglob("*"):
        if path.is_dir():
            images = [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
            if images:
                class_folders.append((path, images))
    return class_folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to the raw downloaded dataset folder")
    parser.add_argument("--dest", type=str, default="data", help="Destination data/ folder")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    class_folders = find_class_folders(source_dir)
    if not class_folders:
        raise RuntimeError(
            f"No folders with images found under {source_dir}. "
            "Double check the path — it should point at (or above) the folder containing your class subfolders."
        )

    print(f"Found {len(class_folders)} class folders:")
    for folder, images in class_folders:
        print(f"  {folder.name}: {len(images)} images")

    train_dir = Path(args.dest) / "train"
    val_dir = Path(args.dest) / "val"

    for folder, images in class_folders:
        class_name = folder.name
        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        val_images = images[:n_val]
        train_images = images[n_val:]

        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)

        print(f"{class_name}: {len(train_images)} train / {len(val_images)} val")

    print(f"\nDone. Data ready at {train_dir} and {val_dir}")


if __name__ == "__main__":
    main()
