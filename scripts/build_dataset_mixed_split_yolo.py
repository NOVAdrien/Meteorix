"""
Build a mixed YOLO dataset from two separate YOLO datasets:
- one dataset with meteor images
- one dataset without meteor images

Each source dataset is expected to look like this:

dataset_with_meteor/
  data.yaml
  train/
    images/
      image_001.jpg
      ...
    labels/
      image_001.txt
      ...

dataset_no_meteor/
  data.yaml
  train/
    images/
      image_abc.jpg
      ...
    labels/
      image_abc.txt
      ...

The script creates a new YOLO dataset:

mixed_dataset/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/

This version does NOT compute split sizes from percentages.
The number of images per split is defined manually below.

Required final split:

  train:
    with_meteor : 2694
    no_meteor   : 1155
    total       : 3849

  valid:
    with_meteor : 385
    no_meteor   : 165
    total       : 550

  test:
    with_meteor : 770
    no_meteor   : 330
    total       : 1100

Total:
  with_meteor : 3849
  no_meteor   : 1650
  total       : 5499

Usage:
1. Update the CONFIGURATION section below if needed.
2. Run:
       python build_dataset_mixed_yolo_manual_split.py
"""

import random
import shutil
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "../data/datasets/annotated/"
DATASET_DIR = DATASET_DIR.resolve()

# Root folder of the Roboflow YOLO export containing images WITH meteor
SOURCE_WITH_METEOR = DATASET_DIR / "Meteorix_Mendeley_dataset_with_meteor_annotated_formatted.yolo26"

# Root folder of the Roboflow YOLO export containing images WITHOUT meteor
SOURCE_NO_METEOR = DATASET_DIR / "Meteorix_Mendeley_dataset_no_meteor_annotated_formatted.yolo26"

# Output folder for the final mixed dataset
DESTINATION = DATASET_DIR / "Meteorix_Mendeley_dataset_mixed_annotated_formatted_split_2.yolo26"

# Random seed for reproducible sampling.
# Change this value if you want a different random selection.
RANDOM_SEED = 42

# YOLO class names.
# For object detection with a single meteor class, keep ["meteor"].
CLASS_NAMES = ["meteor"]

# Accepted image extensions
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# Manual split counts.
# No ratio calculation is used anywhere in this script.
SPLIT_COUNTS_WITH_METEOR = {
    "train": 2694,
    "valid": 385,
    "test": 770,
}

SPLIT_COUNTS_NO_METEOR = {
    "train": 1155,
    "valid": 165,
    "test": 330,
}

# Total number of images to sample from each source dataset.
N_WITH_METEOR = sum(SPLIT_COUNTS_WITH_METEOR.values())
N_NO_METEOR = sum(SPLIT_COUNTS_NO_METEOR.values())


# ============================================================
# FUNCTIONS
# ============================================================

def find_split_folder(dataset_root: Path) -> Path:
    """
    Find the folder that contains images/ and labels/.

    In this project, it should normally be:
        dataset_root/train/

    The function checks train/ first, then searches recursively if needed.
    """
    direct_train = dataset_root / "train"
    if (direct_train / "images").is_dir() and (direct_train / "labels").is_dir():
        return direct_train

    candidates = [
        p for p in dataset_root.rglob("*")
        if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a folder containing images/ and labels/ in: {dataset_root}"
        )

    if len(candidates) > 1:
        print(f"Warning: multiple candidate folders found in {dataset_root}.")
        print(f"Using: {candidates[0]}")

    return candidates[0]


def list_image_label_pairs(dataset_root: Path, category_name: str):
    """
    List image + label pairs.

    For each image, the expected label file is:
        labels/image_name.txt

    If the label file does not exist, the script will create an empty label later.
    This is valid for YOLO negative images, such as images without meteor.
    """
    split_folder = find_split_folder(dataset_root)
    images_dir = split_folder / "images"
    labels_dir = split_folder / "labels"

    image_files = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    pairs = []

    for img_path in image_files:
        relative_img = img_path.relative_to(images_dir)
        label_path = labels_dir / relative_img.with_suffix(".txt")

        pairs.append({
            "image": img_path,
            "label": label_path if label_path.exists() else None,
            "category": category_name,
        })

    return pairs


def ensure_enough_items(items, required_count, label):
    """Check that the source dataset contains enough images."""
    available = len(items)

    if available < required_count:
        raise ValueError(
            f"Not enough images for '{label}'. "
            f"Required: {required_count}, available: {available}"
        )


def validate_manual_split_counts():
    """Validate that manual split dictionaries are well-formed."""
    expected_splits = {"train", "valid", "test"}

    with_keys = set(SPLIT_COUNTS_WITH_METEOR.keys())
    no_keys = set(SPLIT_COUNTS_NO_METEOR.keys())

    if with_keys != expected_splits:
        raise ValueError(
            f"SPLIT_COUNTS_WITH_METEOR must contain exactly these keys: {expected_splits}"
        )

    if no_keys != expected_splits:
        raise ValueError(
            f"SPLIT_COUNTS_NO_METEOR must contain exactly these keys: {expected_splits}"
        )

    for split_name, count in SPLIT_COUNTS_WITH_METEOR.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid count for with_meteor/{split_name}: {count}")

    for split_name, count in SPLIT_COUNTS_NO_METEOR.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid count for no_meteor/{split_name}: {count}")


def check_destination(destination: Path):
    """
    Safety check:
    If the destination folder already exists and is not empty,
    the script stops to avoid mixing two datasets.
    """
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"The destination folder already exists and is not empty: {destination}\n"
            "Delete it, empty it, or change DESTINATION in the script."
        )

    destination.mkdir(parents=True, exist_ok=True)


def make_yolo_folders(destination: Path):
    """Create train/valid/test YOLO folders."""
    for split in ["train", "valid", "test"]:
        (destination / split / "images").mkdir(parents=True, exist_ok=True)
        (destination / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path):
    """Copy a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_output_names(item, split_name, used_names):
    """
    Create a unique output file name.

    This version keeps the original image and label names.
    It only adds _1, _2, etc. if there is a filename collision
    inside the same split.
    """
    base_name = item["image"].stem
    image_suffix = item["image"].suffix.lower()

    candidate_image_name = f"{base_name}{image_suffix}"
    candidate_label_name = f"{base_name}.txt"

    counter = 1
    while (split_name, candidate_image_name) in used_names:
        candidate_image_name = f"{base_name}_{counter}{image_suffix}"
        candidate_label_name = f"{base_name}_{counter}.txt"
        counter += 1

    used_names.add((split_name, candidate_image_name))

    return candidate_image_name, candidate_label_name


def write_empty_label(label_destination: Path):
    """Create an empty YOLO label file."""
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.write_text("", encoding="utf-8")


def place_items(items, split_name, destination: Path, used_names):
    """
    Copy or move images and labels into the target split.

    If an image has no .txt label, an empty .txt label is created.
    """
    for item in items:
        image_name, label_name = safe_output_names(item, split_name, used_names)

        image_destination = destination / split_name / "images" / image_name
        label_destination = destination / split_name / "labels" / label_name

        copy_file(item["image"], image_destination)

        if item["label"] is not None:
            copy_file(item["label"], label_destination)
        else:
            write_empty_label(label_destination)


def split_items(items, split_counts):
    """Split an already-shuffled list using manual counts."""
    train_count = split_counts["train"]
    valid_count = split_counts["valid"]
    test_count = split_counts["test"]

    train_items = items[:train_count]
    valid_items = items[train_count:train_count + valid_count]
    test_items = items[
        train_count + valid_count:
        train_count + valid_count + test_count
    ]

    return {
        "train": train_items,
        "valid": valid_items,
        "test": test_items,
    }


def write_data_yaml(destination: Path):
    """
    Write a YOLO-compatible data.yaml file.

    Format:
      train: ../train/images
      val: ../valid/images
      test: ../test/images

      nc: 1
      names: ['meteor']
    """
    names_yaml = "[" + ", ".join([f"'{name}'" for name in CLASS_NAMES]) + "]"

    content = (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {names_yaml}\n"
    )

    (destination / "data.yaml").write_text(content, encoding="utf-8")


def write_summary(destination: Path, split_with_meteor, split_no_meteor):
    """Write a readable summary of the final dataset."""
    lines = []
    lines.append("YOLO meteor dataset summary")
    lines.append("=" * 40)
    lines.append("")

    total_with = 0
    total_no = 0

    for split in ["train", "valid", "test"]:
        n_with = len(split_with_meteor[split])
        n_no = len(split_no_meteor[split])
        total_split = n_with + n_no

        total_with += n_with
        total_no += n_no

        ratio_with = 100 * n_with / total_split if total_split else 0
        ratio_no = 100 * n_no / total_split if total_split else 0

        lines.append(f"{split}:")
        lines.append(f"  with_meteor : {n_with}")
        lines.append(f"  no_meteor   : {n_no}")
        lines.append(f"  total       : {total_split}")
        lines.append(f"  ratio       : {ratio_with:.2f}% with / {ratio_no:.2f}% without")
        lines.append("")

    total = total_with + total_no
    ratio_with_total = 100 * total_with / total if total else 0
    ratio_no_total = 100 * total_no / total if total else 0

    lines.append("TOTAL:")
    lines.append(f"  with_meteor : {total_with}")
    lines.append(f"  no_meteor   : {total_no}")
    lines.append(f"  total       : {total}")
    lines.append(f"  ratio       : {ratio_with_total:.2f}% with / {ratio_no_total:.2f}% without")
    lines.append("")
    lines.append(f"Random seed   : {RANDOM_SEED}")
    lines.append(f"YOLO classes  : {CLASS_NAMES}")
    lines.append("")
    lines.append("Manual split counts were used. No ratio-based rounding was applied.")

    (destination / "dataset_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def verify_final_dataset(destination: Path):
    """Check that every image has a matching label file."""
    problems = []

    for split in ["train", "valid", "test"]:
        images_dir = destination / split / "images"
        labels_dir = destination / split / "labels"

        image_files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for img in image_files:
            label = labels_dir / img.with_suffix(".txt").name
            if not label.exists():
                problems.append((split, img.name))

    if problems:
        print("")
        print("Warning: some images do not have a matching label file:")
        for split, name in problems[:20]:
            print(f"  {split}: {name}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
    else:
        print("Verification OK: every image has a matching label file.")


def count_final_images(destination: Path):
    """Count final images in each split."""
    counts = {}

    for split in ["train", "valid", "test"]:
        images_dir = destination / split / "images"
        image_files = [
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        counts[split] = len(image_files)

    return counts


def main():
    print("Building manually split mixed YOLO meteor dataset")
    print("-" * 60)

    validate_manual_split_counts()
    random.seed(RANDOM_SEED)

    print(f"Dataset WITH meteor    : {SOURCE_WITH_METEOR}")
    with_meteor_items = list_image_label_pairs(SOURCE_WITH_METEOR, "with_meteor")

    print(f"Dataset WITHOUT meteor : {SOURCE_NO_METEOR}")
    no_meteor_items = list_image_label_pairs(SOURCE_NO_METEOR, "no_meteor")

    print("")
    print(f"Images found with meteor    : {len(with_meteor_items)}")
    print(f"Images found without meteor : {len(no_meteor_items)}")

    ensure_enough_items(with_meteor_items, N_WITH_METEOR, "with_meteor")
    ensure_enough_items(no_meteor_items, N_NO_METEOR, "no_meteor")

    print("")
    print("Randomly shuffling and selecting images...")
    random.shuffle(with_meteor_items)
    random.shuffle(no_meteor_items)

    selected_with_meteor = with_meteor_items[:N_WITH_METEOR]
    selected_no_meteor = no_meteor_items[:N_NO_METEOR]

    print("")
    print("Manual split plan:")
    for split in ["train", "valid", "test"]:
        n_with = SPLIT_COUNTS_WITH_METEOR[split]
        n_no = SPLIT_COUNTS_NO_METEOR[split]
        print(
            f"  {split}: "
            f"{n_with} with meteor + "
            f"{n_no} without meteor = "
            f"{n_with + n_no} images"
        )

    check_destination(DESTINATION)
    make_yolo_folders(DESTINATION)

    split_with_meteor = split_items(selected_with_meteor, SPLIT_COUNTS_WITH_METEOR)
    split_no_meteor = split_items(selected_no_meteor, SPLIT_COUNTS_NO_METEOR)

    print("")
    print("Copying/moving images and labels...")

    used_names = set()

    for split in ["train", "valid", "test"]:
        place_items(split_with_meteor[split], split, DESTINATION, used_names)
        place_items(split_no_meteor[split], split, DESTINATION, used_names)

        n_with = len(split_with_meteor[split])
        n_no = len(split_no_meteor[split])
        print(f"  {split}: {n_with + n_no} image files")

    write_data_yaml(DESTINATION)
    write_summary(DESTINATION, split_with_meteor, split_no_meteor)

    print("")
    verify_final_dataset(DESTINATION)

    print("")
    print("Final image counts:")
    final_counts = count_final_images(DESTINATION)
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {final_counts[split]} images")

    print("")
    print("-" * 60)
    print("Final dataset created successfully.")
    print(f"Output folder   : {DESTINATION.resolve()}")
    print(f"YAML file       : {(DESTINATION / 'data.yaml').resolve()}")
    print(f"Summary file    : {(DESTINATION / 'dataset_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
