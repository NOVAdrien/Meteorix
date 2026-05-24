"""
Build a mixed COCO dataset from two separate COCO datasets:
- one dataset with meteor images
- one dataset without meteor images

Final invariant enforced by this script:
- The only category in every final COCO JSON is:
  {"id": 1, "name": "meteor", "supercategory": "none"}
- All kept annotations have category_id = 1.
- Roboflow pseudo-categories such as project/dataset names are ignored.
- Images without annotations are kept in the JSON, which is valid for COCO negative images.

Each source dataset is expected to look like this:

dataset_with_meteor/
  train/
    image_001.jpg
    image_002.jpg
    ...
    _annotations_formatted.coco.json

dataset_no_meteor/
  train/
    image_abc.jpg
    image_def.jpg
    ...
    _annotations_formatted.coco.json

The script creates a new COCO dataset:

mixed_dataset/
  train/
    image_xxx.jpg
    ...
    _annotations_formatted.coco.json
  valid/
    image_xxx.jpg
    ...
    _annotations_formatted.coco.json
  test/
    image_xxx.jpg
    ...
    _annotations_formatted.coco.json

This version does NOT compute split sizes from percentages.
The number of images per split is defined manually below.
"""

import copy
import json
import random
import shutil
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "../data/datasets/annotated/"
DATASET_DIR = DATASET_DIR.resolve()

# Root folder of the formatted COCO export containing images WITH meteor.
SOURCE_WITH_METEOR = DATASET_DIR / "Meteorix_Mendeley_dataset_with_meteor_annotated_formatted.coco"

# Root folder of the formatted COCO export containing images WITHOUT meteor.
SOURCE_NO_METEOR = DATASET_DIR / "Meteorix_Mendeley_dataset_no_meteor_annotated_formatted.coco"

# Output folder for the final mixed COCO dataset.
DESTINATION = DATASET_DIR / "Meteorix_Mendeley_dataset_mixed_annotated_formatted_split.coco"

# COCO annotation filename used by formatted exports.
INPUT_ANNOTATION_FILENAME = "_annotations_formatted.coco.json"
OUTPUT_ANNOTATION_FILENAME = "_annotations.coco.json"

# The only real detection class allowed in the final dataset.
TARGET_CATEGORY_ID = 1
TARGET_CATEGORY_NAME = "meteor"
TARGET_CATEGORY_SUPERCATEGORY = "none"

# Random seed for reproducible sampling.
RANDOM_SEED = 42

# Accepted image extensions.
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

def target_categories():
    return [
        {
            "id": TARGET_CATEGORY_ID,
            "name": TARGET_CATEGORY_NAME,
            "supercategory": TARGET_CATEGORY_SUPERCATEGORY,
        }
    ]


def find_coco_split_folder(dataset_root: Path) -> Path:
    """
    Find the folder that contains the COCO annotation JSON.

    In this project, it should normally be:
        dataset_root/train/
    """
    direct_train = dataset_root / "train"
    if (direct_train / INPUT_ANNOTATION_FILENAME).is_file():
        return direct_train

    candidates = [
        p for p in dataset_root.rglob("*")
        if p.is_dir() and (p / INPUT_ANNOTATION_FILENAME).is_file()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {INPUT_ANNOTATION_FILENAME} in: {dataset_root}"
        )

    if len(candidates) > 1:
        print(f"Warning: multiple COCO annotation folders found in {dataset_root}.")
        print(f"Using: {candidates[0]}")

    return candidates[0]


def load_json(path: Path):
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    """Save a JSON file with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_file_name(file_name: str) -> str:
    """
    Normalize a COCO file_name to its basename.

    Roboflow usually stores only the filename, but this also handles paths
    such as train/image.jpg or ./image.jpg.
    """
    return Path(file_name).name


def get_meteor_category_ids(coco: dict, annotation_path: Path) -> set:
    """
    Return source category IDs whose category name is exactly 'meteor'.

    Roboflow pseudo-categories such as project/dataset names are deliberately
    ignored, even if they appear in the source categories list.
    """
    meteor_ids = set()

    for cat in coco.get("categories", []):
        old_id = cat.get("id")
        name = cat.get("name")

        if old_id is None or name is None:
            continue

        if str(name).strip().lower() == TARGET_CATEGORY_NAME.lower():
            meteor_ids.add(old_id)

    annotations = coco.get("annotations", [])
    if annotations and not meteor_ids:
        raise ValueError(
            f"Found {len(annotations)} annotations in {annotation_path}, "
            f"but no category named '{TARGET_CATEGORY_NAME}'.\n"
            "Cannot safely identify which annotations are meteors."
        )

    return meteor_ids


def load_coco_dataset(dataset_root: Path, source_name: str):
    """
    Load one source COCO dataset and return image records with annotations.

    Only annotations belonging to the real 'meteor' class are kept.
    Roboflow pseudo-categories are ignored at load time.
    """
    split_folder = find_coco_split_folder(dataset_root)
    annotation_path = split_folder / INPUT_ANNOTATION_FILENAME
    coco = load_json(annotation_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    meteor_category_ids = get_meteor_category_ids(coco, annotation_path)

    annotations_by_image_id = {}
    skipped_non_meteor_annotations = 0

    for ann in annotations:
        old_category_id = ann.get("category_id")

        # Keep only real meteor annotations.
        if old_category_id not in meteor_category_ids:
            skipped_non_meteor_annotations += 1
            continue

        image_id = ann.get("image_id")
        annotations_by_image_id.setdefault(image_id, []).append(ann)

    items = []
    missing_files = []

    for img in images:
        original_file_name = img.get("file_name")
        if original_file_name is None:
            raise ValueError(f"An image entry in {annotation_path} has no file_name.")

        file_name = normalize_file_name(original_file_name)
        image_path = split_folder / file_name

        if not image_path.is_file():
            matches = [
                p for p in split_folder.rglob(file_name)
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if matches:
                image_path = matches[0]
            else:
                missing_files.append(file_name)
                continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        items.append({
            "image_info": img,
            "annotations": annotations_by_image_id.get(img.get("id"), []),
            "image_path": image_path,
            "source": source_name,
            "source_annotation_path": annotation_path,
        })

    if skipped_non_meteor_annotations:
        print(
            f"Warning: skipped {skipped_non_meteor_annotations} non-meteor "
            f"or Roboflow pseudo-category annotations in {annotation_path}."
        )

    if missing_files:
        print("")
        print(f"Warning: {len(missing_files)} image files listed in {annotation_path} were not found.")
        print("First missing files:")
        for name in missing_files[:20]:
            print(f"  {name}")
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")

    return {
        "root": dataset_root,
        "split_folder": split_folder,
        "annotation_path": annotation_path,
        "coco": coco,
        "items": items,
        "meteor_category_ids": meteor_category_ids,
    }


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


def ensure_enough_items(items, required_count, label):
    """Check that the source dataset contains enough images."""
    available = len(items)

    if available < required_count:
        raise ValueError(
            f"Not enough images for '{label}'. "
            f"Required: {required_count}, available: {available}"
        )


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


def make_coco_folders(destination: Path):
    """Create train/valid/test folders."""
    for split in ["train", "valid", "test"]:
        (destination / split).mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path):
    """Copy a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_output_image_name(item, split_name, used_names):
    """
    Create a unique output image filename.

    This version keeps the original image filename.
    It only adds _1, _2, etc. if there is a filename collision
    inside the same split.
    """
    image_path = item["image_path"]
    base_name = image_path.stem
    image_suffix = image_path.suffix.lower()

    candidate_image_name = f"{base_name}{image_suffix}"

    counter = 1
    while (split_name, candidate_image_name) in used_names:
        candidate_image_name = f"{base_name}_{counter}{image_suffix}"
        counter += 1

    used_names.add((split_name, candidate_image_name))

    return candidate_image_name


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


def make_empty_coco_template():
    """Create a minimal single-class COCO dictionary."""
    return {
        "info": {
            "description": "Mixed meteor COCO dataset generated with manual split counts",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": "",
        },
        "licenses": [],
        "categories": target_categories(),
        "images": [],
        "annotations": [],
    }


def create_split_coco(
    split_name,
    split_items_with_meteor,
    split_items_no_meteor,
    destination,
    used_names,
):
    """
    Copy images for one split and create its COCO annotation JSON.
    """
    split_output_dir = destination / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    output_coco = make_empty_coco_template()

    combined_items = []
    combined_items.extend(split_items_with_meteor)
    combined_items.extend(split_items_no_meteor)

    new_image_id = 1
    new_annotation_id = 1

    for item in combined_items:
        output_image_name = safe_output_image_name(item, split_name, used_names)
        output_image_path = split_output_dir / output_image_name

        copy_file(item["image_path"], output_image_path)

        original_image = item["image_info"]
        new_image = copy.deepcopy(original_image)

        new_image["id"] = new_image_id
        new_image["file_name"] = output_image_name

        output_coco["images"].append(new_image)

        for original_annotation in item["annotations"]:
            # At this point load_coco_dataset has already filtered out every
            # non-meteor annotation. We still force category_id=1 here.
            new_annotation = copy.deepcopy(original_annotation)
            new_annotation["id"] = new_annotation_id
            new_annotation["image_id"] = new_image_id
            new_annotation["category_id"] = TARGET_CATEGORY_ID

            output_coco["annotations"].append(new_annotation)
            new_annotation_id += 1

        new_image_id += 1

    save_json(output_coco, split_output_dir / OUTPUT_ANNOTATION_FILENAME)

    return output_coco


def write_summary(destination: Path, split_with_meteor, split_no_meteor):
    """Write a readable summary of the final dataset."""
    lines = []
    lines.append("COCO meteor dataset summary")
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
    lines.append("")
    lines.append("Images were copied only. No move option is available.")
    lines.append("Manual split counts were used. No ratio-based rounding was applied.")
    lines.append(f"Input COCO annotation filename : {INPUT_ANNOTATION_FILENAME}")
    lines.append(f"Output COCO annotation filename: {OUTPUT_ANNOTATION_FILENAME}")
    lines.append("")
    lines.append("Final category schema:")
    lines.append(f"  id {TARGET_CATEGORY_ID}: {TARGET_CATEGORY_NAME}")

    (destination / "dataset_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def verify_final_dataset(destination: Path):
    """Verify that each split JSON matches the copied images and is single-class meteor."""
    for split in ["train", "valid", "test"]:
        split_dir = destination / split
        annotation_path = split_dir / OUTPUT_ANNOTATION_FILENAME

        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

        coco = load_json(annotation_path)

        categories = coco.get("categories", [])
        if categories != target_categories():
            raise ValueError(f"Invalid categories in {annotation_path}: {categories}")

        invalid_annotations = [
            ann for ann in coco.get("annotations", [])
            if ann.get("category_id") != TARGET_CATEGORY_ID
        ]
        if invalid_annotations:
            raise ValueError(
                f"Found {len(invalid_annotations)} annotations with category_id != "
                f"{TARGET_CATEGORY_ID} in {annotation_path}"
            )

        json_image_names = {
            normalize_file_name(img.get("file_name", ""))
            for img in coco.get("images", [])
        }

        copied_image_names = {
            p.name
            for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        missing_on_disk = sorted(json_image_names - copied_image_names)
        missing_in_json = sorted(copied_image_names - json_image_names)

        if missing_on_disk:
            print(f"Warning: {split} has {len(missing_on_disk)} JSON images missing on disk.")
            for name in missing_on_disk[:10]:
                print(f"  Missing on disk: {name}")

        if missing_in_json:
            print(f"Warning: {split} has {len(missing_in_json)} copied images missing in JSON.")
            for name in missing_in_json[:10]:
                print(f"  Missing in JSON: {name}")

        if not missing_on_disk and not missing_in_json:
            print(f"Verification OK for {split}: images, JSON and category schema match.")


def count_final_images(destination: Path):
    """Count final images in each split."""
    counts = {}

    for split in ["train", "valid", "test"]:
        split_dir = destination / split
        image_files = [
            p for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        counts[split] = len(image_files)

    return counts


def count_final_annotations(destination: Path):
    """Count final COCO annotations in each split."""
    counts = {}

    for split in ["train", "valid", "test"]:
        annotation_path = destination / split / OUTPUT_ANNOTATION_FILENAME
        coco = load_json(annotation_path)
        counts[split] = len(coco.get("annotations", []))

    return counts


def main():
    print("Building manually split mixed single-class COCO meteor dataset")
    print("-" * 70)

    validate_manual_split_counts()
    random.seed(RANDOM_SEED)

    print(f"Dataset WITH meteor    : {SOURCE_WITH_METEOR}")
    with_meteor_dataset = load_coco_dataset(SOURCE_WITH_METEOR, "with_meteor")

    print(f"Dataset WITHOUT meteor : {SOURCE_NO_METEOR}")
    no_meteor_dataset = load_coco_dataset(SOURCE_NO_METEOR, "no_meteor")

    with_meteor_items = with_meteor_dataset["items"]
    no_meteor_items = no_meteor_dataset["items"]

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
    make_coco_folders(DESTINATION)

    split_with_meteor = split_items(selected_with_meteor, SPLIT_COUNTS_WITH_METEOR)
    split_no_meteor = split_items(selected_no_meteor, SPLIT_COUNTS_NO_METEOR)

    print("")
    print("Copying images and writing single-class COCO annotation files...")

    used_names = set()

    for split in ["train", "valid", "test"]:
        create_split_coco(
            split_name=split,
            split_items_with_meteor=split_with_meteor[split],
            split_items_no_meteor=split_no_meteor[split],
            destination=DESTINATION,
            used_names=used_names,
        )

        n_with = len(split_with_meteor[split])
        n_no = len(split_no_meteor[split])
        print(f"  {split}: {n_with + n_no} image files")

    write_summary(DESTINATION, split_with_meteor, split_no_meteor)

    print("")
    verify_final_dataset(DESTINATION)

    print("")
    print("Final image counts:")
    final_image_counts = count_final_images(DESTINATION)
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {final_image_counts[split]} images")

    print("")
    print("Final COCO annotation counts:")
    final_annotation_counts = count_final_annotations(DESTINATION)
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {final_annotation_counts[split]} annotations")

    print("")
    print("-" * 70)
    print("Final single-class COCO dataset created successfully.")
    print(f"Output folder   : {DESTINATION.resolve()}")
    print(f"Summary file    : {(DESTINATION / 'dataset_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
