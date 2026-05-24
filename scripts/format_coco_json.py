#!/usr/bin/env python3
"""
Formatte un fichier d'annotations COCO Roboflow en ne gardant que
les annotations de la classe cible "meteor", sans toucher aux images
ni à l'organisation du dataset.

Ce script :
- lit un fichier _annotations.coco.json ;
- supprime les catégories qui ne correspondent pas à "meteor" ;
- supprime les annotations qui ne sont pas de type "meteor" ;
- garde toutes les images, y compris celles sans annotations ;
- ne copie, ne déplace et ne renomme aucune image ;
- écrit un nouveau fichier JSON formaté avec indentation.

Utilisation :
    python format_coco_jsonv3.py

Par défaut, le script crée dans le même dossier :
    _annotations_formatted.coco.json
"""

import copy
import json
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Dossier racine contenant tes datasets annotés.
DATASET_DIR = (BASE_DIR / "../data/datasets/annotated/").resolve()

# Fichier COCO d'entrée.
# Change ce chemin selon le dataset que tu veux reformater.
# INPUT_PATH = DATASET_DIR / "Meteorix_Mendeley_dataset_no_meteor_annotated_formatted.coco" / "train" / "_annotations.coco.json"
INPUT_PATH = DATASET_DIR / "Meteorix_Mendeley_dataset_with_meteor_annotated_formatted.coco" / "train" / "_annotations.coco.json"

# Fichier COCO de sortie.
# Mets None pour créer automatiquement _annotations_formatted.coco.json
# dans le même dossier que INPUT_PATH.
OUTPUT_PATH = None

# Nom de la seule classe à garder.
TARGET_CATEGORY_NAME = "meteor"
TARGET_CATEGORY_SUPERCATEGORY = "none"

# ID final de la catégorie meteor.
# Mets 1 si ton pipeline attend des category_id qui commencent à 1.
TARGET_CATEGORY_ID = 1

# Indentation du JSON de sortie.
INDENT = 2


# ============================================================
# FONCTIONS
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def output_path_for(input_path: Path, output_path: str | Path | None) -> Path:
    if output_path is None:
        return input_path.with_name("_annotations_formatted.coco.json")
    return Path(output_path)


def target_category() -> dict:
    return {
        "id": TARGET_CATEGORY_ID,
        "name": TARGET_CATEGORY_NAME,
        "supercategory": TARGET_CATEGORY_SUPERCATEGORY,
    }


def find_target_category_ids(coco: dict) -> set:
    """
    Trouve les anciens category_id correspondant à la classe TARGET_CATEGORY_NAME.

    Exemple : si le JSON source contient :
        {"id": 3, "name": "meteor"}
    alors cette fonction renvoie {3}.
    """
    target_ids = set()

    for category in coco.get("categories", []):
        category_id = category.get("id")
        category_name = category.get("name")

        if category_id is None or category_name is None:
            continue

        if str(category_name).strip().lower() == TARGET_CATEGORY_NAME.lower():
            target_ids.add(category_id)

    return target_ids


def filter_coco_to_target_class(coco: dict) -> dict:
    """
    Retourne un nouveau dictionnaire COCO contenant :
    - toutes les images du JSON source ;
    - seulement les annotations de la classe cible ;
    - une seule catégorie finale : TARGET_CATEGORY_NAME.

    Les images ne sont ni copiées, ni renommées, ni supprimées.
    Les image_id sont conservés.
    Les file_name sont conservés.
    Les annotation_id sont régénérés proprement à partir de 1.
    """
    target_source_category_ids = find_target_category_ids(coco)
    source_annotations = coco.get("annotations", [])

    if source_annotations and not target_source_category_ids:
        raise ValueError(
            f"Le fichier contient {len(source_annotations)} annotations, "
            f"mais aucune catégorie nommée '{TARGET_CATEGORY_NAME}'."
        )

    output_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "categories": [target_category()],
        "images": copy.deepcopy(coco.get("images", [])),
        "annotations": [],
    }

    new_annotation_id = 1
    skipped_annotations = 0

    for annotation in source_annotations:
        old_category_id = annotation.get("category_id")

        if old_category_id not in target_source_category_ids:
            skipped_annotations += 1
            continue

        new_annotation = copy.deepcopy(annotation)
        new_annotation["id"] = new_annotation_id
        new_annotation["category_id"] = TARGET_CATEGORY_ID

        output_coco["annotations"].append(new_annotation)
        new_annotation_id += 1

    kept_annotations = len(output_coco["annotations"])

    print("Résumé du filtrage COCO")
    print("-" * 40)
    print(f"Images conservées              : {len(output_coco['images'])}")
    print(f"Annotations meteor conservées  : {kept_annotations}")
    print(f"Annotations non-meteor retirées: {skipped_annotations}")
    print(f"Catégorie finale               : {target_category()}")

    return output_coco


def format_coco_json(
    input_path: str | Path,
    output_path: str | Path | None = None,
    indent: int = 2,
) -> Path:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Le fichier d'entrée n'existe pas : {input_file}")

    output_file = output_path_for(input_file, output_path)

    coco = load_json(input_file)
    formatted_coco = filter_coco_to_target_class(coco)
    save_json(formatted_coco, output_file, indent=indent)

    return output_file


# ============================================================
# POINT D'ENTRÉE
# ============================================================

def main() -> None:
    try:
        output_file = format_coco_json(INPUT_PATH, OUTPUT_PATH, INDENT)
        print("")
        print(f"Fichier JSON formaté créé : {output_file.resolve()}")

    except json.JSONDecodeError as e:
        print("Erreur : le fichier fourni n'est pas un JSON valide.")
        print(f"Détail : {e}")
        sys.exit(1)

    except Exception as e:
        print(f"Erreur : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
