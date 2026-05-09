from pathlib import Path
import hashlib
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent

DOSSIER = BASE_DIR / "../data/datasets/raw/Meteors/CroppedImages/meteor/"
DOSSIER = DOSSIER.resolve()

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

if not DOSSIER.exists():
    raise FileNotFoundError(f"Le dossier n'existe pas : {DOSSIER.resolve()}")

if not DOSSIER.is_dir():
    raise NotADirectoryError(f"Le chemin existe mais n'est pas un dossier : {DOSSIER.resolve()}")

hashes = defaultdict(list)

for img in DOSSIER.rglob("*"):
    if img.is_file() and img.suffix.lower() in extensions:
        h = hashlib.md5(img.read_bytes()).hexdigest()
        hashes[h].append(img)

duplicates = {h: files for h, files in hashes.items() if len(files) > 1}

print(f"Images trouvées : {sum(len(v) for v in hashes.values())}")
print(f"Groupes de doublons exacts : {len(duplicates)}")

for h, files in duplicates.items():
    print("\nDoublon détecté :")
    for f in files:
        print("  ", f)