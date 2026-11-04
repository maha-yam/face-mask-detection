"""
Télécharge un vrai dataset de visages avec/sans masque depuis GitHub.
Source: prajnasb/observations (dataset public, sans compte requis)
"""
import os
import sys
import shutil
import zipfile
import urllib.request

import config
from dataset import split_raw_dataset


DATASET_URL = "https://github.com/prajnasb/observations/archive/refs/heads/master.zip"
ZIP_PATH    = os.path.join(config.DATA_DIR, "_download.zip")
EXTRACT_DIR = os.path.join(config.DATA_DIR, "_extracted")


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct / 2)
        print(f"\r  [{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%  "
              f"({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)", end="", flush=True)


def download():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    print(f"Téléchargement du dataset réel...")
    print(f"  Source: {DATASET_URL}\n")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, show_progress)
        print()
    except Exception as e:
        print(f"\nErreur de téléchargement: {e}")
        print("Vérifiez votre connexion internet.")
        sys.exit(1)

    print("Extraction du zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    os.remove(ZIP_PATH)

    # Localiser le dossier data dans l'archive extraite
    source = os.path.join(EXTRACT_DIR, "observations-master", "experiements", "data")
    if not os.path.isdir(source):
        # Chercher récursivement un dossier contenant with_mask/
        for root, dirs, _ in os.walk(EXTRACT_DIR):
            if "with_mask" in dirs or "without_mask" in dirs:
                source = root
                break

    if not os.path.isdir(source):
        print(f"Structure inattendue dans l'archive. Contenu:")
        for root, dirs, files in os.walk(EXTRACT_DIR):
            level = root.replace(EXTRACT_DIR, "").count(os.sep)
            print("  " * level + os.path.basename(root) + "/")
        sys.exit(1)

    # Nettoyer ancien dataset
    for split in ("train", "val", "test"):
        split_path = os.path.join(config.DATA_DIR, split)
        if os.path.isdir(split_path):
            shutil.rmtree(split_path)
            print(f"  Ancien split '{split}' supprimé")

    print(f"\nOrganisation du dataset depuis: {source}")
    split_raw_dataset(source)

    # Nettoyer les fichiers temporaires
    shutil.rmtree(EXTRACT_DIR)
    print("\nFichiers temporaires supprimés.")


def print_summary():
    print("\n=== Dataset téléchargé ===")
    for split in ("train", "val", "test"):
        split_path = os.path.join(config.DATA_DIR, split)
        if not os.path.isdir(split_path):
            continue
        total = 0
        details = []
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                n = len([f for f in os.listdir(cls_path)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                total += n
                details.append(f"{cls}={n}")
        print(f"  {split:5s}: {total:4d} images  ({', '.join(details)})")


if __name__ == "__main__":
    download()
    print_summary()
    print("\n=== PROCHAINE ETAPE ===")
    print("Relancez l'entrainement avec les vraies données:")
    print("  python train.py --epochs 30")
