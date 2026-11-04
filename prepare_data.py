"""
Script de préparation des données.

Usage:
  1. Avec un dossier local:
       python prepare_data.py --source /chemin/vers/dataset_brut

  2. Téléchargement automatique depuis Kaggle (nécessite ~/.kaggle/kaggle.json):
       python prepare_data.py --kaggle

Le dataset attendu en entrée:
    source/
        with_mask/      (images avec masque)
        without_mask/   (images sans masque)
        mask_incorrect/ (masque mal porté, optionnel)
"""
import os
import argparse
import zipfile

import config
from dataset import split_raw_dataset


def download_from_kaggle():
    """Télécharge le dataset Kaggle 'Face Mask Detection'."""
    try:
        import kaggle
    except ImportError:
        print("Installez kaggle: pip install kaggle")
        print("Configurez votre clé API: https://www.kaggle.com/docs/api")
        return None

    dataset = "andrewmvd/face-mask-detection"
    download_dir = os.path.join(config.BASE_DIR, "data", "raw_kaggle")
    os.makedirs(download_dir, exist_ok=True)

    print(f"Téléchargement du dataset Kaggle: {dataset}")
    os.system(f"kaggle datasets download -d {dataset} -p {download_dir} --unzip")
    return download_dir


def create_demo_structure():
    """
    Crée une structure de démo avec des images synthétiques (cercles colorés)
    pour tester le pipeline sans données réelles.
    """
    import random
    from PIL import Image, ImageDraw, ImageFont

    print("\nCréation d'un dataset de démonstration (images synthétiques)…")
    print("ATTENTION: Ce dataset est factice, pour tester le pipeline uniquement.\n")

    colors_map = {
        "with_mask": [(0, 180, 0), (0, 200, 50), (50, 220, 50)],
        "without_mask": [(220, 50, 50), (200, 0, 0), (180, 30, 30)],
        "mask_incorrect": [(220, 140, 0), (200, 120, 0), (180, 100, 0)],
    }
    n_per_class = {"train": 200, "val": 50, "test": 50}
    random.seed(42)

    for split, n in n_per_class.items():
        for cls, colors in colors_map.items():
            dest = os.path.join(config.DATA_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            for i in range(n):
                img = Image.new("RGB", (224, 224), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)
                color = random.choice(colors)
                # Face circle
                draw.ellipse([40, 30, 184, 194], fill=(255, 220, 180), outline=(200, 160, 120), width=2)
                # Mask or no mask indicator
                if cls == "with_mask":
                    draw.rectangle([50, 120, 174, 185], fill=color)
                elif cls == "without_mask":
                    pass
                else:  # incorrect
                    draw.rectangle([50, 140, 174, 185], fill=color)
                # Eyes
                draw.ellipse([75, 80, 105, 110], fill=(60, 60, 60))
                draw.ellipse([119, 80, 149, 110], fill=(60, 60, 60))
                # Noise
                pixels = img.load()
                for _ in range(500):
                    x, y = random.randint(0, 223), random.randint(0, 223)
                    noise = random.randint(-20, 20)
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        max(0, min(255, r + noise)),
                        max(0, min(255, g + noise)),
                        max(0, min(255, b + noise)),
                    )
                img.save(os.path.join(dest, f"{cls}_{i:04d}.jpg"), quality=90)

    print("Dataset de démonstration créé:")
    for split in ("train", "val", "test"):
        total = sum(
            len(os.listdir(os.path.join(config.DATA_DIR, split, cls)))
            for cls in colors_map
        )
        print(f"  {split}: {total} images")


def main(args):
    if args.demo:
        create_demo_structure()
        return

    if args.kaggle:
        source = download_from_kaggle()
        if source is None:
            return
    elif args.source:
        source = args.source
    else:
        print("Spécifiez --source, --kaggle, ou --demo")
        print("  --demo   : dataset synthétique pour tester le pipeline")
        print("  --source : chemin vers votre dataset local")
        print("  --kaggle : téléchargement automatique via l'API Kaggle")
        return

    print(f"\nDivision du dataset depuis: {source}")
    split_raw_dataset(source)
    print("\nPréparation terminée!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation des données")
    parser.add_argument("--source", type=str, default=None,
                        help="Chemin vers le dataset brut local")
    parser.add_argument("--kaggle", action="store_true",
                        help="Télécharger depuis Kaggle")
    parser.add_argument("--demo", action="store_true",
                        help="Créer un dataset synthétique de démonstration")
    args = parser.parse_args()
    main(args)
