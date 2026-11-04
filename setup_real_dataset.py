"""
Extrait et organise le dataset HuggingFace (poolrf2001/FaceMask)
en train/val/test prêts pour l'entraînement.
"""
import os
import shutil
import zipfile

import config

RAW_DIR = os.path.join(config.DATA_DIR, "_hf_raw")


def inspect_zip(path):
    """Retourne les sous-dossiers de premier niveau dans le zip."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    folders = set()
    for n in names:
        parts = n.split("/")
        if len(parts) > 1:
            folders.add(parts[0])
    return sorted(folders)


def extract_all():
    for fname in ["train.zip", "test.zip", "validation.zip"]:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"[MANQUANT] {fname} — relancez le téléchargement.")
            continue
        size_mb = os.path.getsize(path) / 1e6
        print(f"Extraction {fname} ({size_mb:.1f} MB)...")
        with zipfile.ZipFile(path) as z:
            z.extractall(RAW_DIR)
        top = inspect_zip(path)
        print(f"  Dossiers: {top}")


def find_class_dirs(base: str):
    """Trouve les dossiers with_mask / without_mask / mask_weared_incorrect."""
    known = {"with_mask", "without_mask", "mask_weared_incorrect",
             "WithMask", "WithoutMask", "mask_incorrect"}
    found = {}
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if d in known or d.lower().replace(" ", "_") in {k.lower() for k in known}:
                key = d.lower().replace(" ", "_")
                if "without" in key or "no_mask" in key:
                    found["without_mask"] = os.path.join(root, d)
                elif "incorrect" in key or "weared_incorrect" in key:
                    found["mask_incorrect"] = os.path.join(root, d)
                elif "with" in key or "mask" in key:
                    found.setdefault("with_mask", os.path.join(root, d))
    return found


def copy_split(src_dir: str, split: str, class_name: str, file_list: list):
    dest = os.path.join(config.DATA_DIR, split, class_name)
    os.makedirs(dest, exist_ok=True)
    for fname in file_list:
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(dest, fname))


def organize():
    import random
    random.seed(42)

    # Nettoyer les anciens splits
    for split in ("train", "val", "test"):
        p = os.path.join(config.DATA_DIR, split)
        if os.path.isdir(p):
            shutil.rmtree(p)
            print(f"  Ancien '{split}' supprimé")

    # Trouver les classes dans les extraits
    class_dirs = find_class_dirs(RAW_DIR)
    print(f"\nClasses trouvées: {list(class_dirs.keys())}")

    if not class_dirs:
        print("ERREUR: Aucune classe trouvée. Vérifiez les zips extraits.")
        # Lister le contenu pour debug
        for root, dirs, files in os.walk(RAW_DIR):
            level = root.replace(RAW_DIR, "").count(os.sep)
            if level < 3:
                print("  " + "  " * level + os.path.basename(root) + "/")
        return

    for class_name, class_path in class_dirs.items():
        images = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])
        random.shuffle(images)
        n = len(images)
        n_train = int(n * config.TRAIN_RATIO)
        n_val   = int(n * config.VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }
        for split, files in splits.items():
            copy_split(class_path, split, class_name, files)

        print(f"  {class_name}: {n} images → "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


def main():
    print("=== Organisation du dataset réel ===\n")
    extract_all()
    print()
    organize()

    # Résumé
    print("\n=== Dataset final ===")
    for split in ("train", "val", "test"):
        p = os.path.join(config.DATA_DIR, split)
        if not os.path.isdir(p):
            continue
        total = 0
        parts = []
        for cls in sorted(os.listdir(p)):
            cp = os.path.join(p, cls)
            if os.path.isdir(cp):
                n = len(os.listdir(cp))
                total += n
                parts.append(f"{cls}={n}")
        print(f"  {split:5s}: {total:5d} images  ({', '.join(parts)})")

    print("\n=== PROCHAINE ETAPE ===")
    print("  python train.py --epochs 30")


if __name__ == "__main__":
    main()
