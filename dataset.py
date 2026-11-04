import os
import shutil
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

import config


def get_transforms(phase: str) -> transforms.Compose:
    """Returns augmentation transforms for train and simple resize for val/test."""
    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ])


def build_dataloaders():
    """Creates train/val/test DataLoaders from the data directory."""
    datasets_dict = {}
    for phase in ("train", "val", "test"):
        path = {"train": config.TRAIN_DIR, "val": config.VAL_DIR, "test": config.TEST_DIR}[phase]
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Dossier '{path}' introuvable.\n"
                "Lancez d'abord : python prepare_data.py --source /chemin/vers/images_brutes"
            )
        datasets_dict[phase] = datasets.ImageFolder(path, transform=get_transforms(phase))

    # Verify class consistency
    train_classes = datasets_dict["train"].classes
    if set(train_classes) != set(config.CLASSES):
        print(f"[AVERTISSEMENT] Classes trouvées: {train_classes}")
        print(f"Classes attendues: {config.CLASSES}")

    loaders = {
        phase: DataLoader(
            ds,
            batch_size=config.BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=0,   # 0 required on Windows to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available(),
        )
        for phase, ds in datasets_dict.items()
    }

    print("\n=== Dataset chargé ===")
    for phase, ds in datasets_dict.items():
        counts = {cls: 0 for cls in ds.classes}
        for _, label in ds.samples:
            counts[ds.classes[label]] += 1
        print(f"  {phase:5s}: {len(ds):5d} images  {counts}")

    return loaders, datasets_dict["train"].classes


def split_raw_dataset(source_dir: str):
    """
    Splits a flat dataset (source_dir/class_name/img.jpg) into
    train / val / test folders under config.DATA_DIR.

    Expected source structure:
        source_dir/
            with_mask/      *.jpg
            without_mask/   *.jpg
            mask_incorrect/ *.jpg   (optional)
    """
    source_dir = Path(source_dir)
    random.seed(42)

    for cls in config.CLASSES:
        cls_path = source_dir / cls
        if not cls_path.is_dir():
            print(f"  [SKIP] Classe '{cls}' absente dans {source_dir}")
            continue

        images = list(cls_path.glob("*.[jJpP][pPnN][gG]"))
        images += list(cls_path.glob("*.jpeg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)
        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, files in splits.items():
            dest = Path(config.DATA_DIR) / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

        print(f"  {cls}: {n} images → train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")

    print(f"\nDataset divisé dans : {config.DATA_DIR}")
