import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux:      {total:,}")
    print(f"Paramètres entraîn.:   {trainable:,}")
    return trainable


def show_batch(loader, classes, n=16, save_path=None):
    """Visualise un batch d'images avec leurs labels."""
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]

    # Dénormaliser
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)

    grid = make_grid(images, nrow=4, padding=4)
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    title = "  |  ".join(classes[l] for l in labels.tolist())
    plt.title(title[:100], fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Batch visualisé: {save_path}")
    else:
        plt.show()
    plt.close()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Exécution sur CPU")
    return device


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    device = next(model.parameters()).device
    x = torch.zeros(input_size).to(device)
    try:
        from torchinfo import summary
        summary(model, input_size=input_size)
    except ImportError:
        out = model(x)
        print(f"Input shape:  {tuple(x.shape)}")
        print(f"Output shape: {tuple(out.shape)}")
        count_parameters(model)
