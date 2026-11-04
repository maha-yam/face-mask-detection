import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes: int = config.NUM_CLASSES, model_name: str = config.MODEL_NAME) -> nn.Module:
    """
    Builds a fine-tuned classifier on top of a pretrained backbone.
    Supported: mobilenet_v2, resnet50, efficientnet_b0
    """
    if model_name == "mobilenet_v2":
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if config.PRETRAINED else None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT / 2),
            nn.Linear(256, num_classes),
        )

    elif model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if config.PRETRAINED else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=config.DROPOUT / 2),
            nn.Linear(512, num_classes),
        )

    elif model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if config.PRETRAINED else None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Modèle non supporté: {model_name}. Choisir parmi: mobilenet_v2, resnet50, efficientnet_b0")

    return backbone


def freeze_backbone(model: nn.Module, model_name: str = config.MODEL_NAME):
    """Freezes all layers except the final classifier (for feature extraction phase)."""
    if model_name == "mobilenet_v2":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    elif model_name == "efficientnet_b0":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def unfreeze_all(model: nn.Module):
    """Unfreezes all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"Tous les paramètres dégelés: {total:,}")


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Checkpoint chargé depuis '{path}' (epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc={checkpoint.get('val_acc', '?'):.4f})")
    return checkpoint
