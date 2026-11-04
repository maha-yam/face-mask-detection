import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from dataset import build_dataloaders
from model import build_model, freeze_backbone, unfreeze_all


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }, path)


def plot_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Loss par époque")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy par époque")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Courbes sauvegardées: {save_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Data
    loaders, classes = build_dataloaders()
    print(f"Classes: {classes}\n")

    # Model
    model = build_model(num_classes=len(classes), model_name=args.model)
    model = model.to(device)

    # Phase 1 – Feature extraction: only train the classifier head
    print("=== Phase 1: Feature extraction (backbone gelé) ===")
    freeze_backbone(model, args.model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=config.LR_PATIENCE,
                                  factor=config.LR_FACTOR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")

    phase1_epochs = min(10, args.epochs // 3)
    total_epochs = args.epochs

    for epoch in range(1, total_epochs + 1):
        # Switch to full fine-tuning after phase1_epochs
        if epoch == phase1_epochs + 1:
            print(f"\n=== Phase 2: Fine-tuning complet (epoch {epoch}) ===")
            unfreeze_all(model)
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1,
                                   weight_decay=config.WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=config.LR_PATIENCE,
                                          factor=config.LR_FACTOR)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_acc)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:3d}/{total_epochs}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, best_path)
            print(f"  [OK] Meilleur modele sauvegarde (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping déclenché après {epoch} époques.")
                break

    print(f"\nEntraînement terminé. Meilleure val_acc: {best_val_acc:.4f}")
    plot_history(history, os.path.join(config.RESULTS_DIR, "training_curves.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement - Détection de masque")
    parser.add_argument("--model", default=config.MODEL_NAME,
                        choices=["mobilenet_v2", "resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    config.BATCH_SIZE = args.batch
    train(args)
