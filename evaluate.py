import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from tqdm import tqdm

import config
from dataset import build_dataloaders
from model import build_model, load_checkpoint


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Évaluation"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Matrice de confusion (nombres)", "Matrice de confusion (normalisée)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Vérité terrain")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}")


def plot_roc_curves(y_true, y_probs, classes, save_path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=range(len(classes)))

    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#F44336", "#FF9800"]
    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC par classe")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Courbes ROC sauvegardées: {save_path}")


def plot_precision_recall(y_true, y_probs, classes, save_path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=range(len(classes)))

    plt.figure(figsize=(8, 6))
    colors = ["#2196F3", "#F44336", "#FF9800"]
    for i, (cls, color) in enumerate(zip(classes, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = auc(rec, prec)
        plt.plot(rec, prec, color=color, lw=2,
                 label=f"{cls} (AP = {ap:.3f})")

    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbes Précision-Rappel par classe")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Courbes P-R sauvegardées: {save_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    loaders, classes = build_dataloaders()
    model = build_model(num_classes=len(classes))
    model = model.to(device)

    checkpoint_path = args.checkpoint or os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
    load_checkpoint(model, checkpoint_path, device)

    split = args.split
    print(f"\nÉvaluation sur le split: {split}")
    y_true, y_pred, y_probs = predict_all(model, loaders[split], device)

    # Metrics
    acc = (y_true == y_pred).mean()
    print(f"\nAccuracy globale: {acc:.4f} ({acc*100:.2f}%)")
    print("\nRapport de classification détaillé:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Plots
    plot_confusion_matrix(y_true, y_pred, classes,
                          os.path.join(config.RESULTS_DIR, "confusion_matrix.png"))
    plot_roc_curves(y_true, y_probs, classes,
                    os.path.join(config.RESULTS_DIR, "roc_curves.png"))
    plot_precision_recall(y_true, y_probs, classes,
                          os.path.join(config.RESULTS_DIR, "precision_recall_curves.png"))

    print(f"\nTous les résultats sauvegardés dans: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation - Détection de masque")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Chemin vers le fichier .pth (défaut: checkpoints/best_model.pth)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split à évaluer (défaut: test)")
    args = parser.parse_args()
    evaluate(args)
