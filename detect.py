"""
Détection de masque en temps réel — webcam, image ou vidéo.
"""
import os
import argparse
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import config
from model import build_model, load_checkpoint

# ── Paramètres ────────────────────────────────────────────────────────────────
SMOOTHING_FRAMES    = 5     # nb de frames pour lisser la prédiction
MIN_FACE_SIZE       = 80    # taille minimale du visage en pixels
FACE_CONF_THRESHOLD = 0.45  # seuil confiance détecteur de visage
CLASS_CONF_THRESHOLD = 0.50 # afficher la prédiction seulement si confiance >= 50%

# ── Transform pour l'inférence ────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.MEAN, config.STD),
])

# ── Couleurs et étiquettes ────────────────────────────────────────────────────
COLORS = {
    "with_mask":      (0,   200, 0),    # vert
    "without_mask":   (0,   0,   220),  # rouge
    "mask_incorrect": (0,   165, 255),  # orange
}
LABELS = {
    "with_mask":      "MASQUE PORTE",
    "without_mask":   "SANS MASQUE",
    "mask_incorrect": "MASQUE MAL PORTE",
}


# ── Chargement du détecteur de visages ───────────────────────────────────────
def load_face_detector():
    proto   = config.FACE_DETECTOR_PROTO
    weights = config.FACE_DETECTOR_MODEL

    if os.path.exists(proto) and os.path.exists(weights):
        net = cv2.dnn.readNet(proto, weights)
        print("Detecteur de visages: OpenCV DNN (SSD ResNet-10)")
        return net, "dnn"

    # Fallback: Haar cascade (intégré à OpenCV, toujours disponible)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    print("Detecteur de visages: Haar Cascade (OpenCV integre)")
    return cv2.CascadeClassifier(cascade_path), "haar"


def detect_faces_dnn(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < FACE_CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = (dets[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


def detect_faces_haar(frame, cascade):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    return [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces] if len(faces) else []


# ── Inférence masque ──────────────────────────────────────────────────────────
@torch.no_grad()
def classify_face(face_bgr: np.ndarray, model, classes, device):
    """Retourne (label, confidence, probs_array)."""
    pil  = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    t    = _transform(pil).unsqueeze(0).to(device)
    out  = model(t)
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    idx   = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), probs


# ── Affichage ─────────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, confidence, classes, all_probs):
    color     = COLORS.get(label, (200, 200, 200))
    text_main = LABELS.get(label, label)
    text_conf = f"{confidence * 100:.0f}%"

    # Rectangle visage
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Fond label principal
    fw, fh = x2 - x1, y2 - y1
    bar_h  = 40
    y_bar  = max(0, y1 - bar_h)
    cv2.rectangle(frame, (x1, y_bar), (x2, y1), color, -1)

    # Texte label
    font      = cv2.FONT_HERSHEY_DUPLEX
    font_size = min(0.75, fw / 200)
    cv2.putText(frame, f"{text_main}  {text_conf}",
                (x1 + 6, y1 - 10), font, font_size, (255, 255, 255), 2)

    # Barre de probabilité pour chaque classe (sous le visage)
    bar_w_total = x2 - x1
    bar_y_start = y2 + 8
    bar_h_cls   = 14
    for i, (cls, prob) in enumerate(zip(classes, all_probs)):
        c    = COLORS.get(cls, (180, 180, 180))
        bw   = int(prob * bar_w_total)
        by   = bar_y_start + i * (bar_h_cls + 4)
        cv2.rectangle(frame, (x1, by), (x1 + bar_w_total, by + bar_h_cls), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, by), (x1 + bw, by + bar_h_cls), c, -1)
        short = LABELS.get(cls, cls)[:15]
        cv2.putText(frame, f"{short}: {prob*100:.0f}%",
                    (x1 + 4, by + bar_h_cls - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


def draw_summary(frame, face_results, classes):
    """Barre de résumé en haut de l'écran."""
    h, w = frame.shape[:2]
    total = len(face_results)
    counts = {c: 0 for c in classes}
    for label, _, _ in face_results:
        if label in counts:
            counts[label] += 1

    bar_h = 42
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)

    summary = f"Visages detectes: {total}"
    for cls in classes:
        short = LABELS.get(cls, cls)
        summary += f"  |  {short}: {counts[cls]}"

    cv2.putText(frame, summary, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ── Smoothing temporel ────────────────────────────────────────────────────────
class FaceTracker:
    """Lisse les prédictions sur les N dernières frames pour éviter le clignotement."""
    def __init__(self, n=SMOOTHING_FRAMES):
        self.history = {}  # face_id -> deque of probs
        self.n = n

    def update(self, face_id, probs):
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.n)
        self.history[face_id].append(probs)

    def get_smooth(self, face_id):
        if face_id not in self.history or not self.history[face_id]:
            return None
        stack = np.array(self.history[face_id])
        return stack.mean(axis=0)

    def cleanup(self, active_ids):
        for k in list(self.history):
            if k not in active_ids:
                del self.history[k]


# ── Point d'entrée ────────────────────────────────────────────────────────────
def run(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = config.CLASSES
    print(f"Device: {device}")
    print(f"Classes: {classes}")

    # Modèle
    model = build_model(num_classes=len(classes))
    model = model.to(device)
    ckpt  = args.checkpoint or os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
    load_checkpoint(model, ckpt, device)
    model.eval()

    # Détecteur de visages
    face_net, det_type = load_face_detector()

    # Source
    src = args.source
    is_image = isinstance(src, str) and src.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    # ── Image fixe ──────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(src)
        if frame is None:
            raise FileNotFoundError(f"Image introuvable: {src}")

        boxes = detect_faces_dnn(frame, face_net) if det_type == "dnn" else detect_faces_haar(frame, face_net)
        results = []
        for (x1, y1, x2, y2, _) in boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            label, conf, probs = classify_face(crop, model, classes, device)
            results.append((label, conf, probs))
            draw_box(frame, x1, y1, x2, y2, label, conf, classes, probs)

        draw_summary(frame, results, classes)
        out = src.rsplit(".", 1)[0] + "_detected." + src.rsplit(".", 1)[1]
        cv2.imwrite(out, frame)
        print(f"Sauvegarde: {out}")
        cv2.imshow("Detection masque", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ── Vidéo / Webcam ──────────────────────────────────────────────────
    source = 0 if src == "0" else src
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir: {source}")

    tracker = FaceTracker(n=SMOOTHING_FRAMES)
    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes   = detect_faces_dnn(frame, face_net) if det_type == "dnn" else detect_faces_haar(frame, face_net)
        results = []

        active_ids = set()
        for i, (x1, y1, x2, y2, _) in enumerate(boxes):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label_raw, conf_raw, probs_raw = classify_face(crop, model, classes, device)

            # Smoothing: moyenne glissante sur les N dernières frames
            face_id = i
            active_ids.add(face_id)
            tracker.update(face_id, probs_raw)
            smooth_probs = tracker.get_smooth(face_id)

            if smooth_probs is not None:
                idx   = int(np.argmax(smooth_probs))
                label = classes[idx]
                conf  = float(smooth_probs[idx])
                probs = smooth_probs
            else:
                label, conf, probs = label_raw, conf_raw, probs_raw

            # N'afficher que si la confiance est suffisante
            if conf >= CLASS_CONF_THRESHOLD:
                draw_box(frame, x1, y1, x2, y2, label, conf, classes, probs)
                results.append((label, conf, probs))
            else:
                # Faible confiance: rectangle gris + "Analyse..."
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
                cv2.putText(frame, "Analyse...", (x1 + 4, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        tracker.cleanup(active_ids)
        draw_summary(frame, results, classes)

        cv2.putText(frame, "q = quitter", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow("Detection de masque  (q pour quitter)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="0",  help="0=webcam, chemin image/video")
    parser.add_argument("--checkpoint", default=None, help="Chemin vers le .pth")
    args = parser.parse_args()
    run(args)
