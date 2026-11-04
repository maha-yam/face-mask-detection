"""
Crée un vrai dataset à partir de votre webcam.

- Capture des frames de votre visage réel SANS masque
- Génère automatiquement les versions AVEC masque (overlay réaliste)
- Crée le split train/val/test et lance l'entraînement

Usage:
    python create_dataset_webcam.py
"""
import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path

import config

# ── Paramètres ────────────────────────────────────────────────────────────────
N_CAPTURES   = 150   # frames à capturer sans masque
INTERVAL_MS  = 200   # intervalle entre captures (ms)
OUTPUT_DIRS  = {
    "without_mask": os.path.join(config.DATA_DIR, "raw_webcam", "without_mask"),
    "with_mask":    os.path.join(config.DATA_DIR, "raw_webcam", "with_mask"),
}


def load_face_detector():
    proto = config.FACE_DETECTOR_PROTO
    weights = config.FACE_DETECTOR_MODEL

    # Téléchargement automatique si absent
    if not os.path.exists(proto) or not os.path.exists(weights):
        import urllib.request
        os.makedirs(os.path.dirname(proto), exist_ok=True)
        print("Téléchargement du détecteur de visages...")
        urls = [
            ("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", proto),
            ("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", weights),
        ]
        for url, path in urls:
            if not os.path.exists(path):
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"  Téléchargé: {os.path.basename(path)}")
                except Exception as e:
                    print(f"  Echec: {e}")

    # Fallback sur Haar cascade (toujours disponible avec OpenCV)
    if not os.path.exists(weights):
        print("Utilisation du détecteur Haar cascade (OpenCV intégré)")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return cv2.CascadeClassifier(cascade_path), "haar"

    return cv2.dnn.readNet(proto, weights), "dnn"


def detect_faces_dnn(frame, net, conf_thresh=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > conf_thresh:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            boxes.append((x1, y1, x2, y2))
    return boxes


def detect_faces_haar(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    return [(x, y, x+w, y+h) for (x, y, w, h) in faces]


def add_mask_overlay(face_img: np.ndarray) -> np.ndarray:
    """
    Ajoute un overlay masque chirurgical réaliste sur le visage.
    Couvre le bas du visage (nez + bouche + menton).
    """
    img = face_img.copy()
    h, w = img.shape[:2]

    # Zone masque: du tiers supérieur au bas du visage
    y_top    = int(h * 0.38)   # hauteur du nez environ
    y_bottom = int(h * 0.95)
    x_left   = int(w * 0.04)
    x_right  = int(w * 0.96)

    # Couleur masque chirurgical (bleu clair médical)
    mask_colors = [
        (180, 140, 100),   # beige/blanc
        (200, 160, 110),
        (100, 160, 200),   # bleu médical
        (120, 180, 210),
        (130, 200, 130),   # vert médical
    ]
    base_color = random.choice(mask_colors)

    # Forme trapézoïde (masque) avec coins légèrement arrondis
    pts = np.array([
        [x_left  + int(w*0.06), y_top],
        [x_right - int(w*0.06), y_top],
        [x_right,               int(h * 0.60)],
        [x_right,               y_bottom],
        [x_left,                y_bottom],
        [x_left,                int(h * 0.60)],
    ], dtype=np.int32)

    # Fond masque semi-transparent
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], base_color)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    # Lignes horizontales (texture tissu)
    line_color = tuple(max(0, c - 25) for c in base_color)
    n_lines = 6
    for i in range(n_lines):
        y_line = y_top + int((y_bottom - y_top) * (i + 1) / (n_lines + 1))
        cv2.line(img, (x_left, y_line), (x_right, y_line), line_color, 1)

    # Pli central vertical
    x_center = w // 2
    cv2.line(img, (x_center, y_top + 5), (x_center, y_bottom - 5), line_color, 1)

    # Bords du masque
    cv2.polylines(img, [pts], True, tuple(max(0, c - 40) for c in base_color), 2)

    # Élastiques (oreilles)
    ear_y = y_top + (y_bottom - y_top) // 2
    cv2.line(img, (x_left, ear_y), (0, ear_y - 10), (100, 100, 100), 2)
    cv2.line(img, (x_right, ear_y), (w-1, ear_y - 10), (100, 100, 100), 2)

    return img


def capture_faces(net, detector_type: str, n_target: int):
    """Ouvre la webcam et capture des visages réels."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    os.makedirs(OUTPUT_DIRS["without_mask"], exist_ok=True)
    os.makedirs(OUTPUT_DIRS["with_mask"],    exist_ok=True)

    captured  = 0
    last_time = 0
    print(f"\nWebcam ouverte. Regardez la caméra SANS masque.")
    print(f"Objectif: {n_target} captures")
    print("Appuyez sur 'q' pour arrêter.\n")

    while captured < n_target:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        tick = cv2.getTickCount() / cv2.getTickFrequency() * 1000

        # Détection visage
        if detector_type == "dnn":
            boxes = detect_faces_dnn(frame, net)
        else:
            boxes = detect_faces_haar(frame, net)

        face_crop = None
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Prendre le plus grand visage
            if (x2 - x1) > 60 and (y2 - y1) > 60:
                face_crop = frame[y1:y2, x1:x2].copy()

        # Capture automatique périodique
        if face_crop is not None and (tick - last_time) > INTERVAL_MS:
            last_time = tick
            idx = f"{captured:04d}"
            # Sans masque
            without_path = os.path.join(OUTPUT_DIRS["without_mask"], f"real_{idx}.jpg")
            cv2.imwrite(without_path, face_crop)
            # Avec masque (overlay automatique)
            with_path = os.path.join(OUTPUT_DIRS["with_mask"], f"real_{idx}.jpg")
            masked = add_mask_overlay(face_crop)
            cv2.imwrite(with_path, masked)
            captured += 1

        # Affichage
        progress = int(50 * captured / n_target)
        bar = f"[{'=' * progress}{' ' * (50 - progress)}] {captured}/{n_target}"
        cv2.putText(display, bar, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if face_crop else (0, 0, 255), 2)
        status = "Visage detecte" if face_crop else "Pas de visage - approchez-vous"
        cv2.putText(display, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if face_crop else (0, 0, 255), 2)
        cv2.putText(display, "q = quitter", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Capture dataset - Restez SANS masque", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def organize_dataset(raw_dir: str):
    """Split le dataset capturé en train/val/test."""
    from dataset import split_raw_dataset

    # Nettoyer ancien dataset
    for split in ("train", "val", "test"):
        p = os.path.join(config.DATA_DIR, split)
        if os.path.isdir(p):
            shutil.rmtree(p)

    split_raw_dataset(raw_dir)


def main():
    print("=" * 55)
    print("  CREATION DU DATASET DEPUIS LA WEBCAM")
    print("=" * 55)

    net, dtype = load_face_detector()
    captured = capture_faces(net, dtype, N_CAPTURES)

    if captured < 20:
        print(f"\nSeulement {captured} images capturées (minimum 20). Recommencez.")
        return

    print(f"\n{captured} paires d'images capturées.")
    print("Organisation du dataset...")
    organize_dataset(os.path.join(config.DATA_DIR, "raw_webcam"))

    print("\n=== Dataset pret ===")
    for split in ("train", "val", "test"):
        p = os.path.join(config.DATA_DIR, split)
        if os.path.isdir(p):
            total = sum(
                len(os.listdir(os.path.join(p, c)))
                for c in os.listdir(p)
                if os.path.isdir(os.path.join(p, c))
            )
            print(f"  {split}: {total} images")

    print("\n=== PROCHAINE ETAPE ===")
    print("Lancez l'entrainement:")
    print("  python train.py --epochs 30")


if __name__ == "__main__":
    main()
