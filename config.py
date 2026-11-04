import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Classes — auto-détectées depuis le dossier train/ si disponible
def _detect_classes():
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train")
    if os.path.isdir(train_path):
        found = sorted([d for d in os.listdir(train_path)
                        if os.path.isdir(os.path.join(train_path, d))])
        if found:
            return found
    return ["with_mask", "without_mask", "mask_incorrect"]

CLASSES = _detect_classes()
NUM_CLASSES = len(CLASSES)

# Image settings
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7          # early stopping patience
LR_PATIENCE = 3       # LR scheduler patience
LR_FACTOR = 0.5       # LR reduction factor

# Model
MODEL_NAME = "mobilenet_v2"  # or "resnet50"
PRETRAINED = True
DROPOUT = 0.3

# Dataset split ratios (if splitting from raw data)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Face detector (for real-time detection)
FACE_DETECTOR_PROTO = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
FACE_DETECTOR_MODEL = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
FACE_CONFIDENCE_THRESHOLD = 0.5

# Detection colors (BGR)
COLORS = {
    "with_mask": (0, 255, 0),       # green
    "without_mask": (0, 0, 255),    # red
    "mask_incorrect": (0, 165, 255) # orange
}

# Labels in French for display
LABELS_FR = {
    "with_mask": "Masque correctement porte",
    "without_mask": "Sans masque",
    "mask_incorrect": "Masque mal porte"
}
