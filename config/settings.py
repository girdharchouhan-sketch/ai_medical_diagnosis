# config/settings.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

XRAY_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models/pretrained/xray_model.h5"
)

IMAGE_SIZE = (224, 224)

XRAY_CLASSES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis"
]