# models/inference.py

import cv2
import numpy as np

from tensorflow.keras.models import load_model

from config.settings import (
    XRAY_MODEL_PATH,
    IMAGE_SIZE,
    XRAY_CLASSES
)

# ================= LOAD MODEL =================

xray_model = load_model(
    XRAY_MODEL_PATH
)

# ================= VALIDATE XRAY =================

def is_valid_xray(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    height, width, _ = image.shape

    # Reject small images
    if height < 100 or width < 100:
        return False

    # X-rays are mostly grayscale
    b, g, r = cv2.split(image)

    if (
        np.mean(np.abs(b - g)) > 15
        or
        np.mean(np.abs(g - r)) > 15
    ):
        return False

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    std_dev = np.std(gray)

    # Reject blank/random images
    if std_dev < 15:
        return False

    return True

# ================= PREPROCESS =================

def preprocess_xray(image_path):

    image = cv2.imread(image_path)

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image



def predict_xray(image_path):

    # Validate X-ray
    if not is_valid_xray(image_path):

        raise ValueError(
            "Please upload a valid Chest X-ray image"
        )

    image = preprocess_xray(
        image_path
    )

    predictions = xray_model.predict(
        image
    )

    class_index = np.argmax(
        predictions
    )

    confidence = float(
        predictions[0][class_index] * 100
    )

    # Reject low-confidence predictions
    if confidence < 75:

        raise ValueError(
            "Invalid Image. Please upload a clear Chest X-ray scan."
        )

    predicted_class = XRAY_CLASSES[
        class_index
    ]

    return predicted_class, confidence