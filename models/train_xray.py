import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(PROJECT_ROOT)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.xray_cnn import build_xray_model

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = train_gen.flow_from_directory(
    'data/datasets/xray/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_data = train_gen.flow_from_directory(
    'data/datasets/xray/val',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

model = build_xray_model()

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

model.save(
    'models/pretrained/xray_model.h5'
)

print("Model Trained Successfully")