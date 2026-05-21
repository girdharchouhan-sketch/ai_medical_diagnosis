# models/xray_cnn.py

from tensorflow.keras.models import Sequential

from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

# THIS FUNCTION MUST EXIST
def build_xray_model(
    input_shape=(224, 224, 3),
    classes=3
):

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=input_shape
        )
    )

    model.add(
        MaxPooling2D(2, 2)
    )

    model.add(
        Conv2D(
            64,
            (3, 3),
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(2, 2)
    )

    model.add(
        Conv2D(
            128,
            (3, 3),
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(2, 2)
    )

    model.add(
        Flatten()
    )

    model.add(
        Dense(
            128,
            activation='relu'
        )
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(
            classes,
            activation='softmax'
        )
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model