# utils/gradcam.py

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model

from config.settings import XRAY_MODEL_PATH

model = tf.keras.models.load_model(
    XRAY_MODEL_PATH
)

def generate_gradcam(
    image_path,
    layer_name='conv2d_2'
):

    image = cv2.imread(image_path)

    image = cv2.resize(
        image,
        (224, 224)
    )

    input_image = np.expand_dims(
        image / 255.0,
        axis=0
    )

    grad_model = Model(
        [model.inputs],
        [
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_image
        )

        class_index = tf.argmax(
            predictions[0]
        )

        loss = predictions[:, class_index]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(
    heatmap,
    (224, 224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    output_path = "gradcam_output.jpg"

    cv2.imwrite(
        output_path,
        superimposed
    )

    return output_path