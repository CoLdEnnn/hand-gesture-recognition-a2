from pathlib import Path
import time

import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "real_plus_artaug.keras"

CLASS_NAMES = [
    "fist",
    "peace",
    "stop",
    "thumbs_down",
    "thumbs_up"
]

model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

augmentation = model.layers[1]
backbone = model.layers[2]
pooling = model.layers[3]
dropout = model.layers[4]
classifier = model.layers[5]

def predict_gesture(image):
    if image is None:
        return None, None, "Please upload an image."

    start = time.perf_counter()

    original = image.convert("RGB")
    resized = original.resize((224, 224))

    image_array = np.asarray(resized, dtype=np.float32)
    image_batch = tf.convert_to_tensor(image_array[None, ...])

    with tf.GradientTape() as tape:
        x = augmentation(image_batch, training=False)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

        feature_maps = backbone(x, training=False)
        tape.watch(feature_maps)

        x = pooling(feature_maps)
        x = dropout(x, training=False)

        probabilities = classifier(x)

        predicted_index = int(tf.argmax(probabilities[0]).numpy())

        class_score = tf.math.log(
            probabilities[:, predicted_index] + 1e-7
        )

    gradients = tape.gradient(class_score, feature_maps)

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    heatmap = tf.reduce_sum(
        feature_maps[0] * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    colors = plt.get_cmap("jet")(heatmap_array)[..., :3] * 255

    original_array = np.asarray(original, dtype=np.float32)

    alpha = (0.55 * heatmap_array)[..., None]

    overlay = (
        original_array * (1 - alpha)
        + colors * alpha
    )

    gradcam_image = Image.fromarray(
        np.uint8(np.clip(overlay, 0, 255))
    )

    scores = {
        name: float(probability)
        for name, probability in zip(
            CLASS_NAMES,
            probabilities[0].numpy()
        )
    }

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        probabilities[0, predicted_index]
    ) * 100

    processing_time = (time.perf_counter() - start) * 1000

    status = (
        f"Prediction: {predicted_class} | "
        f"Confidence: {confidence:.2f}% | "
        f"Processing time: {processing_time:.0f} ms"
    )

    return gradcam_image, scores, status

with gr.Blocks(title="Hand Gesture Recognition") as demo:

    gr.Markdown("# Hand Gesture Recognition")

    gr.Markdown(
        "Recognize five gestures using MobileNetV2 "
        "with Grad-CAM visual explanations."
    )

    gr.Markdown(
        "Test Accuracy: 91.50% | Macro F1-score: 91.70%"
    )

    gr.Markdown(
        "Please use only images you own or have permission to use."
    )

    with gr.Row():
        input_image = gr.Image(
            sources=["upload", "webcam"],
            type="pil",
            label="Input Image"
        )

        gradcam_output = gr.Image(
            label="Grad-CAM Explanation"
        )

    prediction_result = gr.Label(
        label="Gesture Probabilities",
        num_top_classes=5
    )

    status = gr.Textbox(label="Status")

    predict_button = gr.Button(
        "Recognize Gesture",
        variant="primary"
    )

    predict_button.click(
        fn=predict_gesture,
        inputs=input_image,
        outputs=[
            gradcam_output,
            prediction_result,
            status
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True)
