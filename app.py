import gradio as gr

def predict_gesture(image):
    if image is None:
        return None, "Please upload an image or use your webcam."

    return image, "Model not connected yet."

with gr.Blocks(title="Hand Gesture Recognition") as demo:
    gr.Markdown("# Hand Gesture Recognition")

    gr.Markdown(
        "Recognize five hand gestures: "
        "fist, peace, stop, thumbs_down, thumbs_up."
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

        output_image = gr.Image(label="Image Preview")

    status = gr.Textbox(label="Status")

    predict_button = gr.Button(
        "Recognize Gesture",
        variant="primary"
    )

    predict_button.click(
        fn=predict_gesture,
        inputs=input_image,
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    demo.launch()
