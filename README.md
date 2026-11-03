# Hand Gesture Recognition with MobileNetV2, ArtAug and Grad-CAM

Computer Vision — Assignment 2  
Astana IT University

## Overview

This project recognizes five hand gestures from photographs using a pretrained MobileNetV2 model. It also integrates Grad-CAM visual explanations and an exploratory ArtAug synthetic-data experiment.

**Supported classes:** `fist`, `peace`, `stop`, `thumbs_down`, `thumbs_up`.

The Gradio web application allows users to upload or capture an image and view the prediction, confidence, class probabilities, and Grad-CAM heatmap.

## Dataset

The updated dataset contained 2,267 real images:

| Split | Images |
|---|---:|
| Training | 1,586 |
| Validation | 340 |
| Test | 341 |

The data was divided using stratified splitting with random seed 42. No exact file-path overlap was found between partitions.

Two training files became unavailable before the ArtAug pilot. They were excluded from both experimental conditions, leaving 1,584 real training images. Validation and test sets were unchanged.

## Model

The classifier uses:

- MobileNetV2 with ImageNet pretrained weights
- Input resolution of 224 × 224 pixels
- Global average pooling
- Dropout of 0.3
- Dense output layer with five softmax classes

The original model was trained using Adam, data augmentation, validation-loss checkpointing, and early stopping.

## Results

The original updated-real checkpoint achieved:

- **Test accuracy:** 91.20%
- **Macro F1:** 91.47%
- **Average CPU inference latency:** 395.37 ms

### ArtAug Ablation

The ArtAug experiment added five accepted synthetic images produced offline using FLUX.1-dev and the pretrained ArtAug LoRA adapter.

| Training condition | Test accuracy | Macro F1 |
|---|---:|---:|
| Real only | 91.20% | 91.41% |
| Real + ArtAug | **91.50%** | **91.70%** |

The increase of approximately 0.29 percentage points corresponds to one additional correct prediction among 341 test images. This exploratory result does not establish a statistically reliable ArtAug improvement.

## Run the Application

Required repository files:

```text
app.py
requirements.txt
real_plus_artaug.keras
README.md
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch:

```bash
python app.py
```

Open the Gradio URL displayed in the terminal.

The application supports image upload, camera input, five-class classification, confidence scores, and Grad-CAM explanations.

## Ethics and Limitations

Users should upload only images they own or have permission to use. The private photographic dataset is not distributed publicly.

Synthetic images were used for training only, never for validation or testing. Grad-CAM provides an approximate explanation and may highlight irrelevant background regions.

The ArtAug pilot used only five accepted synthetic examples, so further experiments are needed before claiming reliable performance gains.
