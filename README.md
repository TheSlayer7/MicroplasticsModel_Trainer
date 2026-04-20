# Microplastics Detection

This repository contains a microplastics detection pipeline built around a residual U-Net-style segmentation model. It can train a model, run inference on microscopy images, and export visual detections as bounding boxes.

## Highlights

- Segmentation-based microplastics detection.
- Training and inference scripts are included.
- A bundled model artifact is provided for immediate testing.
- Detection outputs both a rendered image and summary stats.

## Repository Layout

- `train_model.py`: trains the segmentation model and writes the best checkpoint.
- `det_p.py`: loads a trained model and runs detection on a single image.
- `standardize_masks.py`: optional utility for normalizing mask values and polarity.
- `myModel.keras`: bundled model artifact.
- `my_custom_resunet.keras`: trained model checkpoint produced by training.
- `detection_calibration.json`: saved threshold calibration used by inference.

## Requirements

Use Python 3.10+ and install the runtime dependencies:

```bash
pip install tensorflow numpy matplotlib opencv-python tifffile
```

If you are using a virtual environment, create and activate it first.

## Quick Start

1. Clone the repository.
2. Install the dependencies.
3. Run training or inference from the project root.

## Training

Train the model with:

```bash
python train_model.py
```

Training writes these outputs:

- `my_custom_resunet.keras`
- `custom_resunet_training.png`
- `detection_calibration.json`

If you want to standardize mask files before training, run:

```bash
python standardize_masks.py --masks-dir data/masks --output-dir data/masks_standardized
```

## Inference

Run detection on a single image with:

```bash
python det_p.py --image path/to/image.png
```

Optional flags:

- `--model path/to/model.keras`: use a specific model file.
- `--domain auto|clam|spiked`: choose the inference profile.
- `--no-calibration`: ignore `detection_calibration.json`.
- `--score-threshold 0.15`: override the base score threshold.
- `--color-threshold 0.05`: override the base color threshold.
- `--output detected_boxes.png`: set the output image path.

Inference writes:

- `detected_boxes.png`

## Notes

- The repository is set up for microplastics detection and visualization, not for a specific dataset walkthrough.
- If you train your own data, make sure image and mask pairs follow the filename stem convention used by the scripts.
- The detector will prefer the saved calibration file when available.

## License

This project is released under the MIT License. See `LICENSE` for the full text.

Third-party library credits and license notices are listed in `THIRD_PARTY_LICENSES.md`.