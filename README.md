# Microplastics Model Trainer

This project trains and runs a deep learning pipeline for microplastics detection in microscopy images.

The training script builds a residual U-Net-style segmentation model, learns from paired image and mask folders, and saves the best checkpoint as `my_custom_resunet.keras`. The inference script loads that model, predicts a segmentation mask, extracts contours, and draws bounding boxes around detected plastic regions.

## Project Files

- `train_model.py`: trains the segmentation model and saves training plots.
- `det_p.py`: loads the trained model and runs box-based detection on a test image.
- `myModel.keras`: bundled model artifact included in the repository.

## Requirements

The scripts import and use these third-party libraries:

- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (`cv2`)

## Configuration

Before training, update the directory paths in `train_model.py`:

- `IMAGES_DIR`: folder containing input microscopy images.
- `MASKS_DIR`: folder containing the matching segmentation masks.

Before running inference, update the test image path in `det_p.py`:

- `TEST_IMAGE_PATH`: path to the image you want to analyze.

## Training Output

Running `train_model.py` produces these artifacts:

- `my_custom_resunet.keras`: best model checkpoint saved during training.
- `custom_resunet_training.png`: training and validation loss / Dice plots.

## Inference Output

Running `det_p.py` produces:

- `detected_boxes.png`: the input image with detected plastic regions marked by bounding boxes.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

Third-party dependency notices are collected in `THIRD_PARTY_LICENSES.md`.