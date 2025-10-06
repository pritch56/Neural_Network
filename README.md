# Neural Network Projects

This repository contains two main neural network applications:

## 1. Neural Style Transfer App

A Python app that uses PyTorch and VGG19 to combine the content of one image with the style of another.

### Requirements

- Python 3.7+
- See `requirements.txt` for dependencies.

### Usage (Command Line)

```bash
pip install -r requirements.txt
python cli.py --content path/to/content.jpg --style path/to/style.jpg --output path/to/output.jpg
```

- `--content`: Path to the content image.
- `--style`: Path to the style image.
- `--output`: Path to save the stylized image.
- `--steps`: (Optional) Number of optimization steps (default: 300).

### Optional: Streamlit Front-End

To run the web UI:

```bash
streamlit run streamlit_app.py
```

Upload a content and style image, then click "Run Style Transfer".

### Notes

- The app uses VGG19 pretrained on ImageNet.
- Output image is saved and displayed after processing.

## 2. Neural Network Image Classifier

A convolutional neural network implementation for image classification tasks using PyTorch.

### Features

- Custom CNN architecture with configurable layers
- Support for multiple image classification datasets
- Training and evaluation pipelines
- Data augmentation and preprocessing
- Model checkpointing and loading
- Visualization of training metrics and predictions

### Usage

```bash
# Train a new model
python image_classifier.py --train --dataset path/to/dataset --epochs 50 --batch_size 32

# Evaluate existing model
python image_classifier.py --evaluate --model path/to/model.pth --test_data path/to/test

# Predict single image
python image_classifier.py --predict --model path/to/model.pth --image path/to/image.jpg
```

### Model Architecture

- Convolutional layers with ReLU activation
- Batch normalization for stable training
- Dropout for regularization
- Adaptive pooling for flexible input sizes
- Fully connected layers for classification

### Supported Features

- Multi-class classification
- Transfer learning with pretrained models
- Custom loss functions and optimizers
- Learning rate scheduling
- Early stopping and model validation
