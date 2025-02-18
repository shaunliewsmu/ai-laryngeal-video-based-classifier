# ResNet50-LSTM Video Classifier for Laryngeal Cancer Screening

This repository contains an implementation of a deep learning model that combines ResNet50 and LSTM for video-based laryngeal cancer screening. The model processes laryngoscopy videos and classifies them into referral and non-referral cases.

## Model Architecture

The model consists of two main components:

1. **Feature Extractor**: Uses a pretrained ResNet3D-18 (initialized with Kinetics-400 weights) to extract spatial-temporal features from video frames.
2. **Sequence Processor**: An LSTM network that processes the temporal sequence of features to make the final classification.

Key features:
- Input: Video clips of 32 frames
- Frame sampling strategies: uniform, random, or sliding window
- Binary classification: referral vs. non-referral
- Class-weighted loss function for handling imbalanced data
- Early stopping and learning rate scheduling

## Project Structure

```
resnet50video-lstm/
├── src/
│   ├── config/          # Configuration settings
│   ├── data/            # Dataset and data loading
│   ├── models/          # Model architecture
│   ├── utils/           # Utility functions
│   └── trainer/         # Training logic
└── main.py             # Training script
```

## Requirements

```
torch
torchvision
opencv-python
numpy
wandb
tqdm
scikit-learn
matplotlib
seaborn
```

## Training the Model

1. Prepare your dataset in the following structure:
```
dataset/
├── train/
│   ├── referral/
│   └── non_referral/
├── val/
│   ├── referral/
│   └── non_referral/
└── test/
    ├── referral/
    └── non_referral/
```

2. Run the training script:
```bash
python resnet50video-lstm/main.py \
    --data_dir path/to/dataset \
    --log_dir logs \
    --model_dir models \
    --train_sampling uniform \
    --val_sampling uniform \
    --test_sampling uniform
```

### Training Arguments

- `--data_dir`: Path to the dataset directory
- `--log_dir`: Directory for saving logs and visualizations
- `--model_dir`: Directory for saving model checkpoints
- `--train_sampling`: Frame sampling method for training (uniform/random/sliding)
- `--val_sampling`: Frame sampling method for validation
- `--test_sampling`: Frame sampling method for testing

## Model Performance Tracking

The training process is tracked using Weights & Biases, monitoring:
- Training and validation losses
- AUROC scores
- Accuracy, precision, recall, and F1 scores
- Confusion matrices
- Learning rate changes