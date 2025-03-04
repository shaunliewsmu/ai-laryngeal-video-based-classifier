# ResNet50-LSTM Video Classifier for Laryngeal Cancer Screening

This repository contains an implementation of a deep learning model that combines ResNet50 and LSTM for video-based laryngeal cancer screening. The model processes laryngoscopy videos and classifies them into referral and non-referral cases.

## Model Architecture

The model consists of two main components:

1. **Feature Extractor**: Uses a pretrained ResNet50 (initialized with ImageNet weights) to extract frame-level features.
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
python3 resnet50-2d-lstm/main.py \
    --data_dir path/to/train_and_val_dataset \
    --test_dir path/to/test_dataset \
    --log_dir logs \
    --model_dir models \
    --train_sampling uniform \
    --val_sampling uniform \
    --test_sampling uniform
```

## Sampling Methods

The model supports three different frame sampling strategies:

1. **Uniform Sampling**: Samples frames at regular intervals across the video, ensuring equal spacing between frames. This is useful for getting a consistent view of the entire video.

2. **Random Sampling**: Randomly selects frames from the entire video. This introduces randomness that can help with model generalization.

3. **Random Window Sampling**: Divides the video into equal segments and randomly samples one frame from each segment. This ensures the model sees frames from throughout the video while maintaining some randomness.

### Training Arguments

- `--data_dir`: Path to the dataset directory for training and validation data
- `--test_dir`: Path to the separate test dataset directory (optional, uses data_dir if not specified)
- `--log_dir`: Directory for saving logs and visualizations
- `--model_dir`: Directory for saving model checkpoints
- `--train_sampling`: Frame sampling method for training (uniform/random/random_window)
- `--val_sampling`: Frame sampling method for validation
- `--test_sampling`: Frame sampling method for testing
- `--loss_weight`: Weight for loss in model selection (0-1). Higher values prioritize minimizing loss over maximizing AUROC (default: 0.3)

## Model Performance Tracking

The training process is tracked using Weights & Biases, monitoring:
- Training and validation losses
- AUROC scores
- Accuracy, precision, recall, and F1 scores
- Confusion matrices
- Learning rate changes