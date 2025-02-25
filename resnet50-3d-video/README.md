# 3D ResNet-50 for Laryngeal Cancer Screening

This repository implements a 3D ResNet-50 model for classifying laryngeal endoscopy videos into "referral" and "non-referral" categories, designed to assist in triaging patients for laryngeal cancer screening.

## Overview

- Implements a 3D ResNet-50 architecture using PyTorchVideo
- Supports different frame sampling strategies (uniform, random, sliding window)
- Includes complete training, evaluation, and inference pipelines
- Provides visualization tools for analyzing model performance

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- pytorchvideo
- scikit-learn
- matplotlib
- seaborn
- tqdm

You can install the required packages using pip:

```bash
pip install torch torchvision pytorchvideo scikit-learn matplotlib seaborn tqdm
```

## Project Structure

```
resnet50video-pytorchvideo/
├── main.py                       # Main training script
├── inference.py                  # Inference script for single videos
├── video_classifier/
│   ├── data_config/              # Dataset and dataloader configuration
│   ├── evaluators/               # Model evaluation utilities
│   ├── models/                   # Model architecture definitions
│   ├── trainers/                 # Training loop implementations
│   └── utils/                    # Utility functions (logging, visualization)
```

## How to Run

### Training

To train a new model, use the `main.py` script:

```bash
python3 resnet50-3d-video/main.py \
  --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
  --test_data_dir artifacts/laryngeal_dataset_iqm_filtered:v0/dataset \
  --log_dir logs \
  --model_dir resnet50-models \
  --train_sampling random \
  --val_sampling uniform \
  --test_sampling uniform \
  --num_frames 32 \
  --fps 2 \
  --stride 0.5 \
  --batch_size 2 \
  --epochs 40 \
  --learning_rate 0.01 \
  --num_workers 2 \
  --patience 7
```

#### Important Parameters:

- `--data_dir`: Path to the training/validation dataset
- `--test_data_dir`: Path to the test dataset (optional)
- `--log_dir`: Directory to save logs and visualizations
- `--model_dir`: Directory to save model checkpoints
- `--train_sampling`: Sampling method for training clips ('random', 'uniform', 'sliding')
- `--num_frames`: Number of frames to sample from each video
- `--fps`: Frames per second to sample
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--patience`: Early stopping patience (epochs without improvement)

The dataset directory should have the following structure:

```
dataset/
├── train/
│   ├── referral/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   └── non-referral/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── val/
│   ├── referral/
│   │   └── ...
│   └── non-referral/
│       └── ...
└── test/
    ├── referral/
    │   └── ...
    └── non-referral/
        └── ...
```

### Inference

To run inference on a single video using a trained model:

```bash
python3 resnet50-3d-video/inference.py \
  --video_path artifacts/laryngeal_dataset_balanced:v0/dataset/val/referral/0047.mp4 \
  --model_path model/20250220_175039_resnet50_best_model.pth \
  --num_frames 32 \
  --fps 2
```

#### Important Parameters:

- `--video_path`: Path to the input video file
- `--model_path`: Path to the trained model checkpoint
- `--num_frames`: Number of frames to sample (should match training configuration)
- `--fps`: Frames per second (should match training configuration)

## Output and Visualizations

During training, the following outputs are generated:

- Training logs (saved to the specified log directory)
- Training history plots (loss and accuracy)
- Confusion matrix visualization
- Sample predictions visualization
- Best model checkpoint

During inference, the model produces:

- Classification result (referral or non-referral)
- Confidence score for the prediction
- JSON file with inference results

## Example Results

After successful inference, you'll see output like:

```
Results:
Class: referral
Confidence: 0.9245
```

The detailed results are saved in a JSON file in the logs directory.

## Model Architecture

The implemented model is a 3D ResNet-50 using PyTorchVideo's implementation. Key features include:

- 3D convolutions for spatio-temporal feature extraction
- 50-layer deep residual architecture
- Binary classification head (referral vs. non-referral)
- Optimized for medical video analysis

## Training Features

- Early stopping to prevent overfitting
- Visualization of training metrics
- Support for multiple clip sampling strategies
- Multi-GPU training support
- Consistent evaluation metrics (AUROC, F1 Score)

## Notes

- For best performance, use a GPU with sufficient memory
- The model performance depends on the quality of the input videos
- The inference script is designed to process single videos for quick evaluation
- Adjust the frame sampling parameters based on the typical length of your videos