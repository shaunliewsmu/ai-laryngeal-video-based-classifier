# TimeSformer Transformer for Laryngeal Cancer Screening

This repository implements a [TimeSformer (Time-Space Transformer)](https://huggingface.co/docs/transformers/model_doc/timesformer) model for the classification of laryngeal endoscopy videos into "referral" and "non-referral" categories. The model is designed to assist in triaging patients for laryngeal cancer screening, particularly in low-resource settings.

## Overview

- Implements the TimeSformer architecture using Hugging Face Transformers
- Supports different frame sampling methods using PyTorchVideo samplers
- Includes training, evaluation, and inference pipelines
- Provides visualization tools for model performance analysis

## Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers
- pytorchvideo
- av (PyAV)
- scikit-learn
- matplotlib
- seaborn
- tqdm

You can install the required packages using pip:

```bash
pip install torch torchvision pytorchvideo transformers av scikit-learn matplotlib seaborn tqdm
```

## Project Structure

```
timesformer_transformer/
├── main.py                          # Main training script
├── inference.py                     # Inference script for running on single videos
├── timesformer_classifier/
│   ├── data_config/                 # Dataset and dataloader configuration
│   ├── evaluators/                  # Model evaluation utilities
│   ├── models/                      # Model architecture definitions
│   ├── trainers/                    # Training loop implementations
│   └── utils/                       # Utility functions (logging, visualization)
```

## How to Run

### Training

To train a new model, use the `main.py` script:

```bash
python timesformer_transformer/main.py \
  --data_dir artifacts/laryngeal_dataset_iqm_filtered:v0 \
  --test_data_dir artifacts/laryngeal_dataset_iqm_filtered:v0/dataset \
  --log_dir logs \
  --model_dir timesformer-models \
  --train_sampling random \
  --val_sampling uniform \
  --test_sampling uniform \
  --num_frames 32 \
  --fps 8 \
  --stride 0.5 \
  --batch_size 4 \
  --epochs 40 \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --num_workers 4 \
  --patience 7
```

#### Important Parameters:

- `--data_dir`: Path to the training/validation dataset
- `--test_data_dir`: Path to the test dataset (optional)
- `--log_dir`: Directory to save logs and visualizations
- `--model_dir`: Directory to save model checkpoints
- `--train_sampling`: Sampling method for training ('uniform', 'random', 'sliding')
- `--val_sampling`: Sampling method for validation ('uniform', 'random', 'sliding')
- `--test_sampling`: Sampling method for testing ('uniform', 'random', 'sliding')
- `--num_frames`: Number of frames to sample from each video
- `--fps`: Frames per second for clip sampling
- `--stride`: Stride fraction for sliding window sampling
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--patience`: Early stopping patience

### Frame Sampling Methods

This implementation uses PyTorchVideo's clip samplers to ensure consistent sampling across models:

1. **Uniform**: Frames are evenly sampled from the entire video at regular intervals
2. **Random**: A random continuous clip of specified duration is extracted from the video
3. **Sliding**: A sliding window approach with specified stride is used to extract multiple overlapping clips

The dataset directory should follow this structure:

```bash
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
python timesformer_transformer/inference.py \
  --video_path artifacts/laryngeal_dataset_iqm_filtered:v0/dataset/val/referral/0088_processed.mp4 \
  --model_path timesformer-models/20250227_123456_timesformer-classifier_best_model.pth \
  --num_frames 32 \
  --fps 8 \
  --sampling_method uniform \
  --stride 0.5 \
  --save_viz
```

## Model Architecture

TimeSformer (Time-Space Transformer) is a video understanding model that divides self-attention into two operations:
1. Spatial attention: applied to frames independently
2. Temporal attention: applied across frames

This division of attention makes the model computationally efficient while still being able to model complex spatio-temporal relationships within video data. The model adapts the Vision Transformer (ViT) architecture to video understanding by enabling spatiotemporal feature learning directly from frame-level patches.

## Performance Evaluation

The model is evaluated using several metrics:
- Accuracy: Overall classification accuracy
- F1 Score: Balance between precision and recall
- AUROC: Area under the Receiver Operating Characteristic curve
- Confusion Matrix: Detailed breakdown of predictions by class

## Troubleshooting

If you encounter CUDA out-of-memory errors:
- Reduce the batch size (`--batch_size`)
- Reduce the number of frames (`--num_frames`)
- Set `--num_workers` to 0

For data format errors, ensure your videos:
- Have sufficient frames (at least equal to `num_frames`)
- Are in a standard format (MP4 with H.264 encoding is recommended)
- Have consistent frame dimensions (resize if necessary)