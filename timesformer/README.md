# ViViT Transformer for Laryngeal Cancer Screening

This repository implements a [Vision Video Transformer (ViViT)](https://huggingface.co/docs/transformers/model_doc/vivit) model for the classification of laryngeal endoscopy videos into "referral" and "non-referral" categories. The model is designed to assist in triaging patients for laryngeal cancer screening, particularly in low-resource settings.

## Overview

- Implements the ViViT architecture using Hugging Face Transformers
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
vivittransformer/
├── main.py                          # Main training script
├── inference.py                     # Inference script for running on single videos
├── vivit_classifier/
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
python vivittransformer/main.py \
  --data_dir artifacts/laryngeal_dataset_iqm_filtered:v0 \
  --test_data_dir artifacts/laryngeal_dataset_iqm_filtered:v0/dataset \
  --log_dir logs \
  --model_dir vivit-models \
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

The sampling method controls how frames are selected from videos, which can affect model performance. For example:
- 'uniform' is best for capturing the overall video content
- 'random' provides more variability during training
- 'sliding' with a small stride can provide higher temporal resolution for detecting fast movements

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
python vivittransformer/inference.py \
  --video_path artifacts/laryngeal_dataset_iqm_filtered:v0/dataset/val/referral/0088_processed.mp4 \
  --model_path vivit-models/20250227_123456_vivit-classifier_best_model.pth \
  --num_frames 32 \
  --fps 8 \
  --sampling_method uniform \
  --stride 0.5 \
  --save_viz
```

#### Important Parameters:

- `--video_path`: Path to the input video file
- `--model_path`: Path to the trained model checkpoint
- `--num_frames`: Number of frames to sample (should match training configuration)
- `--fps`: Frames per second for clip sampling
- `--sampling_method`: Method to sample frames from video ('uniform', 'random', 'sliding')
- `--stride`: Stride fraction for sliding window sampling
- `--save_viz`: Flag to save visualization of sampled frames

## Output and Visualizations

During training, the following outputs are generated:

- Training logs (saved to the specified log directory)
- Training history plots (loss and accuracy)
- Confusion matrix visualization
- ROC curve visualization
- Best model checkpoint

During inference, the model produces:

- Classification result (referral or non-referral)
- Confidence score for the prediction
- JSON file with inference results
- Optional visualization of sampled frames

## Example Results

After successful inference, you'll see output like:

```bash
Results:
Class: referral
Confidence: 0.9245
```

The detailed results are saved in a JSON file in the logs directory.

## Model Architecture

The ViViT (Video Vision Transformer) model extends the Vision Transformer (ViT) architecture for video understanding. It applies self-attention mechanisms to both spatial and temporal dimensions of the input video, allowing it to model complex relationships between frames. The model can process sequences of frames and capture motion patterns that are essential for accurate video classification.

## Performance Evaluation

The model is evaluated using several metrics:
- Accuracy: Overall classification accuracy
- F1 Score: Balance between precision and recall
- AUROC: Area under the Receiver Operating Characteristic curve
- Confusion Matrix: Detailed breakdown of predictions by class

## Notes

- For best performance, use a GPU with sufficient memory
- The model performance may vary depending on the sampling method used
- Videos with fewer frames than required will be skipped or have frames sampled with replacement
- The stride parameter is used only for the 'sliding' sampling method
- Using the same sampling method during inference as used during training generally yields better results
