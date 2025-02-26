# Video Swin Transformer for Laryngeal Cancer Screening

This repository implements a [3D Swin Transformer](https://pytorch.org/vision/main/models/video_swin_transformer.html) model for the classification of laryngeal endoscopy videos into "referral" and "non-referral" categories. The model is designed to assist in triaging patients for laryngeal cancer screening, particularly in low-resource settings.

## Overview

- Implements a Video Swin Transformer (Swin3D) architecture
- Supports multiple model variants (tiny, small, base)
- Includes training, evaluation, and inference pipelines
- Provides visualization tools for model performance analysis

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
videoswintransformer/
├── main.py                           # Main training script
├── inference.py                      # Inference script for running on single videos
├── swin_video_classifier/
│   ├── data_config/                  # Dataset and dataloader configuration
│   ├── evaluators/                   # Model evaluation utilities
│   ├── models/                       # Model architecture definitions
│   ├── trainers/                     # Training loop implementations
│   └── utils/                        # Utility functions (logging, visualization)
```

## How to Run

### Training

To train a new model, use the `main.py` script:

```bash
python3 videoswintransformer/main.py \
  --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
  --test_data_dir artifacts/laryngeal_dataset_iqm_filtered:v0/dataset \
  --log_dir logs \
  --model_dir swin3d-models \
  --model_size tiny \
  --pretrained \
  --train_sampling random \
  --val_sampling uniform \
  --test_sampling uniform \
  --num_frames 32 \
  --fps 8 \
  --stride 0.5 \
  --batch_size 2 \
  --epochs 40 \
  --learning_rate 0.0001 \
  --weight_decay 0.05 \
  --num_workers 2 \
  --patience 7
```

#### Important Parameters:

- `--data_dir`: Path to the training/validation dataset
- `--test_data_dir`: Path to the test dataset (optional)
- `--log_dir`: Directory to save logs and visualizations
- `--model_dir`: Directory to save model checkpoints
- `--model_size`: Size of Swin3D model (tiny, small, base, base_in22k)
- `--pretrained`: Use pretrained weights
- `--num_frames`: Number of frames to sample from each video
- `--fps`: Frames per second to sample
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--patience`: Early stopping patience

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
python videoswintransformer/inference.py \
  --video_path artifacts/laryngeal_dataset_balanced:v0/dataset/val/referral/0047.mp4 \
  --model_path swin3d-models/20250225_162321_swin3d-tiny_best_model.pth \
  --model_size tiny \
  --num_frames 32 \
  --fps 8
```

#### Important Parameters:

- `--video_path`: Path to the input video file
- `--model_path`: Path to the trained model checkpoint
- `--model_size`: Size of the Swin3D model (should match the trained model)
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

```bash
Results:
Class: referral
Confidence: 0.9245
```

The detailed results are saved in a JSON file in the logs directory.

## Notes

- For best performance, use a GPU with sufficient memory
- The model performance depends on the quality of the input videos
- Preprocessing steps like glottis detection can improve accuracy
- The inference script is configured to use GPU 1 by default - modify this if needed