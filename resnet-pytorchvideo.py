import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorchvideo.data.clip_sampling import ClipInfo
from pytorchvideo.data import (
    make_clip_sampler,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    RandomShortSideScale,
    ShortSideScale,
    Normalize,
)
from pytorchvideo.models.resnet import create_resnet

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import random
import logging
from datetime import datetime
import json
import sys

def setup_logger(log_dir):
    """Set up logger with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('VideoClassifier')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_{timestamp}.log')
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', sampling_method='uniform', 
                 num_frames=32, fps=30, stride=0.5, logger=None):
        """
        Args:
            root_dir (str): Root directory of dataset
            mode (str): 'train', 'val', or 'test'
            sampling_method (str): 'uniform', 'random', or 'sliding'
            num_frames (int): Number of frames to sample
            fps (int): Frames per second of videos 
            stride (float): Stride fraction for sliding window (overlap = 1 - stride)
            logger (logging.Logger): Logger instance
        """
        self.root_dir = Path(root_dir) / mode
        self.mode = mode
        self.num_frames = num_frames
        self.sampling_method = sampling_method
        self.logger = logger
        self.fps = fps
        self.video_paths = []
        self.labels = []
        
        # Get all video paths and labels
        for class_path in self.root_dir.iterdir():
            if class_path.is_dir():
                label = 1 if class_path.name == 'referral' else 0
                for video_path in class_path.glob('*.mp4'):
                    self.video_paths.append(str(video_path))
                    self.labels.append(label)
        
        if logger:
            logger.info(f"Found {len(self.video_paths)} videos for {mode} split")
            logger.info(f"Class distribution: {sum(self.labels)} referral, "
                       f"{len(self.labels) - sum(self.labels)} non-referral")
        
        # Calculate clip duration
        self.clip_duration = float(num_frames) / fps  # Duration in seconds
        
        # Create clip sampler based on sampling method
        if sampling_method == 'sliding':
            # For sliding window, use uniform sampler with smaller stride
            self.stride = self.clip_duration * stride
            self.clip_sampler = make_clip_sampler(
                'uniform',
                self.clip_duration,
                self.stride
            )
            logger.info(f"Using sliding window with clip duration: {self.clip_duration:.2f}s, "
                       f"stride: {self.stride:.2f}s ({stride*100:.0f}% overlap)")
        else:
            # For uniform or random sampling
            self.clip_sampler = make_clip_sampler(
                sampling_method,
                self.clip_duration
            )
            
        # Define transforms
        if mode == 'train':
            self.transform = ApplyTransformToKey(
                key="video",
                transform=transforms.Compose([
                    UniformTemporalSubsample(num_frames),
                    RandomShortSideScale(min_size=256, max_size=320),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
                ]),
            )
        else:
            self.transform = ApplyTransformToKey(
                key="video",
                transform=transforms.Compose([
                    UniformTemporalSubsample(num_frames),
                    ShortSideScale(256),
                    transforms.CenterCrop(224),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
                ]),
            )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """Get video clips and labels."""
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]
            
            # Load video
            video = EncodedVideo.from_path(video_path)
            
            # Get video duration
            duration = video.duration
            if duration is None:
                duration = 10.0
                
            # Sample clips using the clip sampler
            clip_start_sec = 0
            clips = []
            labels = []
            
            while True:
                # Get next clip
                clip_info = self.clip_sampler(clip_start_sec, duration, None)
                if clip_info is None:
                    break
                    
                # Extract the clip
                try:
                    video_data = video.get_clip(
                        clip_info.clip_start_sec, 
                        clip_info.clip_end_sec
                    )
                    
                    # Apply transforms
                    if self.transform is not None:
                        video_data = self.transform(video_data)
                    
                    clips.append(video_data["video"])
                    labels.append(label)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting clip at {clip_info.clip_start_sec}: {str(e)}")
                    continue
                
                # Update start time for next clip
                clip_start_sec = clip_info.clip_end_sec
                
                # Break if this was the last clip
                if clip_info.is_last_clip:
                    break
            
            if not clips:
                raise ValueError(f"No valid clips extracted from video {video_path}")
            
            # Stack all clips along batch dimension
            clips = torch.stack(clips)
            labels = torch.tensor(labels, dtype=torch.long)
            
            return clips, labels
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading video {video_path}: {str(e)}")
            raise
        
def video_collate_fn(batch):
    """Custom collate function for video batches."""
    # Find minimum number of clips across batch to ensure consistent size
    min_clips = min(sample_clips.size(0) for sample_clips, _ in batch)
    
    # Separate clips and labels
    clips = []
    labels = []
    
    for sample_clips, sample_labels in batch:
        # Take only min_clips number of clips from each video
        clips.append(sample_clips[:min_clips])
        labels.append(sample_labels[:min_clips])
    
    # Stack along batch dimension
    clips = torch.stack(clips, dim=0)  # (B, min_clips, C, T, H, W)
    labels = torch.stack(labels, dim=0)
    
    return clips, labels

def create_dataloaders(args, logger):
    """Create data loaders for train, validation, and test sets."""
    sampling_methods = {
        'train': args.train_sampling,
        'val': args.val_sampling,
        'test': args.test_sampling
    }
    
    logger.info("Creating datasets with the following sampling methods:")
    for split, method in sampling_methods.items():
        logger.info(f"{split}: {method}")
    
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            # Create dataset
            datasets[split] = VideoDataset(
                args.data_dir,
                mode=split,
                sampling_method=sampling_methods[split],
                num_frames=args.num_frames,
                fps=args.fps,
                stride=args.stride,
                logger=logger
            )
            
            # Create dataloader
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=args.batch_size,
                shuffle=(split == 'train'),
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=video_collate_fn
            )
            
            logger.info(
                f"Created {split} dataloader with {len(dataloaders[split])} "
                f"batches (batch size: {args.batch_size})"
            )
            
        except Exception as e:
            logger.error(f"Error creating {split} dataset/dataloader: {str(e)}")
            raise
    
    return dataloaders

def train_model(model, dataloaders, criterion, optimizer, device, args, logger):
    """Train the model."""
    best_val_acc = 0.0
    best_model_weights = None
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save training configuration
    config_path = os.path.join(args.log_dir, 'training_config.json')
    config = vars(args)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved training configuration to {config_path}")
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        logger.info('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            total_clips = 0
            
            for inputs, labels in dataloaders[phase]:
                try:
                    # Handle input shape (B, num_clips, C, T, H, W)
                    b, n, c, t, h, w = inputs.shape
                    inputs = inputs.view(b * n, c, t, h, w)  # Combine batch and clips
                    labels = labels.view(-1)  # Flatten labels accordingly
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_clips += inputs.size(0)
                    
                except Exception as e:
                    logger.error(f"Error during {phase} step: {str(e)}")
                    raise
            
            epoch_loss = running_loss / total_clips
            epoch_acc = running_corrects.double() / total_clips
            
            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_weights = model.state_dict().copy()
                
                # Save best model
                model_path = os.path.join(args.model_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_weights,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                }, model_path)
                logger.info(f"Saved best model to {model_path}")
    
    logger.info(f'Best val Acc: {best_val_acc:4f}')
    model.load_state_dict(best_model_weights)
    return model

def evaluate_model(model, dataloader, device, args, logger):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("Starting model evaluation...")
    
    try:
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Handle multiple clips per video
                batch_size, num_clips, c, t, h, w = inputs.shape
                inputs = inputs.view(-1, c, t, h, w)
                labels = labels.view(-1)
                
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # For evaluation, we can average predictions across clips
                preds = preds.view(batch_size, num_clips).float().mean(dim=1).round().long()
                probs = probs[:, 1].view(batch_size, num_clips).mean(dim=1)
                labels = labels.view(batch_size, num_clips)[:, 0]  # Take first label
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        auroc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Log metrics
        logger.info(f'AUROC: {auroc:.4f}')
        logger.info(f'F1 Score: {f1:.4f}')
        logger.info(f'Confusion Matrix:\n{conf_matrix}')
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = os.path.join(args.log_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {plot_path}")
        
        # Save detailed metrics
        metrics = {
            'auroc': float(auroc),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        metrics_path = os.path.join(args.log_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved detailed metrics to {metrics_path}")
        
        return auroc, f1, conf_matrix
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Classification Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--log_dir', type=str, required=True,
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save models')
    parser.add_argument('--train_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'sliding'],
                      help='Sampling method for training')
    parser.add_argument('--val_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'sliding'],
                      help='Sampling method for validation')
    parser.add_argument('--test_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'sliding'],
                      help='Sampling method for testing')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='Number of frames to sample from each video')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second in the videos')
    parser.add_argument('--stride', type=float, default=0.5,
                      help='Stride fraction for sliding window (overlap = 1 - stride)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main training and evaluation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    logger.info("Starting video classification training")
    logger.info(f"Arguments: {vars(args)}")
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Check GPU memory
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create model
        logger.info("Creating 3D ResNet-50 model...")
        model = create_resnet(
            # Model configs
            model_depth=50,
            model_num_class=2,  # Binary classification
            dropout_rate=0.5,
            
            # Input clip configs
            input_channel=3,
            
            # Normalization configs
            norm=nn.BatchNorm3d,
            
            # Activation configs
            activation=nn.ReLU,
            
            # Stem configs
            stem_dim_out=64,
            stem_conv_kernel_size=(3, 7, 7),
            stem_conv_stride=(1, 2, 2),
            stem_pool=nn.MaxPool3d,
            stem_pool_kernel_size=(1, 3, 3),
            stem_pool_stride=(1, 2, 2),
            
            # Stage configs
            stage_conv_a_kernel_size=((1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)),
            stage_conv_b_kernel_size=((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
            stage_conv_b_num_groups=(1, 1, 1, 1),
            stage_conv_b_dilation=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
            stage_spatial_h_stride=(1, 2, 2, 2),
            stage_spatial_w_stride=(1, 2, 2, 2),
            stage_temporal_stride=(1, 1, 1, 1),
            
            # Head configs
            head_pool=nn.AvgPool3d,
            head_pool_kernel_size=(4, 7, 7),
            head_output_size=(1, 1, 1),
            head_activation=None,
            head_output_with_global_average=True,
        )
        
        # Modify device setup for multi GPU setup
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            
        model = model.to(device)
        logger.info("Model created successfully")
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        dataloaders = create_dataloaders(args, logger)
        logger.info("Dataloaders created successfully")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        logger.info(f"Using Adam optimizer with learning rate {args.learning_rate}")
        
        # Train model
        logger.info("Starting model training...")
        model = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            device,
            args,
            logger
        )
        logger.info("Model training completed")
        
        # Evaluate model
        logger.info("Starting model evaluation on test set...")
        auroc, f1, conf_matrix = evaluate_model(
            model,
            dataloaders['test'],
            device,
            args,
            logger
        )
        logger.info("Model evaluation completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Training and evaluation pipeline completed successfully")

if __name__ == "__main__":
    main()
    
"""
python3 test.py \
    --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
    --log_dir logs \
    --model_dir model \
    --train_sampling uniform \
    --val_sampling uniform \
    --test_sampling uniform \
    --num_frames 32 \
    --fps 2 \
    --stride 0.5 \
    --batch_size 2 \
    --epochs 2 \
    --learning_rate 0.001 \
    --num_workers 2
"""