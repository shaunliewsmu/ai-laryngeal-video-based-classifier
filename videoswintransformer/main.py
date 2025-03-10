import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import cv2
from pathlib import Path

from swin_video_classifier.data_config.dataloader import create_dataloaders
from swin_video_classifier.models.swin3d import create_model
from swin_video_classifier.trainers.trainer import ModelTrainer
from swin_video_classifier.evaluators.evaluator import ModelEvaluator
from swin_video_classifier.utils.logger import ExperimentLogger
from swin_video_classifier.utils.visualization import TrainingVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Video Swin Transformer Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to training/validation dataset directory')
    parser.add_argument('--test_data_dir', type=str, default=None,
                      help='Path to test dataset directory. If not provided, will use data_dir')
    parser.add_argument('--log_dir', type=str, required=True,
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save models')
    
    # Dataset and sampling parameters
    parser.add_argument('--train_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for training')
    parser.add_argument('--val_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for validation')
    parser.add_argument('--test_sampling', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Sampling method for testing')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='Number of frames to sample')
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='tiny',
                      choices=['tiny', 'small', 'base', 'base_in22k'],
                      help='Size of Swin Transformer model')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained weights')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=7,
                      help='Early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                      help='Early stopping delta')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup experiment logging
    exp_logger = ExperimentLogger(args.log_dir, prefix=f'swin3d-{args.model_size}')
    logger = exp_logger.get_logger()
    logger.info("Starting Video Swin Transformer training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Set device
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create model
        model = create_model(
            logger, 
            model_size=args.model_size, 
            pretrained=args.pretrained,
            num_classes=args.num_classes
        )
        
        # Handle multi-GPU
        # if torch.cuda.device_count() > 1:
        #     logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)
        model = model.to(device)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        dataloaders = create_dataloaders(args, logger)
        logger.info("Dataloaders created successfully")
        
        # Create visualization directory
        viz_dir = exp_logger.get_experiment_dir() / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize sampling methods for each split
        visualizer = TrainingVisualizer(exp_logger.get_experiment_dir())
        
        # Visualize sampling methods for one example video from each split
        sampling_methods = {
            'train': args.train_sampling,
            'val': args.val_sampling,
            'test': args.test_sampling
        }
        
        for split, sampling_method in sampling_methods.items():
            # Find an example video
            split_dir = Path(args.data_dir) / split
            if not split_dir.exists() and split == 'test' and args.test_data_dir:
                split_dir = Path(args.test_data_dir) / split
                
            if split_dir.exists():
                # Get first video found
                video_path = None
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        videos = list(class_dir.glob('*.mp4'))
                        if videos:
                            video_path = videos[0]
                            break
                
                if video_path:
                    # Create the visualization
                    save_path = viz_dir / f'sampled_frames_{split}_{sampling_method}.png'
                    try:
                        visualizer.visualize_sampling(
                            video_path,
                            sampling_method,
                            args.num_frames,
                            save_path,
                            f"{split.capitalize()} Split"
                        )
                        logger.info(f"Created sampling visualization for {split} split at {save_path}")
                        
                        # Check if this video needed dynamic FPS adjustment
                        cap = cv2.VideoCapture(video_path)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        if total_frames < args.num_frames:
                            logger.info(f"Note: The example video for {split} split has {total_frames} frames, "
                                     f"which is less than the requested {args.num_frames} frames. "
                                     f"Dynamic FPS adjustment was applied.")
                    except Exception as e:
                        logger.error(f"Error creating sampling visualization for {split}: {str(e)}")
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        logger.info(f"Using AdamW optimizer with learning rate {args.learning_rate}")
        
        # Train model
        logger.info("Starting model training...")
        trainer = ModelTrainer(
            model, dataloaders, criterion, optimizer, device, args, exp_logger
        )
        model = trainer.train()
        logger.info("Model training completed")
        
        # Evaluate model
        logger.info("Starting model evaluation on test set...")
        evaluator = ModelEvaluator(
            model,
            dataloaders['test'],
            device,
            args,
            exp_logger
        )
        auroc, f1, conf_matrix = evaluator.evaluate()
        logger.info("Model evaluation completed")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Training and evaluation pipeline completed successfully")

if __name__ == "__main__":
    main()
    
"""
python3 videoswintransformer/main.py \
  --data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
  --test_data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
  --log_dir logs \
  --model_dir swin3d-models \
  --model_size tiny \
  --pretrained \
  --train_sampling uniform \
  --val_sampling uniform \
  --test_sampling uniform \
  --num_frames 32 \
  --batch_size 2 \
  --epochs 30 \
  --learning_rate 0.0001 \
  --weight_decay 0.05 \
  --num_workers 2 \
  --patience 7
"""