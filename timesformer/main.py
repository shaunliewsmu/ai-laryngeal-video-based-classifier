# timesformer/main.py

import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import cv2
from pathlib import Path

from timesformer_classifier.data_config.dataloader import create_dataloaders
from timesformer_classifier.models.timesformer_model import create_model
from timesformer_classifier.trainers.trainer import ModelTrainer
from timesformer_classifier.evaluators.evaluator import ModelEvaluator
from timesformer_classifier.utils.logger import ExperimentLogger
from timesformer_classifier.utils.visualization import TrainingVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='TimeSformer Transformer Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to training/validation dataset directory')
    parser.add_argument('--test_data_dir', type=str, default=None,
                      help='Path to test dataset directory. If not provided, will use data_dir')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='timesformer-models',
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
    parser.add_argument('--model_name', type=str, default="facebook/timesformer-base-finetuned-k400",
                      help='Model name or path')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=40,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
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
    exp_logger = ExperimentLogger(args.log_dir, prefix=f'timesformer-classifier')
    logger = exp_logger.get_logger()
    logger.info("Starting TimeSformer Transformer training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Set memory optimization for CUDA
        if torch.cuda.is_available():
            # Reduce memory fragmentation
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA memory optimization enabled")
            
        # Create dataloaders with different sampling methods for each split
        logger.info("Creating dataloaders...")
        sampling_methods = {
            'train': args.train_sampling,
            'val': args.val_sampling,
            'test': args.test_sampling
        }
        dataloaders, class_labels = create_dataloaders(args, sampling_methods, logger)
        logger.info(f"Dataloaders created successfully with classes: {class_labels}")
        logger.info(f"Using sampling methods: {sampling_methods}")
        
        # Create visualization directory
        viz_dir = exp_logger.get_experiment_dir() / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize sampling methods for each split
        visualizer = TrainingVisualizer(exp_logger.get_experiment_dir())
        
        # Visualize a few sample videos from each split
        for split, dataloader in dataloaders.items():
            dataset = dataloader.dataset
            sample_count = min(3, len(dataset.video_paths))  # At most 3 videos per split
            
            logger.info(f"Creating sampling visualizations for {split} split...")
            for i in range(sample_count):
                video_path = dataset.video_paths[i]
                sampling_method = sampling_methods[split]
                
                # Create the visualization filename
                video_name = Path(video_path).stem
                save_path = viz_dir / f'{split}_{sampling_method}_{video_name}.png'
                
                try:
                    visualizer.visualize_sampling(
                        video_path,
                        sampling_method,
                        args.num_frames,
                        save_path,
                        f"{split.capitalize()} - {video_name}"
                    )
                    logger.info(f"Created sampling visualization for {video_name}")
                    
                    # Check if this video has fewer frames than requested
                    cap = cv2.VideoCapture(str(video_path))
                    # Check if this video has fewer frames than requested
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if total_frames < args.num_frames:
                        logger.info(f"Note: Video {video_name} has {total_frames} frames, "
                                  f"which is less than the requested {args.num_frames} frames.")
                except Exception as e:
                    logger.error(f"Error creating visualization for {video_name}: {str(e)}")
        
        # Create model
        model = create_model(
            args.model_name,
            args.num_classes,
            class_labels,
            args.num_frames,
            device,
            logger
        )
        
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
        trained_model = trainer.train()
        logger.info("Model training completed")
        
        # Evaluate model
        logger.info("Starting model evaluation on test set...")
        evaluator = ModelEvaluator(
            trained_model,
            dataloaders['test'],
            device,
            args,
            exp_logger
        )
        auroc, f1, conf_matrix = evaluator.evaluate()
        logger.info(f"Model evaluation completed. AUROC: {auroc:.4f}, F1: {f1:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Training and evaluation pipeline completed successfully")

if __name__ == "__main__":
    main()

"""
Example usage:
python3 timesformer/main.py \
  --data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
  --test_data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
  --log_dir logs \
  --model_dir timesformer-models \
  --train_sampling random \
  --val_sampling uniform \
  --test_sampling uniform \
  --num_frames 32 \
  --batch_size 2 \
  --epochs 40 \
  --learning_rate 0.001 \
  --weight_decay 0.01 \
  --num_workers 2 \
  --patience 7
"""