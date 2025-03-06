import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import os
from pathlib import Path
import cv2

from video_classifier.data_config.dataloader import create_dataloaders
from video_classifier.models.resnet3d import create_model
from video_classifier.trainers.trainer import ModelTrainer
from video_classifier.evaluators.evaluator import ModelEvaluator
from video_classifier.utils.logger import ExperimentLogger
from video_classifier.utils.visualization import TrainingVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to training/validation dataset directory')
    # for test dataset
    parser.add_argument('--test_data_dir', type=str, default=None,
                      help='Path to test dataset directory. If not provided, will use data_dir')
    parser.add_argument('--log_dir', type=str, required=True,
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save models')
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
    parser.add_argument('--patience', type=int, default=7,
                      help='Early stopping patience')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001,
                      help='Early stopping delta')
    parser.add_argument('--skip_train', action='store_true',
                      help='Skip training and just run evaluation')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to model checkpoint to load for evaluation')
    parser.add_argument('--weighted_sampling', action='store_true',
                      help='Use weighted sampling for imbalanced classes')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Setup experiment logging
    exp_logger = ExperimentLogger(args.log_dir)
    logger = exp_logger.get_logger()
    logger.info("Starting video classification training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Set device
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create model
        model = create_model(logger)
        
        # Handle multi-GPU
        # if torch.cuda.device_count() > 1:
        #     logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        #     model = nn.DataParallel(model)
        model = model.to(device)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        datasets, dataloaders = create_dataloaders(args, logger)
        logger.info("Dataloaders created successfully")
        
        # Create visualization directory
        viz_dir = exp_logger.get_experiment_dir() / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Visualize sampling methods for each split
        visualizer = TrainingVisualizer(exp_logger.get_experiment_dir())
        
        # Visualize sampling methods for one example video from each split
        for split, dataset in datasets.items():
            if len(dataset.video_paths) > 0:
                example_video = dataset.video_paths[0]
                sampling_method = getattr(args, f"{split}_sampling")
                
                # Create the visualization
                save_path = viz_dir / f'sampled_frames_{split}_{sampling_method}.png'
                try:
                    visualizer.visualize_sampling(
                        example_video,
                        sampling_method,
                        args.num_frames,
                        save_path,
                        f"{split.capitalize()} Split"
                    )
                    logger.info(f"Created sampling visualization for {split} split at {save_path}")
                    
                    # Check if this video needed dynamic FPS adjustment
                    cap = cv2.VideoCapture(example_video)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if total_frames < args.num_frames:
                        logger.info(f"Note: The example video for {split} split has {total_frames} frames, "
                                   f"which is less than the requested {args.num_frames} frames. "
                                   f"Dynamic FPS adjustment was applied.")
                except Exception as e:
                    logger.error(f"Error creating sampling visualization for {split}: {str(e)}")
        
        # Load checkpoint if provided, otherwise train model
        if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
            logger.info(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            
            # Load state dict depending on format
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            logger.info("Checkpoint loaded successfully")
        elif not args.skip_train:
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            logger.info(f"Using Adam optimizer with learning rate {args.learning_rate}")
            
            # Train model
            logger.info("Starting model training...")
            trainer = ModelTrainer(
                model, dataloaders, criterion, optimizer, device, args, exp_logger
            )
            model = trainer.train()
            logger.info("Model training completed")
        else:
            logger.info("Skipping training as requested")
        
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
        
        logger.info(f"Final AUROC: {auroc:.4f}")
        logger.info(f"Final F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("Training and evaluation pipeline completed successfully")

if __name__ == "__main__":
    main()
    
"""
python3 resnet50-3d-video/main.py \
--data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
--test_data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
--log_dir logs \
--model_dir resnet50-3d-video-models \
--train_sampling random_window \
--val_sampling uniform \
--test_sampling random_window \
--num_frames 32 \
--batch_size 2 \
--epochs 1 \
--learning_rate 0.01 \
--num_workers 2 \
--patience 7
"""