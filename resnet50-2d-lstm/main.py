import argparse
import torch
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path

from src.config.config import SEED, DEFAULT_CONFIG
from src.utils.logging_utils import set_seed, create_directories, setup_logging
from src.utils.visualization import EnhancedVisualizer
from src.data_config.dataset import VideoDataset
from src.models.model import VideoResNet50LSTM
from src.trainer.trainer import EnhancedTrainer
from src.evaluators.evaluator import ModelEvaluator
from src.utils.logger import ExperimentLogger

def main():
    parser = argparse.ArgumentParser(description='ResNet50-LSTM Video Classification Training with Enhanced Visualization')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory for training and validation')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to separate test dataset directory (if not specified, will use data_dir)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Path to model directory')
    parser.add_argument('--train_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method for training')
    parser.add_argument('--val_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method for validation')
    parser.add_argument('--test_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method for testing')
    parser.add_argument('--loss_weight', type=float, default=0.3,
                        help='Weight for loss in model selection (0-1). Higher values prioritize minimizing loss.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM number of layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Number of frames to sample from each video')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and only run evaluation')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a trained model checkpoint for evaluation')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')
    args = parser.parse_args()

    torch.cuda.empty_cache()  # Clear GPU cache
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Setup
    set_seed(SEED)
    create_directories(args.log_dir)
    create_directories(args.model_dir)
    
    # Setup experiment logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_logger = ExperimentLogger(
        os.path.join(args.log_dir, f'resnet50_lstm_enhanced_{timestamp}')
    )
    logger = exp_logger.get_logger()
    logger.info("Starting enhanced ResNet50-LSTM training")
    
    # Determine test directory (use data_dir if test_dir is not specified)
    test_dir = args.test_dir if args.test_dir else args.data_dir
    
    # Create configuration dictionary
    config = {
        'data_dir': args.data_dir,
        'test_dir': test_dir,
        'log_dir': exp_logger.get_experiment_dir(),
        'model_dir': args.model_dir,
        'train_sampling': args.train_sampling,
        'val_sampling': args.val_sampling,
        'test_sampling': args.test_sampling,
        'loss_weight': args.loss_weight,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'sequence_length': args.sequence_length
    }
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create visualization directory
    viz_dir = Path(exp_logger.get_experiment_dir()) / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = EnhancedVisualizer(viz_dir)
    
    # Data loading   
    datasets = {
        'train': VideoDataset(
            root_dir=args.data_dir, 
            split='train', 
            sampling_method=args.train_sampling,
            sequence_length=config['sequence_length'],
            logger=logger
        ),
        'val': VideoDataset(
            root_dir=args.data_dir, 
            split='val',
            sampling_method=args.val_sampling,
            sequence_length=config['sequence_length'],
            logger=logger
        ),
        'test': VideoDataset(
            root_dir=config['test_dir'], 
            split='test',
            sampling_method=args.test_sampling,
            sequence_length=config['sequence_length'],
            logger=logger
        )
    }
    
    logger.info(f"Using training/validation data from: {args.data_dir}")
    logger.info(f"Using test data from: {config['test_dir']}")
    
    # Visualize sampling for each split
    for split, dataset in datasets.items():
        if len(dataset.video_paths) > 0:
            example_video = dataset.video_paths[0]
            
            # Visualize frame sampling
            sampling_method = getattr(args, f"{split}_sampling")
            try:
                visualizer.visualize_sampling(
                    example_video,
                    sampling_method,
                    config['sequence_length'],
                    viz_dir / f'sampled_frames_{split}_{sampling_method}.png',
                    f"{split.capitalize()} Split"
                )
                logger.info(f"Created sampling visualization for {split} split")
            except Exception as e:
                logger.error(f"Error creating sampling visualization for {split}: {str(e)}")
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=config['batch_size'],
                           shuffle=True, num_workers=args.num_workers, pin_memory=True,
                           drop_last=True),
        'val': DataLoader(datasets['val'], batch_size=config['batch_size'],
                         shuffle=False, num_workers=args.num_workers, pin_memory=True,
                         drop_last=True),
        'test': DataLoader(datasets['test'], batch_size=config['batch_size'],
                          shuffle=False, num_workers=args.num_workers, pin_memory=True)
    }
    
    # Model initialization
    model = VideoResNet50LSTM(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Training or loading checkpoint
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        best_model_path = args.checkpoint_path
    elif not args.skip_train:
        # Train the model with enhanced trainer
        trainer = EnhancedTrainer(
            model, 
            dataloaders['train'], 
            dataloaders['val'],
            device,
            config,
            exp_logger
        )
        
        model, best_model_path = trainer.train()
        logger.info(f"Training completed. Best model saved to {best_model_path}")
    else:
        logger.info("Skipping training as requested")
        best_model_path = None
    
    # Evaluation
    if best_model_path:
        logger.info(f"Evaluating model from {best_model_path}")
        # Make sure we've loaded the best model
        if not args.skip_train:  # Already loaded during training
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # Initialize and run evaluator
        evaluator = ModelEvaluator(
            model,
            dataloaders['test'],
            device,
            exp_logger,
            class_names=['non_referral', 'referral']
        )
        
        auroc, f1, confusion_matrix = evaluator.evaluate()
        
        logger.info(f"Test Results:")
        logger.info(f"AUROC: {auroc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix}")
    else:
        logger.error("No model available for evaluation")

if __name__ == "__main__":
    main()

"""
Example usage:
python3 resnet50-2d-lstm/main.py \
    --data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
    --test_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
    --log_dir logs \
    --model_dir resnet50-2d-lstm-models \
    --train_sampling uniform \
    --val_sampling uniform \
    --test_sampling uniform \
    --batch_size 4 \
    --epochs 30 \
    --learning_rate 0.001 \
    --patience 10 \
    --loss_weight 0.3
"""