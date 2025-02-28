import torch
import torch.nn as nn
import numpy as np
import random
import argparse

from vivit_classifier.data_config.dataloader import create_dataloaders
from vivit_classifier.models.vivit_model import create_model
from vivit_classifier.trainers.trainer import ModelTrainer
from vivit_classifier.evaluators.evaluator import ModelEvaluator
from vivit_classifier.utils.logger import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser(description='ViViT Transformer Training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to training/validation dataset directory')
    parser.add_argument('--test_data_dir', type=str, default=None,
                      help='Path to test dataset directory. If not provided, will use data_dir')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save logs')
    parser.add_argument('--model_dir', type=str, default='vivit-models',
                      help='Directory to save models')
    
    # Dataset and sampling parameters
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
                      help='Number of frames to sample')
    parser.add_argument('--fps', type=int, default=8,
                      help='Frames per second for clip sampling')
    parser.add_argument('--stride', type=float, default=0.5,
                      help='Stride fraction for sliding window sampling')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default="google/vivit-b-16x2-kinetics400",
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
    exp_logger = ExperimentLogger(args.log_dir, prefix=f'vivit-classifier')
    logger = exp_logger.get_logger()
    logger.info("Starting ViViT Transformer training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
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
python3 vivit_transformer/main.py \
  --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
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
"""