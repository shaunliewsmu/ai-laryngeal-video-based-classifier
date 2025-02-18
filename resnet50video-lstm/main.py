import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import logging
import cv2
from tqdm import tqdm
import numpy as np
from datetime import datetime
from src.config.config import SEED, DEFAULT_CONFIG
from src.utils.logging_utils import set_seed, create_directories, setup_logging
from src.utils.visualization import plot_sampled_frames,plot_confusion_matrix
from src.data_config.dataset import VideoDataset
from src.models.model import VideoResNetLSTM
from src.trainer.trainer import train_model
from src.utils.metrics import calculate_metrics, print_class_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Path to model directory')
    parser.add_argument('--train_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method for training')
    parser.add_argument('--val_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method for validation')
    parser.add_argument('--test_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method for testing')
    args = parser.parse_args()

    # Setup
    set_seed(SEED)
    create_directories(args.log_dir)
    create_directories(args.model_dir)
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.log_dir, f'training_{timestamp}')
    create_directories(run_dir)
    
    # Create visualization directory inside run directory
    viz_dir = os.path.join(run_dir, 'visualizations')
    create_directories(viz_dir)

    
    # Setup logging
    log_file = os.path.join(run_dir, 'training.log')
    setup_logging(log_file)
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        'data_dir': args.data_dir,
        'log_dir': run_dir,
        'model_dir': args.model_dir,
        'viz_dir': viz_dir,
        'train_sampling': args.train_sampling,
        'val_sampling': args.val_sampling,
        'test_sampling': args.test_sampling
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loading
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]),
    ])
    
    datasets = {
        'train': VideoDataset(root_dir=args.data_dir, split='train', 
                             sequence_length=config['sequence_length'],
                             transform=transform, sampling_method=args.train_sampling),
        'val': VideoDataset(root_dir=args.data_dir, split='val',
                           sequence_length=config['sequence_length'],
                           transform=transform, sampling_method=args.val_sampling),
        'test': VideoDataset(root_dir=args.data_dir, split='test',
                            sequence_length=config['sequence_length'],
                            transform=transform, sampling_method=args.test_sampling)
    }
    
    # Visualize sampling for each split
    for split, dataset in datasets.items():
        if len(dataset.video_paths) > 0:
            example_video = dataset.video_paths[0]
            cap = cv2.VideoCapture(example_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            sampled_indices = dataset.sampler(total_frames, config['sequence_length'])
            plot_sampled_frames(
                example_video,
                sampled_indices,
                os.path.join(viz_dir, f'sampled_frames_{split}_{args.train_sampling}.png')
            )
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=config['batch_size'],
                           shuffle=True, num_workers=4, pin_memory=True,
                           drop_last=True, persistent_workers=True),
        'val': DataLoader(datasets['val'], batch_size=config['batch_size'],
                         shuffle=False, num_workers=4, pin_memory=True,
                         drop_last=True, persistent_workers=True),
        'test': DataLoader(datasets['test'], batch_size=config['batch_size'],
                          shuffle=False, num_workers=4, pin_memory=True)
    }
    
    # Model initialization and training
    model = VideoResNetLSTM(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    best_model_path, train_aurocs, val_aurocs = train_model(
        model, dataloaders['train'], dataloaders['val'], device, config
    )
    
    # Final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloaders['test'], desc='Testing'):
            videos = videos.to(device)
            outputs = model(videos)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.numpy())
    
    test_predictions = np.array(test_predictions).squeeze()
    test_labels = np.array(test_labels)
    
    if len(test_predictions) > 0 and len(test_labels) > 0:
        test_metrics = calculate_metrics(test_labels, test_predictions)
        
        plot_confusion_matrix(
            test_labels, 
            test_predictions,
            os.path.join(viz_dir, 'test_confusion_matrix.png'),
            'Test Set Confusion Matrix'
        )
        
        logging.info('Test Results:')
        for metric, value in test_metrics.items():
            logging.info(f'{metric}: {value:.4f}')
        print_class_metrics(test_labels, test_predictions, 'Test')
    else:
        logging.error("No test predictions or labels collected")

if __name__ == "__main__":
    main()

"""
python3 resnet50video-lstm/main.py \
    --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
    --log_dir logs \
    --model_dir resnet-models \
    --train_sampling random \
    --val_sampling random \
    --test_sampling random
"""