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
from src.utils.visualization import plot_clip_visualization,plot_confusion_matrix
from pytorchvideo.data.encoded_video import EncodedVideo
from src.data_config.dataset import VideoDataset
from src.models.model import VideoResNet50LSTM
from src.trainer.trainer import train_model
from src.utils.metrics import calculate_metrics, print_class_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory for training and validation')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='Path to separate test dataset directory (if not specified, will use data_dir)')
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
    parser.add_argument('--loss_weight', type=float, default=0.3,
                        help='Weight for loss in model selection (0-1). Higher values prioritize minimizing loss.')
    args = parser.parse_args()

    # Setup
    set_seed(SEED)
    create_directories(args.log_dir)
    create_directories(args.model_dir)
    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.log_dir, f'resnet50_lstm_training_{timestamp}')
    create_directories(run_dir)
    
    # Create visualization directory inside run directory
    viz_dir = os.path.join(run_dir, 'visualizations')
    create_directories(viz_dir)

    
    # Setup logging
    log_file = os.path.join(run_dir, 'training.log')
    setup_logging(log_file)
    # Update config with command line arguments
    # Determine test directory (use data_dir if test_dir is not specified)
    test_dir = args.test_dir if args.test_dir else args.data_dir
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        'data_dir': args.data_dir,
        'test_dir': test_dir,
        'log_dir': run_dir,
        'model_dir': args.model_dir,
        'viz_dir': viz_dir,
        'train_sampling': args.train_sampling,
        'val_sampling': args.val_sampling,
        'test_sampling': args.test_sampling,
        'loss_weight': args.loss_weight
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loading   
    datasets = {
    'train': VideoDataset(
        root_dir=args.data_dir, 
        split='train', 
        sampling_method=args.train_sampling,
        sequence_length=config['sequence_length'],
        logger=logging
    ),
    'val': VideoDataset(
        root_dir=args.data_dir, 
        split='val',
        sampling_method=args.val_sampling,
        sequence_length=config['sequence_length'],
        logger=logging
    ),
    'test': VideoDataset(
        root_dir=config['test_dir'], 
        split='test',
        sampling_method=args.test_sampling,
        sequence_length=config['sequence_length'],
        logger=logging
    )
}
    
    logging.info(f"Using training/validation data from: {args.data_dir}")
    logging.info(f"Using test data from: {config['test_dir']}")
    
    # Visualize sampling for each split
    for split, dataset in datasets.items():
        if len(dataset.video_paths) > 0:
            example_video = dataset.video_paths[0]
            
            # Get a sample clip using PyTorchVideo
            video = EncodedVideo.from_path(example_video)
            duration = video.duration or 10.0
            
            # Get clip info
            clip_sampler = dataset.clip_sampler
            if clip_sampler:
                # Try multiple times for random sampler to demonstrate randomness
                if dataset.sampling_method == 'random':
                    clip_infos = []
                    for _ in range(3):  # Generate 3 different random samples
                        clip_info = clip_sampler(0, duration, None)
                        if clip_info:
                            clip_infos.append(clip_info)
                            
                    # Visualize each clip
                    for i, clip_info in enumerate(clip_infos):
                        plot_clip_visualization(
                            example_video,
                            clip_info,
                            os.path.join(viz_dir, f'sampled_frames_{split}_{dataset.sampling_method}_{i+1}.png'),
                            f'{split} frames - {dataset.sampling_method} sampling (sample {i+1})',
                            fps=dataset.fps
                        )
                else:
                    # For non-random methods, just visualize one clip
                    clip_info = clip_sampler(0, duration, None)
                    if clip_info:
                        plot_clip_visualization(
                            example_video,
                            clip_info,
                            os.path.join(viz_dir, f'sampled_frames_{split}_{dataset.sampling_method}.png'),
                            f'{split} frames - {dataset.sampling_method} sampling',
                            fps=dataset.fps
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
    
    # Model initialization and training - Updated to use VideoResNet50LSTM
    model = VideoResNet50LSTM(
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
python3 resnet50-2d-lstm/main.py \
    --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
    --test_dir artifacts/laryngeal_dataset_iqm_filtered:v0/dataset \
    --log_dir logs \
    --model_dir resnet50-2d-lstm-models \
    --train_sampling random \
    --val_sampling random \
    --test_sampling random \
    --loss_weight 0.3
"""