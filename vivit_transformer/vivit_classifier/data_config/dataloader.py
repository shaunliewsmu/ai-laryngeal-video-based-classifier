import torch
from torch.utils.data import DataLoader
from .dataset import VideoDataset

def video_collate_fn(batch):
    """Custom collate function for video batches."""
    pixel_values = []
    labels = []
    
    for sample in batch:
        if 'pixel_values' in sample and sample['pixel_values'] is not None:
            pixel_values.append(sample['pixel_values'])
        if 'labels' in sample and sample['labels'] is not None:
            labels.append(sample['labels'])
    
    # Only stack labels, but keep pixel_values as a list
    if labels:
        labels = torch.stack(labels)
    else:
        labels = torch.zeros(len(batch), dtype=torch.long)
    
    return {
        'pixel_values': pixel_values,  # Keep as list
        'labels': labels
    }
    
def create_dataloaders(args, sampling_methods, logger):
    """
    Create data loaders for train, validation, and test sets with specific sampling methods.
    
    Args:
        args: Command line arguments
        sampling_methods: Dictionary with keys 'train', 'val', 'test' and sampling method values
        logger: Logger instance
    
    Returns:
        dataloaders: Dictionary of dataloaders
        class_labels: List of class labels
    """
    
    logger.info(f"Creating datasets from {args.data_dir}")
    logger.info(f"Using sampling methods: {sampling_methods}")
    
    datasets = {}
    dataloaders = {}
    class_labels = None
    
    # Create train and validation dataloaders
    for split in ['train', 'val']:
        try:
            datasets[split] = VideoDataset(
                args.data_dir,
                mode=split,
                sampling_method=sampling_methods[split],
                num_frames=args.num_frames,
                fps=args.fps,
                stride=args.stride,
                logger=logger
            )
            
            if class_labels is None:
                class_labels = datasets[split].class_labels
                logger.info(f"Detected class labels: {class_labels}")
            
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
                f"batches (batch size: {args.batch_size}) using {sampling_methods[split]} sampling"
            )
            
        except Exception as e:
            logger.error(f"Error creating {split} dataset/dataloader: {str(e)}")
            raise

    # Create test dataloader with potentially different data source
    try:
        test_data_dir = args.test_data_dir if args.test_data_dir else args.data_dir
        datasets['test'] = VideoDataset(
            test_data_dir,
            mode='test',
            sampling_method=sampling_methods['test'],
            num_frames=args.num_frames,
            fps=args.fps,
            stride=args.stride,
            logger=logger
        )
        
        dataloaders['test'] = DataLoader(
            datasets['test'],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        logger.info(
            f"Created test dataloader from {test_data_dir} with "
            f"{len(dataloaders['test'])} batches (batch size: {args.batch_size}) "
            f"using {sampling_methods['test']} sampling"
        )
        
    except Exception as e:
        logger.error(f"Error creating test dataset/dataloader: {str(e)}")
        raise
    
    return dataloaders, class_labels