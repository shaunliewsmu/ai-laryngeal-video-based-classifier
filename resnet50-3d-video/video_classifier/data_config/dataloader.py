import torch
from torch.utils.data import DataLoader
from .dataset import VideoDataset

def video_collate_fn(batch):
    """Custom collate function for video batches."""
    min_clips = min(sample_clips.size(0) for sample_clips, _ in batch)
    
    clips = []
    labels = []
    
    for sample_clips, sample_labels in batch:
        clips.append(sample_clips[:min_clips])
        labels.append(sample_labels[:min_clips])
    
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
    
    # Create train and validation dataloaders
    for split in ['train', 'val']:
        try:
            datasets[split] = VideoDataset(
                args.data_dir,
                mode=split,
                sampling_method=sampling_methods[split],
                num_frames=args.num_frames,
                logger=logger
            )
            
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=args.batch_size,
                shuffle=(split == 'train'),
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            logger.info(
                f"Created {split} dataloader with {len(dataloaders[split])} "
                f"batches (batch size: {args.batch_size})"
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
            f"{len(dataloaders['test'])} batches (batch size: {args.batch_size})"
        )
        
    except Exception as e:
        logger.error(f"Error creating test dataset/dataloader: {str(e)}")
        raise
    
    return datasets, dataloaders