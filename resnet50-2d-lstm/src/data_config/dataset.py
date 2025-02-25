import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo.data.clip_sampling import ClipInfo
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    RandomShortSideScale,
    ShortSideScale,
    Normalize,
)
from pathlib import Path
import logging

class VideoDataset(Dataset):
    def __init__(self, root_dir, split='train', sampling_method='uniform', 
                 sequence_length=32, transform=None, logger=None):
        """
        Initialize the video dataset.
        
        Args:
            root_dir (str): Root directory of dataset
            split (str): 'train', 'val', or 'test'
            sampling_method (str): 'uniform', 'random', or 'sliding'
            sequence_length (int): Number of frames to sample
            transform: Custom transform function (if None, will use default transforms)
            logger (logging.Logger): Logger instance
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.sampling_method = sampling_method
        self.custom_transform = transform
        self.logger = logger or logging.getLogger(__name__)
        self.fps = 30  # Default FPS assumption
        
        self._setup_data_paths()
        self._setup_clip_sampler()
        self._setup_transforms()
        
    def _setup_data_paths(self):
        """Set up video paths and labels."""
        self.video_paths = []
        self.labels = []
        
        referral_path = self.root_dir / self.split / 'referral'
        non_referral_path = self.root_dir / self.split / 'non_referral'
        
        # Get non-referral videos
        if non_referral_path.exists():
            for video_path in non_referral_path.glob('*.mp4'):
                self.video_paths.append(str(video_path))
                self.labels.append(0)
        
        # Get referral videos
        if referral_path.exists():
            for video_path in referral_path.glob('*.mp4'):
                self.video_paths.append(str(video_path))
                self.labels.append(1)
                    
        self.logger.info(f"Found {len(self.video_paths)} videos for {self.split} split")
        self.logger.info(f"Class distribution: {sum(self.labels)} referral, "
                        f"{len(self.labels) - sum(self.labels)} non-referral")
                        
    def _setup_clip_sampler(self):
        """Set up clip sampler based on sampling method."""
        self.clip_duration = float(self.sequence_length) / self.fps
        
        if self.sampling_method == 'sliding':
            self.stride = self.clip_duration * 0.5  # 50% overlap
            self.clip_sampler = make_clip_sampler(
                'uniform',
                self.clip_duration,
                self.stride
            )
        else:
            self.clip_sampler = make_clip_sampler(
                self.sampling_method,
                self.clip_duration
            )
            
    def _setup_transforms(self):
        """Set up video transforms based on split."""
        if self.custom_transform:
            # If a custom transform is provided, apply it directly to the video tensor
            self.transform = ApplyTransformToKey(
                key="video",
                transform=self.custom_transform,
            )
        else:
            # Default transforms that work with pytorchvideo
            if self.split == 'train':
                self.transform = ApplyTransformToKey(
                    key="video",
                    transform=transforms.Compose([
                        UniformTemporalSubsample(self.sequence_length),
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
                        UniformTemporalSubsample(self.sequence_length),
                        ShortSideScale(256),
                        transforms.CenterCrop(224),
                        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
                    ]),
                )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """Get video clips and labels."""
        if idx >= len(self.video_paths):
            # Handle out of range indices
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.video_paths)} videos")
            
        video_path = self.video_paths[idx]  # Define this first to avoid UnboundLocalError
        try:
            label = self.labels[idx]
            
            video = EncodedVideo.from_path(video_path)
            duration = video.duration or 10.0
            
            # For training/validation/testing, we just need one clip per video
            clip_start_sec = 0
            clip_info = self.clip_sampler(clip_start_sec, duration, None)
            
            if clip_info is None:
                raise ValueError(f"Could not sample clip from video: {video_path}")
                
            video_data = video.get_clip(
                clip_info.clip_start_sec, 
                clip_info.clip_end_sec
            )
            
            if self.transform is not None:
                video_data = self.transform(video_data)
            
            # Extract the video tensor from the transformed data
            video_tensor = video_data["video"]
            
            return video_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {str(e)}")
            raise