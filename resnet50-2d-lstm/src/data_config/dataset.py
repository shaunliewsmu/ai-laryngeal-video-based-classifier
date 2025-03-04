# Updated VideoDataset class in dataset.py
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    RandomShortSideScale,
    ShortSideScale,
    Normalize,
)
import random
from pathlib import Path
import logging

class VideoDataset(Dataset):
    def __init__(self, root_dir, split='train', sampling_method='uniform', 
                 sequence_length=32, transform=None, logger=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.sampling_method = sampling_method
        self.custom_transform = transform
        self.logger = logger or logging.getLogger(__name__)
        self.fps = 30  # Default FPS
        
        self._setup_data_paths()
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
            
    def _setup_transforms(self):
        """Set up video transforms based on split."""
        if self.custom_transform:
            self.transform = self.custom_transform
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
    
    def get_sampling_indices(self, total_frames):
        """Get frame indices based on sampling method"""
        # Set random seed for reproducibility
        random.seed(42)
        
        # Ensure we don't request more frames than available
        num_frames = min(self.sequence_length, total_frames)
        
        if self.sampling_method == 'random':
            # Random sampling without replacement
            indices = sorted(random.sample(range(total_frames), num_frames))
        elif self.sampling_method == 'random_window':
            # Random window sampling
            window_size = total_frames / num_frames
            indices = []
            for i in range(num_frames):
                start = int(i * window_size)
                end = min(int((i + 1) * window_size), total_frames)
                end = max(end, start + 1)  # Ensure window has at least 1 frame
                frame_idx = random.randint(start, end - 1)
                indices.append(frame_idx)
        else:  # Default to uniform sampling
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                step = (total_frames - 1) / (num_frames - 1)
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
        
        return indices

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """Get video clips and labels."""
        if idx >= len(self.video_paths):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.video_paths)} videos")
            
        video_path = self.video_paths[idx]
        try:
            label = self.labels[idx]
            
            # Load video using PyTorchVideo
            video = EncodedVideo.from_path(video_path)
            
            # Get video duration and calculate total frames
            duration = video.duration or 10.0
            total_frames = int(duration * self.fps)
            
            # Get sampling indices based on our custom method
            frame_indices = self.get_sampling_indices(total_frames)
            
            # Convert frame indices to timestamps (seconds)
            timestamps = [idx / self.fps for idx in frame_indices]
            
            # Sort timestamps for proper video loading
            timestamps.sort()
            
            # Calculate start and end times with a small buffer
            start_sec = max(0, timestamps[0] - 0.01)
            end_sec = min(duration, timestamps[-1] + 0.1)
            
            # Get video clip using PyTorchVideo's efficient loading
            video_data = video.get_clip(start_sec, end_sec)
            
            if self.transform is not None:
                video_data = self.transform(video_data)
            
            # Extract the video tensor
            video_tensor = video_data["video"]
            
            # If we have more frames than needed (due to the buffering), subsample
            if video_tensor.shape[1] > self.sequence_length:
                # Use uniform temporal subsample to get exact number of frames
                video_tensor = UniformTemporalSubsample(self.sequence_length)(video_tensor)
            
            return video_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {str(e)}")
            raise