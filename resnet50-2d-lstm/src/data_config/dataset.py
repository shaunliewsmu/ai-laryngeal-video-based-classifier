import random
import numpy as np
import cv2
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
    
    def get_sampling_indices(self, video_path, total_frames):
        """
        Get frame indices based on sampling method, with dynamic FPS for short videos.
        
        Args:
            video_path (str): Path to the video file
            total_frames (int): Total number of frames in the video
            
        Returns:
            list: Frame indices to sample
        """
        # Set random seed for reproducibility
        random.seed(42)
        
        # For videos with enough frames, use standard sampling
        if total_frames >= self.sequence_length:
            if self.sampling_method == 'random':
                # Random sampling without replacement
                indices = sorted(random.sample(range(total_frames), self.sequence_length))
            elif self.sampling_method == 'random_window':
                # Random window sampling
                window_size = total_frames / self.sequence_length
                indices = []
                for i in range(self.sequence_length):
                    start = int(i * window_size)
                    end = min(int((i + 1) * window_size), total_frames)
                    end = max(end, start + 1)  # Ensure window has at least 1 frame
                    frame_idx = random.randint(start, end - 1)
                    indices.append(frame_idx)
            else:  # Default to uniform sampling
                if self.sequence_length == 1:
                    indices = [total_frames // 2]  # Middle frame
                else:
                    step = (total_frames - 1) / (self.sequence_length - 1)
                    indices = [min(int(i * step), total_frames - 1) for i in range(self.sequence_length)]
        
        # For videos with fewer frames than requested, use dynamic adjustment
        else:
            # Get original video FPS for proper time scaling
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / original_fps  # in seconds
            cap.release()
            
            # Calculate optimal sampling rate to get the exact number of frames requested
            dynamic_fps = self.sequence_length / duration
            
            if self.logger:
                self.logger.info(f"Dynamic adjustment: Video {video_path} has {total_frames} frames, "
                                f"adjusted to get {self.sequence_length} frames.")
            
            # Use the sampling method with the adjusted parameters
            if self.sampling_method == 'random':
                # With dynamic adjustment, we'll need to allow duplicates since total_frames < sequence_length
                # Use random.choices which allows replacement
                indices = sorted(random.choices(range(total_frames), k=self.sequence_length))
            elif self.sampling_method == 'random_window':
                # For random window with fewer frames, create virtual windows smaller than 1 frame
                indices = []
                window_size = total_frames / self.sequence_length  # Will be < 1
                
                for i in range(self.sequence_length):
                    # Calculate virtual window boundaries
                    virtual_start = i * window_size
                    virtual_end = (i + 1) * window_size
                    
                    # Convert to actual frame indices with potential duplicates
                    actual_index = min(int(np.floor(virtual_start + (virtual_end - virtual_start) * random.random())), 
                                    total_frames - 1)
                    indices.append(actual_index)
            else:  # Uniform sampling
                if self.sequence_length == 1:
                    indices = [total_frames // 2]  # Middle frame
                else:
                    # Create evenly spaced indices that might include duplicates
                    step = total_frames / self.sequence_length
                    indices = [min(int(i * step), total_frames - 1) for i in range(self.sequence_length)]
        
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
            
            # Get total frames using OpenCV for more reliable frame count
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Get sampling indices based on our custom method
            frame_indices = self.get_sampling_indices(video_path, total_frames)
            
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