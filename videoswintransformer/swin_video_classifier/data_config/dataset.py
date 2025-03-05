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
from pathlib import Path
import logging
import numpy as np
import random
import cv2

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', sampling_method='uniform', 
                 num_frames=32, fps=30, stride=0.5, logger=None):
        """
        Initialize the video dataset.
        
        Args:
            root_dir (str): Root directory of dataset
            mode (str): 'train', 'val', or 'test'
            sampling_method (str): 'uniform', 'random', or 'random_window'
            num_frames (int): Number of frames to sample
            fps (int): Frames per second of videos 
            stride (float): Stride fraction for sliding window
            logger (logging.Logger): Logger instance
        """
        self.root_dir = Path(root_dir) / mode
        self.mode = mode
        self.num_frames = num_frames
        self.sampling_method = sampling_method
        self.logger = logger or logging.getLogger(__name__)
        self.fps = fps
        self.stride = stride
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        self._setup_data_paths()
        self._setup_transforms()
        
    def _setup_data_paths(self):
        """Set up video paths and labels."""
        self.video_paths = []
        self.labels = []
        
        for class_path in self.root_dir.iterdir():
            if class_path.is_dir():
                label = 1 if class_path.name == 'referral' else 0
                for video_path in class_path.glob('*.mp4'):
                    self.video_paths.append(str(video_path))
                    self.labels.append(label)
                    
        self.logger.info(f"Found {len(self.video_paths)} videos for {self.mode} split")
        self.logger.info(f"Class distribution: {sum(self.labels)} referral, "
                        f"{len(self.labels) - sum(self.labels)} non-referral")
    
    def get_sampling_indices(self, video_path, total_frames):
        """
        Get frame indices based on sampling method, with dynamic FPS for short videos.
        
        Args:
            video_path (str): Path to the video file
            total_frames (int): Total number of frames in the video
            
        Returns:
            list: Frame indices to sample
            float: Dynamic FPS used if applicable, None otherwise
        """
        # Initialize dynamic FPS to None
        dynamic_fps = None
        
        # For videos with enough frames, use standard sampling
        if total_frames >= self.num_frames:
            if self.sampling_method == 'random':
                # Random sampling without replacement
                indices = sorted(random.sample(range(total_frames), self.num_frames))
            elif self.sampling_method == 'random_window':
                # Random window sampling
                window_size = total_frames / self.num_frames
                indices = []
                for i in range(self.num_frames):
                    start = int(i * window_size)
                    end = min(int((i + 1) * window_size), total_frames)
                    end = max(end, start + 1)  # Ensure window has at least 1 frame
                    frame_idx = random.randint(start, end - 1)
                    indices.append(frame_idx)
            else:  # Default to uniform sampling
                if self.num_frames == 1:
                    indices = [total_frames // 2]  # Middle frame
                else:
                    step = (total_frames - 1) / (self.num_frames - 1)
                    indices = [min(int(i * step), total_frames - 1) for i in range(self.num_frames)]
        
        # For videos with fewer frames than requested, use dynamic FPS adjustment
        else:
            # Get original video FPS for proper time scaling
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / original_fps  # in seconds
            cap.release()
            
            # Calculate optimal sampling rate to get the exact number of frames requested
            dynamic_fps = self.num_frames / duration
            
            self.logger.info(f"Dynamic FPS adjustment: Video {video_path} has {total_frames} frames, "
                          f"adjusted from {self.fps} to {dynamic_fps:.2f} fps to get {self.num_frames} frames.")
            
            # Use the sampling method with the adjusted parameters
            if self.sampling_method == 'random':
                # With dynamic FPS, we'll need to allow duplicates since total_frames < num_frames
                # Use random.choices which allows replacement
                indices = sorted(random.choices(range(total_frames), k=self.num_frames))
            elif self.sampling_method == 'random_window':
                # For random window with fewer frames, create virtual windows smaller than 1 frame
                indices = []
                window_size = total_frames / self.num_frames  # Will be < 1
                
                for i in range(self.num_frames):
                    # Calculate virtual window boundaries
                    virtual_start = i * window_size
                    virtual_end = (i + 1) * window_size
                    
                    # Convert to actual frame indices with potential duplicates
                    actual_index = min(int(np.floor(virtual_start + (virtual_end - virtual_start) * random.random())), 
                                     total_frames - 1)
                    indices.append(actual_index)
            else:  # Uniform sampling
                if self.num_frames == 1:
                    indices = [total_frames // 2]  # Middle frame
                else:
                    # Create evenly spaced indices that might include duplicates
                    step = total_frames / self.num_frames
                    indices = [min(int(i * step), total_frames - 1) for i in range(self.num_frames)]
        
        return indices, dynamic_fps
            
    def _setup_transforms(self):
        """Set up video transforms based on mode."""
        if self.mode == 'train':
            self.transform = ApplyTransformToKey(
                key="video",
                transform=transforms.Compose([
                    UniformTemporalSubsample(self.num_frames),
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
                    UniformTemporalSubsample(self.num_frames),
                    ShortSideScale(256),
                    transforms.CenterCrop(224),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
                ]),
            )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """Get video clips and labels."""
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]
            
            # Open video using OpenCV to get frame count
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Sample frames based on sampling method
            frame_indices, dynamic_fps = self.get_sampling_indices(video_path, total_frames)
            
            # Load video using PyTorchVideo
            video = EncodedVideo.from_path(video_path)
            duration = video.duration or 10.0
            
            clips = []
            labels = []
            
            # Calculate start and end time from frame indices
            # We need to convert from frame index to timestamp
            start_sec = max(0, frame_indices[0] / self.fps)
            end_sec = min(duration, (frame_indices[-1] + 1) / self.fps)
            
            # Get the clip
            try:
                video_data = video.get_clip(start_sec, end_sec)
                if self.transform is not None:
                    video_data = self.transform(video_data)
                
                clips.append(video_data["video"])
                labels.append(label)
            except Exception as e:
                self.logger.warning(f"Error extracting clip: {str(e)}")
                raise
            
            if not clips:
                raise ValueError(f"No valid clips extracted from video {video_path}")
            
            return torch.stack(clips), torch.tensor(labels, dtype=torch.long)
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {str(e)}")
            raise