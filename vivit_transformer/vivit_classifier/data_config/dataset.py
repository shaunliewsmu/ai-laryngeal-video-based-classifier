import torch
from torch.utils.data import Dataset
import numpy as np
import av
from pathlib import Path
import logging
from pytorchvideo.data.clip_sampling import ClipInfo
from pytorchvideo.data import make_clip_sampler

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', sampling_method='uniform', 
                 num_frames=32, fps=8, stride=0.5, logger=None):
        """
        Initialize the dataset for ViViT model training with pytorchvideo samplers.
        
        Args:
            root_dir (str): Root directory containing dataset folders
            mode (str): One of 'train', 'val', 'test'
            sampling_method (str): 'uniform', 'random', or 'sliding' 
            num_frames (int): Number of frames to sample per video
            fps (int): Frames per second to sample
            stride (float): Stride fraction for sliding window sampling
            logger (logging.Logger, optional): Logger instance
        """
        self.root_dir = Path(root_dir)
        if not (self.root_dir / 'dataset').exists():
            if (self.root_dir / mode).exists():
                self.data_dir = self.root_dir / mode
            else:
                self.root_dir = self.root_dir / 'dataset'
                self.data_dir = self.root_dir / mode
        else:
            self.data_dir = self.root_dir / 'dataset' / mode
        
        self.mode = mode
        self.num_frames = num_frames
        self.fps = fps
        self.sampling_method = sampling_method
        self.stride = stride
        self.logger = logger or logging.getLogger(__name__)
        
        self.video_paths = []
        self.labels = []
        self.class_labels = []
        
        self._load_dataset()
        self._setup_clip_sampler()
        
    def _load_dataset(self):
        """Load video paths and labels."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.class_labels = sorted([d.name for d in class_dirs])
        
        self.logger.info(f"Found {len(class_dirs)} classes: {self.class_labels}")
        
        label_map = {label: idx for idx, label in enumerate(self.class_labels)}
        
        # Collect video paths and labels
        for class_dir in class_dirs:
            class_label = class_dir.name
            label_idx = label_map[class_label]
            
            video_files = list(class_dir.glob("*.mp4"))
            if not video_files:
                self.logger.warning(f"No .mp4 files found in {class_dir}")
            
            self.logger.info(f"Found {len(video_files)} videos in class '{class_label}'")
            
            for video_path in video_files:
                self.video_paths.append(video_path)
                self.labels.append(label_idx)
        
        self.logger.info(f"Total videos for {self.mode}: {len(self.video_paths)} using {self.sampling_method} sampling")
    
    def _setup_clip_sampler(self):
        """Set up clip sampler based on the sampling method."""
        # Calculate clip duration in seconds based on the number of frames and fps
        self.clip_duration = float(self.num_frames) / self.fps
        
        if self.sampling_method == 'sliding':
            # Sliding window with a defined stride (overlap)
            self.stride_duration = self.clip_duration * self.stride
            self.clip_sampler = make_clip_sampler(
                'uniform', 
                self.clip_duration,
                self.stride_duration
            )
        elif self.sampling_method == 'random':
            # Random clip from video
            self.clip_sampler = make_clip_sampler(
                'random', 
                self.clip_duration
            )
        else:  # default to uniform
            # Uniform clip (evenly spaced)
            self.clip_sampler = make_clip_sampler(
                'uniform',
                self.clip_duration
            )
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get video frames and label using the selected sampling method."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Open video
            container = av.open(str(video_path))
            
            # Get video duration in seconds
            video_duration = float(container.streams.video[0].frames) / container.streams.video[0].average_rate
            
            if not video_duration or video_duration < self.clip_duration:
                self.logger.warning(
                    f"Video {video_path} is shorter than required duration. "
                    f"Video duration: {video_duration}s, Required: {self.clip_duration}s"
                )
                # Use the entire video
                clip_info = ClipInfo(0, min(video_duration, self.clip_duration), 0, True)
            else:
                # Get clip information using pytorchvideo sampler
                clip_info = self.clip_sampler(0, video_duration, self)
            
            if clip_info is None:
                raise ValueError(f"Could not sample clip from video {video_path}")
            
            # Extract the clip
            clip_frames = self._extract_clip(container, clip_info)
            
            # Prepare sample - ensure clip_frames is a numpy array
            sample = {
                'pixel_values': np.array(clip_frames),  # Ensure it's a numpy array
                'labels': torch.tensor(label)
            }
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Error loading video {video_path}: {str(e)}")
            
            # Return a placeholder in case of error
            placeholder = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            return {
                'pixel_values': placeholder,
                'labels': torch.tensor(label)
            }
    
    def _extract_clip(self, container, clip_info):
        """Extract video clip based on clip_info."""
        start_sec = clip_info.clip_start_sec
        end_sec = clip_info.clip_end_sec
        
        # Calculate the number of frames to extract
        frames_to_extract = self.num_frames
        duration = end_sec - start_sec
        
        # Calculate start and end pts
        start_pts = int(start_sec * container.streams.video[0].time_base.denominator)
        
        # Seek to start position
        container.seek(start_pts, stream=container.streams.video[0])
        
        # Extract frames at regular intervals
        frames = []
        video_fps = container.streams.video[0].average_rate
        frame_interval = duration * video_fps / frames_to_extract
        frame_count = 0
        next_frame_to_extract = 0
        
        for frame in container.decode(video=0):
            if frame_count >= next_frame_to_extract:
                # Extract the frame as a numpy array in the correct format
                img = frame.to_ndarray(format="rgb24")
                frames.append(img)
                next_frame_to_extract += frame_interval
                
                # Break if we have enough frames
                if len(frames) >= frames_to_extract:
                    break
            
            frame_count += 1
        
        # If we couldn't extract enough frames, duplicate the last frame
        while len(frames) < frames_to_extract:
            if len(frames) == 0:
                # Create a blank frame if we have no frames
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                frames.append(frames[-1].copy())
        
        # Stack frames into a single array of shape (num_frames, height, width, 3)
        return np.stack(frames)