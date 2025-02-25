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
    def __init__(self, root_dir, mode='train', sampling_method='uniform', 
                 num_frames=32, fps=30, stride=0.5, logger=None):
        """
        Initialize the video dataset.
        
        Args:
            root_dir (str): Root directory of dataset
            mode (str): 'train', 'val', or 'test'
            sampling_method (str): 'uniform', 'random', or 'sliding'
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
        
        self._setup_data_paths()
        self._setup_clip_sampler(stride)
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
                        
    def _setup_clip_sampler(self, stride):
        """Set up clip sampler based on sampling method."""
        self.clip_duration = float(self.num_frames) / self.fps
        
        if self.sampling_method == 'sliding':
            self.stride = self.clip_duration * stride
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
            
            video = EncodedVideo.from_path(video_path)
            duration = video.duration or 10.0
            
            clips = []
            labels = []
            clip_start_sec = 0
            
            while True:
                clip_info = self.clip_sampler(clip_start_sec, duration, None)
                if clip_info is None:
                    break
                    
                try:
                    video_data = video.get_clip(
                        clip_info.clip_start_sec, 
                        clip_info.clip_end_sec
                    )
                    
                    if self.transform is not None:
                        video_data = self.transform(video_data)
                    
                    clips.append(video_data["video"])
                    labels.append(label)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting clip at {clip_info.clip_start_sec}: {str(e)}")
                    continue
                
                clip_start_sec = clip_info.clip_end_sec
                
                if clip_info.is_last_clip:
                    break
            
            if not clips:
                raise ValueError(f"No valid clips extracted from video {video_path}")
            
            return torch.stack(clips), torch.tensor(labels, dtype=torch.long)
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {str(e)}")
            raise