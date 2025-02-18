import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from src.data_config.sampling import VideoSampling

class VideoDataset(Dataset):
    def __init__(self, root_dir, split='train', sequence_length=32, transform=None, sampling_method='uniform'):
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.sampling_method = sampling_method
        self.sampler = VideoSampling.get_sampler(sampling_method)
        
        # Get all video paths and labels
        self.video_paths = []
        self.labels = []
        
        # Get non-referral videos
        non_referral_path = os.path.join(root_dir, split, 'non_referral', '*.mp4')
        non_referral_videos = glob.glob(non_referral_path)
        self.video_paths.extend(non_referral_videos)
        self.labels.extend([0] * len(non_referral_videos))
        
        # Get referral videos
        referral_path = os.path.join(root_dir, split, 'referral', '*.mp4')
        referral_videos = glob.glob(referral_path)
        self.video_paths.extend(referral_videos)
        self.labels.extend([1] * len(referral_videos))
        
        logging.info(f"Found {len(self.video_paths)} videos for {split} using {sampling_method} sampling")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.sampler(total_frames, self.sequence_length)
        
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        frames = np.stack(frames, axis=0) # [T, H, W, C]
        frames = frames.astype(np.float32) / 255.0
        
        if self.transform:
            frames = self.transform(frames)
        
        # Convert to tensor and rearrange dimensions to [C, T, H, W]
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.load_video(video_path)
        return frames, torch.tensor(label, dtype=torch.float32)