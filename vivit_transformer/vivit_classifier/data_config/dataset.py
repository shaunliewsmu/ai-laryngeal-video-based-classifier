import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
from pathlib import Path
import logging
from pytorchvideo.data.encoded_video import EncodedVideo

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', sampling_method='uniform', 
                 num_frames=32, logger=None):
        """
        Initialize the dataset for ViViT model training with custom frame sampling.
        
        Args:
            root_dir (str): Root directory containing dataset folders
            mode (str): One of 'train', 'val', 'test'
            sampling_method (str): 'uniform', 'random', or 'random_window' 
            num_frames (int): Number of frames to sample per video
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
        self.sampling_method = sampling_method
        self.logger = logger or logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        self.video_paths = []
        self.labels = []
        self.class_labels = []
        
        self._load_dataset()
        
    def _verify_video_integrity(self, video_path):
        """Verify that a video can be opened and read correctly."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.warning(f"Could not open video: {video_path}")
                return False
            
            # Check if we can read at least one frame
            ret, _ = cap.read()
            if not ret:
                self.logger.warning(f"Could not read any frames from {video_path}")
                return False
            
            # Check total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                self.logger.warning(f"Video {video_path} has no frames")
                return False
            
            cap.release()
            return True
        except Exception as e:
            self.logger.warning(f"Error verifying video {video_path}: {str(e)}")
            return False
        
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
            
            # Filter valid videos
            valid_videos = []
            for video_path in video_files:
                if self._verify_video_integrity(video_path):
                    valid_videos.append(video_path)
            
            invalid_count = len(video_files) - len(valid_videos)
            if invalid_count > 0:
                self.logger.warning(f"Skipped {invalid_count} invalid videos in class '{class_label}'")
            
            self.logger.info(f"Found {len(valid_videos)} valid videos in class '{class_label}'")
            
            for video_path in valid_videos:
                self.video_paths.append(video_path)
                self.labels.append(label_idx)
        
        self.logger.info(f"Total videos for {self.mode}: {len(self.video_paths)} using {self.sampling_method} sampling")
    
    def get_video_properties(self, video_path):
        """Get video properties using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps
        
        cap.release()
        return total_frames, fps, duration_sec, width, height
    
    def get_sampling_indices(self, video_path, total_frames):
        """
        Get frame indices based on sampling method, handling cases with fewer frames than requested.
        
        Args:
            video_path (str): Path to the video file
            total_frames (int): Total number of frames in the video
            
        Returns:
            list: Frame indices to sample
        """
        # For videos with enough frames, use standard sampling
        if total_frames >= self.num_frames:
            if self.sampling_method == 'random':
                # Random sampling without replacement
                indices = sorted(random.sample(range(total_frames), self.num_frames))
            elif self.sampling_method == 'random_window':
                # Random window sampling (similar to random_window in ResNet implementation)
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
        
        # For videos with fewer frames than requested, handle accordingly
        else:
            self.logger.info(f"Video has {total_frames} frames, which is less than the requested {self.num_frames} frames.")
            
            # Use the sampling method with appropriate handling for short videos
            if self.sampling_method == 'random':
                # With fewer frames, we'll need to allow duplicates
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
        
        return indices
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get video frames and label using pytorchvideo for more robust loading."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Try using pytorchvideo's EncodedVideo
            video = EncodedVideo.from_path(str(video_path))
            
            # Get video duration and properties
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = float(total_frames) / original_fps if original_fps > 0 else 0
            cap.release()
            
            if total_frames == 0 or duration_sec == 0:
                raise ValueError(f"Invalid video: {video_path} - no frames or zero duration")
                
            # Sample frames based on sampling method
            frame_indices = self.get_sampling_indices(video_path, total_frames)
            
            # Calculate clip start and end seconds
            start_idx = max(0, min(frame_indices))
            end_idx = min(total_frames - 1, max(frame_indices))
            start_sec = float(start_idx) / original_fps
            end_sec = float(end_idx + 1) / original_fps  # +1 to include the end frame
            
            # Get a safety margin to ensure we can capture all frames
            clip_duration = end_sec - start_sec
            start_sec = max(0, start_sec - 0.1)  # Add 0.1s buffer at the start
            end_sec = min(duration_sec, end_sec + 0.1)  # Add 0.1s buffer at the end
            
            # Extract clip
            self.logger.debug(f"Extracting clip from {video_path}: {start_sec:.2f}s to {end_sec:.2f}s")
            clip = video.get_clip(start_sec, end_sec)
            
            if 'video' not in clip or clip['video'] is None:
                raise ValueError(f"Failed to extract clip from {video_path}")
                
            # Get video tensor and convert to numpy
            video_tensor = clip['video']
            
            # PyTorchVideo returns tensor in format [C, T, H, W], we need to convert to [T, H, W, C]
            video_tensor = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
            
            # If we got more frames than needed, sample the exact frames we wanted
            frames = []
            if video_tensor.shape[0] >= len(frame_indices):
                # Map our original frame indices to positions in the extracted clip
                clip_frame_positions = [idx - start_idx for idx in frame_indices]
                clip_frame_positions = [max(0, min(p, video_tensor.shape[0] - 1)) for p in clip_frame_positions]
                
                # Extract the frames we need
                for pos in clip_frame_positions:
                    frames.append(video_tensor[pos])
            else:
                # If we didn't get enough frames, just use what we got and duplicate the last frame
                frames = [f for f in video_tensor]
                while len(frames) < self.num_frames:
                    frames.append(frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8))
            
            # Ensure we have exactly num_frames
            frames = frames[:self.num_frames]
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8))
            
            # Stack into a single array
            video_array = np.stack(frames)
            
            # Ensure proper shape and dtype
            if video_array.shape[1:3] != (224, 224):
                # Resize each frame to 224x224
                resized_frames = []
                for frame in video_array:
                    resized = cv2.resize(frame, (224, 224))
                    resized_frames.append(resized)
                video_array = np.stack(resized_frames)
            
            # Ensure dtype is uint8
            if video_array.dtype != np.uint8:
                if video_array.max() <= 1.0:
                    video_array = (video_array * 255).astype(np.uint8)
                else:
                    video_array = video_array.astype(np.uint8)
            
            # Return sample
            return {
                'pixel_values': video_array,
                'labels': torch.tensor(label),
                'video_path': str(video_path),
                'frame_indices': frame_indices
            }
            
        except Exception as e:
            self.logger.warning(f"PyTorchVideo error for {video_path}: {str(e)}, falling back to OpenCV")
            
            # Fall back to OpenCV method with improved robustness
            try:
                # Get video properties
                total_frames, original_fps, duration_sec, width, height = self.get_video_properties(video_path)
                
                # Handle videos with insufficient frames
                if total_frames == 0:
                    self.logger.warning(f"Video {video_path} has no frames. Returning placeholder.")
                    placeholder = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
                    return {
                        'pixel_values': placeholder,
                        'labels': torch.tensor(label),
                        'video_path': str(video_path),
                        'frame_indices': []
                    }
                
                # Sample frames based on sampling method
                frame_indices = self.get_sampling_indices(video_path, total_frames)
                
                # Open video and extract frames
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                
                for frame_idx in frame_indices:
                    try:
                        # Constrain frame index to valid range
                        frame_idx = max(0, min(frame_idx, total_frames - 1))
                        
                        # Try direct seeking first
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            # Try reading sequentially from the beginning
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            for _ in range(frame_idx):
                                cap.grab()  # Skip frames without decoding
                            ret, frame = cap.read()
                        
                        if not ret:
                            # Still couldn't read, use a gray placeholder
                            self.logger.warning(
                                f"Could not read frame {frame_idx} from {video_path} " 
                                f"(total frames: {total_frames})"
                            )
                            frame = np.ones((height, width, 3), dtype=np.uint8) * 127
                        
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resize to 224x224 as expected by ViViT
                        frame = cv2.resize(frame, (224, 224))
                        
                        frames.append(frame)
                        
                    except Exception as inner_e:
                        self.logger.warning(f"Error reading frame {frame_idx}: {str(inner_e)}")
                        # Use a gray placeholder frame
                        frame = np.ones((224, 224, 3), dtype=np.uint8) * 127
                        frames.append(frame)
                
                cap.release()
                
                # Stack frames into a single array
                video_tensor = np.stack(frames)
                
                # Return sample
                return {
                    'pixel_values': video_tensor,
                    'labels': torch.tensor(label),
                    'video_path': str(video_path),
                    'frame_indices': frame_indices
                }
                    
            except Exception as e2:
                self.logger.warning(f"OpenCV fallback also failed for {video_path}: {str(e2)}")
                
                # Return a placeholder in case of error
                placeholder = np.ones((self.num_frames, 224, 224, 3), dtype=np.uint8) * 127
                return {
                    'pixel_values': placeholder,
                    'labels': torch.tensor(label),
                    'video_path': str(video_path),
                    'frame_indices': []
                }