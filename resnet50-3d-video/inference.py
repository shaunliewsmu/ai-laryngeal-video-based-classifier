import torch
import argparse
import json
import random
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from video_classifier.models.resnet3d import create_model
from video_classifier.utils.logger import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification Inference')
    parser.add_argument('--video_path', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save inference logs')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='Number of frames to sample')
    parser.add_argument('--sampling_method', type=str, default='uniform',
                      choices=['uniform', 'random', 'random_window'],
                      help='Frame sampling method')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the frame sampling')
    return parser.parse_args()

class VideoInference:
    def __init__(self, model_path, sampling_method='uniform', num_frames=32, fps=30, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_frames = num_frames
        self.sampling_method = sampling_method
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Initialize model
        from video_classifier.models.resnet3d import create_model
        self.model = create_model(logging.getLogger())
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint) # Handle different checkpoint formats
        
        # Remove 'module.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup normalization values
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
    
    def get_video_properties(self, video_path):
        """Get video properties using OpenCV"""
        cap = cv2.VideoCapture(video_path)
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
            
            print(f"Dynamic FPS adjustment: Video has {total_frames} frames, "
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
    
    def get_frame_from_video(self, video_path, frame_idx):
        """Extract a specific frame from a video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        cap.release()
        
        if not ret:
            print(f"Could not read frame {frame_idx}, using placeholder")
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def visualize_sampling(self, video_path, frame_indices, dynamic_fps, save_path):
        """
        Visualize frame sampling pattern on a video with dynamic FPS adjustment for short videos.
        
        Args:
            video_path (str): Path to the video file
            frame_indices (list): List of frame indices to visualize
            dynamic_fps (float): Dynamic FPS value if applicable, None otherwise
            save_path (str): Path to save the visualization
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / original_fps  # in seconds
        
        # Create figure with 2 rows
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [1, 3]})
        
        # Setup title
        method_name = f"{self.sampling_method.replace('_', ' ').title()}"
        if dynamic_fps:
            method_name += f" (Dynamic FPS: {dynamic_fps:.2f})"
            fps_info = f" (Dynamic FPS: {dynamic_fps:.2f})"
        else:
            fps_info = ""
            
        frames_info = f"{len(frame_indices)} frames from {total_frames} total{fps_info}"
        full_title = f"Inference - {method_name}\n{frames_info}"
        
        # Top row: sampling pattern visualization
        ax1.plot([0, total_frames], [0, 0], 'k-', linewidth=2)
        ax1.plot([0, 0], [-0.2, 0.2], 'k-', linewidth=2)
        ax1.plot([total_frames, total_frames], [-0.2, 0.2], 'k-', linewidth=2)
        
        # Mark windows for Random Window Sampling method
        if self.sampling_method == 'random_window':
            window_size = total_frames / self.num_frames
            for i in range(self.num_frames):
                start = int(i * window_size)
                end = int((i + 1) * window_size) if i < self.num_frames - 1 else total_frames
                ax1.axvspan(start, end, alpha=0.1, color='gray')
                ax1.plot([start, start], [-0.2, 0.2], 'k--', alpha=0.5)
        
        # Mark sampled frames
        for i, frame_idx in enumerate(frame_indices):
            color = plt.cm.rainbow(i / len(frame_indices))
            ax1.plot([frame_idx, frame_idx], [-0.3, 0.3], '-', color=color, linewidth=2)
            ax1.plot(frame_idx, 0, 'o', color=color, markersize=8)
            ax1.text(frame_idx, 0.1, f'{frame_idx}', 
                   horizontalalignment='center', color=color, fontsize=9)
        
        # Customize plot
        ax1.set_title(full_title)
        ax1.set_xlabel('Frame Index')
        ax1.set_xlim(-total_frames*0.05, total_frames*1.05)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_yticks([])
        
        # Display time marks
        time_marks = 5
        for i in range(time_marks + 1):
            frame = int(i * total_frames / time_marks)
            time = frame / original_fps
            ax1.text(frame, -0.2, f'{time:.1f}s', 
                   horizontalalignment='center', color='black', fontsize=8)
        
        # Bottom row: display frames
        ax2.axis('off')
        
        # Calculate grid layout
        cols = min(8, len(frame_indices))
        rows = (len(frame_indices) + cols - 1) // cols
        
        # Extract frames and display in grid
        for i, frame_idx in enumerate(frame_indices):
            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Placeholder for missing frame
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate position in grid
            row, col = i // cols, i % cols
            
            # Create subplot
            sub_ax = ax2.inset_axes([col/cols, 1-(row+1)/rows, 1/cols, 1/rows])
            
            # Display frame
            sub_ax.imshow(frame)
            sub_ax.axis('off')
            sub_ax.set_title(f'Frame {frame_idx}\n({frame_idx/original_fps:.2f}s)', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        cap.release()
    
    def predict_video(self, video_path, visualize=False, viz_dir=None):
        try:
            # Get video properties
            total_frames, original_fps, duration_sec, width, height = self.get_video_properties(video_path)
            
            # Sample frames based on sampling method
            frame_indices, dynamic_fps = self.get_sampling_indices(video_path, total_frames)
            
            # Visualize sampling if requested
            if visualize and viz_dir:
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
                viz_path = Path(viz_dir) / f'sampling_{Path(video_path).stem}_{self.sampling_method}.png'
                self.visualize_sampling(video_path, frame_indices, dynamic_fps, viz_path)
                print(f"Saved sampling visualization to {viz_path}")
            
            # Extract frames
            frames = []
            for frame_idx in frame_indices:
                frame = self.get_frame_from_video(video_path, frame_idx)
                frame = cv2.resize(frame, (224, 224))  # Resize to model input size
                frames.append(frame)
            
            # Stack frames into a batch
            frames = np.stack(frames, axis=0)
            frames = frames.astype(np.float32) / 255.0
            
            # Convert to tensor
            frames = torch.FloatTensor(frames)
            
            # Apply normalization
            frames = frames - torch.tensor(self.mean).view(1, 1, 1, 3)
            frames = frames / torch.tensor(self.std).view(1, 1, 1, 3)
            
            # Rearrange to [C, T, H, W] as expected by the model
            frames = frames.permute(3, 0, 1, 2)
            frames = frames.unsqueeze(0)  # Add batch dimension [B, C, T, H, W]
            
            # Make prediction
            with torch.no_grad():
                frames = frames.to(self.device)
                outputs = self.model(frames)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = probabilities[0][predicted_class].item()
            
            prediction = 'referral' if predicted_class.item() == 1 else 'non-referral'
            
            return {
                'video_path': video_path,
                'prediction': prediction,
                'confidence': float(confidence),
                'status': 'success',
                'sampled_frames': len(frame_indices),
                'total_frames': total_frames,
                'dynamic_fps': dynamic_fps,
                'sampling_method': self.sampling_method
            }
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return {
                'video_path': video_path,
                'prediction': None,
                'confidence': None,
                'status': f'error: {str(e)}'
            }

def main():
    args = parse_args()
    
    # Use ExperimentLogger with a specific prefix for inference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_logger = ExperimentLogger(args.log_dir, prefix=f'inference-{timestamp}')
    logger = exp_logger.get_logger()
    viz_dir = exp_logger.get_experiment_dir() / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize inference class
        inference = VideoInference(
            model_path=args.model_path,
            sampling_method=args.sampling_method,
            num_frames=args.num_frames,
            device=device
        )
        
        # Make prediction
        result = inference.predict_video(
            args.video_path,
            visualize=args.visualize,
            viz_dir=viz_dir
        )
        
        # Save result to file
        result_path = exp_logger.get_experiment_dir() / f"result_{Path(args.video_path).stem}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        if result['status'] == 'success':
            # Add dynamic FPS info if applicable
            fps_info = f" (Dynamic FPS: {result['dynamic_fps']:.2f})" if result['dynamic_fps'] else ""
            
            print(f"\nResults for {args.video_path}:")
            print(f"Class: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Sampling: {result['sampling_method']} with {result['sampled_frames']} frames{fps_info}")
            print(f"Results saved to {result_path}")
        else:
            print(f"Error: {result['status']}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    
"""
Example usage:
python3 resnet50-3d-video/inference.py \
    --video_path artifacts/duhs-gss-split-5:v0/organized_dataset/test/non_referral/0031.mp4 \
    --model_path model/20250220_175039_resnet50_best_model.pth \
    --num_frames 32 \
    --sampling_method uniform \
    --visualize
"""