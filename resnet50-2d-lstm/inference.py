import argparse
import torch
import os
import glob
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.models.model import VideoResNet50LSTM
from src.utils.logger import ExperimentLogger
from src.utils.visualization import EnhancedVisualizer
from src.config.config import DEFAULT_CONFIG

class EnhancedVideoInference:
    def __init__(self, model_path, sampling_method='uniform', sequence_length=32, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.sampling_method = sampling_method
        self.fps = 30  # Default FPS
        
        # Initialize model
        self.model = VideoResNet50LSTM(
            hidden_size=DEFAULT_CONFIG['hidden_size'],
            num_layers=DEFAULT_CONFIG['num_layers'],
            dropout=DEFAULT_CONFIG['dropout']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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
    
    def random_sampling(self, total_frames, num_frames):
        """Randomly sample frames from the entire video"""
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Ensure num_frames doesn't exceed total_frames
        num_frames = min(num_frames, total_frames)
        
        # Random sampling without replacement if possible
        if total_frames >= num_frames:
            frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
        else:
            # If total_frames < num_frames, sample with replacement
            frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=True))
            
        return frame_indices
    
    def uniform_sampling(self, total_frames, num_frames):
        """Sample frames at regular intervals across the video"""
        # Ensure num_frames doesn't exceed total_frames
        num_frames = min(num_frames, total_frames)
        
        if num_frames == 1:
            return [total_frames // 2]  # Middle frame
        
        if total_frames >= num_frames:
            # Calculate step size for uniform sampling
            step = (total_frames - 1) / (num_frames - 1)
            frame_indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
        else:
            # For shorter videos, we might need to duplicate frames
            # Create evenly spaced indices that might include duplicates
            step = total_frames / num_frames
            frame_indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
            
        return frame_indices
    
    def random_window_sampling(self, total_frames, num_frames):
        """Divide video into equal windows and randomly sample from each"""
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Ensure num_frames doesn't exceed total_frames
        num_frames = min(num_frames, total_frames)
        
        # Calculate window size
        window_size = total_frames / num_frames
        
        frame_indices = []
        for i in range(num_frames):
            start = int(i * window_size)
            end = min(int((i + 1) * window_size), total_frames)
            end = max(end, start + 1)  # Ensure window has at least 1 frame
            frame_idx = np.random.randint(start, end)
            frame_indices.append(frame_idx)
        
        return frame_indices
    
    def predict_video(self, video_path, visualizer=None, viz_dir=None):
        try:
            # Get video properties
            total_frames, fps, duration_sec, width, height = self.get_video_properties(video_path)
            
            # Sample frames based on sampling method
            if self.sampling_method == 'random':
                frame_indices = self.random_sampling(total_frames, self.sequence_length)
            elif self.sampling_method == 'random_window':
                frame_indices = self.random_window_sampling(total_frames, self.sequence_length)
            else:  # default to uniform
                frame_indices = self.uniform_sampling(total_frames, self.sequence_length)
            
            # Visualize sampling if requested
            if visualizer and viz_dir:
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
                viz_path = Path(viz_dir) / f'inference_sampling_{Path(video_path).stem}.png'
                visualizer.visualize_sampling(
                    video_path, 
                    self.sampling_method, 
                    self.sequence_length, 
                    viz_path, 
                    f"Inference Sampling"
                )
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
                probability = torch.sigmoid(outputs).cpu().numpy()[0][0]
                prediction = 'referral' if probability >= 0.5 else 'non_referral'
            
            return {
                'video_path': video_path,
                'prediction': prediction,
                'probability': float(probability),
                'sampled_frames': len(frame_indices),
                'total_frames': total_frames,
                'frame_indices': frame_indices,
                'status': 'success'
            }
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return {
                'video_path': video_path,
                'prediction': None,
                'probability': None,
                'status': f'error: {str(e)}'
            }

def main():
    parser = argparse.ArgumentParser(description='Enhanced inference for ResNet50-LSTM laryngeal cancer classification')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing videos to process')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--sampling_method', type=str, default='uniform',
                        choices=['uniform', 'random', 'random_window'],
                        help='Frame sampling method')
    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Number of frames to sample from each video')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable detailed result visualization')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process all videos in the directory')
    parser.add_argument('--single_video', type=str, default=None,
                        help='Path to a single video for inference (overrides videos_dir)')
    args = parser.parse_args()

    # Create experiment logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_logger = ExperimentLogger(
        os.path.join(args.output_dir, f'inference_{timestamp}'),
        prefix='inference'
    )
    logger = exp_logger.get_logger()
    
    # Initialize visualizer if needed
    if args.visualize:
        viz_dir = exp_logger.get_visualization_dir()
        visualizer = EnhancedVisualizer(viz_dir)
    else:
        viz_dir = None
        visualizer = None
    
    # Initialize inference class
    inference = EnhancedVideoInference(
        model_path=args.model_path,
        sampling_method=args.sampling_method,
        sequence_length=args.sequence_length,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Get videos to process
    if args.single_video:
        if os.path.exists(args.single_video):
            video_files = [args.single_video]
        else:
            logger.error(f"Video file not found: {args.single_video}")
            return
    elif args.batch_mode:
        video_extensions = ('*.mp4', '*.avi', '*.mov')
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.videos_dir, '**', ext), recursive=True))
    else:
        video_files = []
        for dir_name in ['referral', 'non_referral']:
            dir_path = os.path.join(args.videos_dir, dir_name)
            if os.path.exists(dir_path):
                video_files.extend(glob.glob(os.path.join(dir_path, '*.mp4')))

    if not video_files:
        logger.error(f"No video files found to process")
        return

    logger.info(f"Found {len(video_files)} videos to process")

    # Process videos
    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        result = inference.predict_video(video_path, visualizer, viz_dir)
        results.append(result)
        
        # Print the result
        if result['status'] == 'success':
            confidence = result['probability'] if result['prediction'] == 'referral' else 1 - result['probability']
            logger.info(f"Video: {os.path.basename(video_path)}, "
                      f"Prediction: {result['prediction']}, "
                      f"Confidence: {confidence:.4f}")
        else:
            logger.error(f"Error processing {os.path.basename(video_path)}: {result['status']}")

    # Create summary
    successful_predictions = [r for r in results if r['status'] == 'success']
    failed_predictions = [r for r in results if r['status'] != 'success']

    if successful_predictions:
        referral_count = sum(1 for r in successful_predictions if r['prediction'] == 'referral')
        non_referral_count = sum(1 for r in successful_predictions if r['prediction'] == 'non_referral')

        summary = {
            'total_videos': len(results),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'referral_cases': referral_count,
            'non_referral_cases': non_referral_count,
            'sampling_method': args.sampling_method,
            'sequence_length': args.sequence_length,
            'model_path': args.model_path
        }

        # Save results as CSV files
        results_df = pd.DataFrame(successful_predictions)
        results_df.to_csv(os.path.join(exp_logger.get_experiment_dir(), 'inference_results.csv'), index=False)
        
        if failed_predictions:
            failed_df = pd.DataFrame(failed_predictions)
            failed_df.to_csv(os.path.join(exp_logger.get_experiment_dir(), 'inference_failures.csv'), index=False)
        
        # Save summary as both JSON and CSV
        with open(os.path.join(exp_logger.get_experiment_dir(), 'inference_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        pd.DataFrame([summary]).to_csv(os.path.join(exp_logger.get_experiment_dir(), 'inference_summary.csv'), index=False)

        # Visualize results distribution if visualizer is available
        if visualizer:
            try:
                import matplotlib.pyplot as plt
                
                # Create prediction distribution pie chart
                plt.figure(figsize=(10, 6))
                plt.pie([non_referral_count, referral_count], 
                       labels=['Non-Referral', 'Referral'],
                       autopct='%1.1f%%', 
                       colors=['#3498db', '#e74c3c'],
                       explode=(0.05, 0.05))
                plt.title('Prediction Distribution')
                plt.savefig(os.path.join(viz_dir, 'prediction_distribution.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
                # Create confidence histogram
                plt.figure(figsize=(10, 6))
                confidence_values = [r['probability'] if r['prediction'] == 'referral' else 1-r['probability'] 
                                   for r in successful_predictions]
                plt.hist(confidence_values, bins=20, color='skyblue', edgecolor='black')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.title('Prediction Confidence Distribution')
                plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(viz_dir, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.error(f"Error creating visualization: {str(e)}")

        # Print summary
        logger.info("\nInference Summary:")
        logger.info(f"Total videos processed: {len(results)}")
        logger.info(f"Successful predictions: {len(successful_predictions)}")
        logger.info(f"Failed predictions: {len(failed_predictions)}")
        logger.info(f"Referral cases: {referral_count}")
        logger.info(f"Non-referral cases: {non_referral_count}")
        logger.info(f"Results saved to {exp_logger.get_experiment_dir()}")
    else:
        logger.error("No successful predictions were made")

if __name__ == "__main__":
    main()
    
"""
Example usage:
python3 resnet50-2d-lstm/inference.py \
    --videos_dir artifacts/duhs-gss-split-5:v0/organized_dataset/test \
    --model_path resnet50-2d-lstm-models/model_20250304_170408.pth \
    --output_dir enhanced_inference_results \
    --sampling_method uniform \
    --sequence_length 32 \
    --visualize \
    --batch_mode
"""