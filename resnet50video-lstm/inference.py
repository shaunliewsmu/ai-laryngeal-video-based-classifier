import argparse
import torch
from torchvision import transforms
import os
import glob
import logging
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.models.model import VideoResNetLSTM
from src.data_config.sampling import VideoSampling
from src.utils.logging_utils import setup_logging, create_directories
from src.config.config import DEFAULT_CONFIG

class VideoInference:
    def __init__(self, model_path, sampling_method='uniform', sequence_length=32, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.sampling_method = sampling_method
        self.sampler = VideoSampling.get_sampler(sampling_method)
        
        # Initialize model
        self.model = VideoResNetLSTM(
            hidden_size=DEFAULT_CONFIG['hidden_size'],
            num_layers=DEFAULT_CONFIG['num_layers'],
            dropout=DEFAULT_CONFIG['dropout']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]),
        ])
    
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
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
        frames = np.stack(frames, axis=0)
        frames = frames.astype(np.float32) / 255.0
        
        if self.transform:
            frames = self.transform(frames)
        
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        return frames
    
    def predict_video(self, video_path):
        try:
            frames = self.load_video(video_path)
            frames = frames.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(frames)
                probability = torch.sigmoid(outputs).cpu().numpy()[0][0]
                prediction = 'referral' if probability >= 0.5 else 'non_referral'
                
            return {
                'video_path': video_path,
                'prediction': prediction,
                'probability': float(probability),
                'status': 'success'
            }
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {str(e)}")
            return {
                'video_path': video_path,
                'prediction': None,
                'probability': None,
                'status': f'error: {str(e)}'
            }

def main():
    parser = argparse.ArgumentParser(description='Inference script for laryngeal cancer video classification')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing videos to process')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--sampling_method', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method')
    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Number of frames to sample from each video')
    args = parser.parse_args()

    # Create output directory
    create_directories(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.output_dir, f'inference_{timestamp}')
    setup_logging(os.path.join(log_path, 'inference.log'))
    
    # Initialize inference class
    inference = VideoInference(
        model_path=args.model_path,
        sampling_method=args.sampling_method,
        sequence_length=args.sequence_length
    )

    # Get all video files
    video_extensions = ('*.mp4', '*.avi', '*.mov')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(args.videos_dir, '**', ext), recursive=True))

    if not video_files:
        logging.error(f"No video files found in {args.videos_dir}")
        return

    logging.info(f"Found {len(video_files)} videos to process")

    # Process videos
    results = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        result = inference.predict_video(video_path)
        results.append(result)

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
            'non_referral_cases': non_referral_count
        }

        # Save results as CSV files
        csv_prefix = os.path.join(log_path, 'inference_results')
        results_df = pd.DataFrame(successful_predictions)
        results_df.to_csv(f'{csv_prefix}_successful.csv', index=False)
        
        if failed_predictions:
            failed_df = pd.DataFrame(failed_predictions)
            failed_df.to_csv(f'{csv_prefix}_failed.csv', index=False)
        
        # Save summary as CSV
        pd.DataFrame([summary]).to_csv(f'{csv_prefix}_summary.csv', index=False)

        # Save detailed JSON
        json_path = os.path.join(log_path, 'inference_results.json')
        with open(json_path, 'w') as f:
            json.dump({
                'summary': summary,
                'predictions': successful_predictions,
                'failures': failed_predictions
            }, f, indent=4)

        # Print summary
        logging.info("\nInference Summary:")
        logging.info(f"Total videos processed: {len(results)}")
        logging.info(f"Successful predictions: {len(successful_predictions)}")
        logging.info(f"Failed predictions: {len(failed_predictions)}")
        logging.info(f"Referral cases: {referral_count}")
        logging.info(f"Non-referral cases: {non_referral_count}")
        logging.info(f"\nResults saved to {log_path}")

if __name__ == "__main__":
    main()
    
"""
python3 resnet50video-lstm/inference.py \
    --videos_dir artifacts/laryngeal_dataset_iqm_filtered:v0/dataset/val/non_referral \
    --model_path resnet-models/best_model.pth \
    --output_dir resnet_inference_results \
    --sampling_method uniform \
    --sequence_length 32
"""