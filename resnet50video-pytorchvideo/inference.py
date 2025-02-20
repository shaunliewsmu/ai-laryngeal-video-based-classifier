import torch
import argparse
from pathlib import Path
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from video_classifier.models.resnet3d import create_model
from video_classifier.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification Inference')
    parser.add_argument('--video_path', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model checkpoint')
    parser.add_argument('--log_dir', type=str, default='inference_logs',
                      help='Directory to save inference logs')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='Number of frames to sample')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second')
    return parser.parse_args()

def create_video_transform(num_frames):
    """Create video transform for inference."""
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            ShortSideScale(256),
            CenterCropVideo(224),
            NormalizeVideo(
                mean=[0.45, 0.45, 0.45], 
                std=[0.225, 0.225, 0.225]
            ),
        ]),
    )

def load_model(model_path, device, logger):
    """Load the saved model."""
    logger.info(f"Loading model from {model_path}")
    
    # Create model
    model = create_model(logger)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model

def predict_video(video_path, model, transform, device, num_frames, fps, logger):
    """Make prediction on a video."""
    try:
        # Load video
        video = EncodedVideo.from_path(video_path)
        logger.info(f"Loaded video: {video_path}")
        
        # Calculate clip duration
        clip_duration = float(num_frames) / fps
        
        # Extract video clip
        video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
        
        # Apply transforms
        video_data = transform(video_data)
        inputs = video_data["video"]
        inputs = inputs.to(device)
        
        # Add batch dimension
        inputs = inputs.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        class_name = "referral" if predicted_class.item() == 1 else "non-referral"
        logger.info(f"Prediction: {class_name} (confidence: {confidence:.4f})")
        
        return class_name, confidence
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def main():
    args = parse_args()
    logger = setup_logger(args.log_dir)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Create transform
        transform = create_video_transform(args.num_frames)
        
        # Load model
        model = load_model(args.model_path, device, logger)
        
        # Make prediction
        class_name, confidence = predict_video(
            args.video_path,
            model,
            transform,
            device,
            args.num_frames,
            args.fps,
            logger
        )
        
        print(f"\nResults:")
        print(f"Class: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
    
"""
python3 resnet50video-pytorchvideo/inference.py \
    --video_path artifacts/laryngeal_dataset_balanced:v0/dataset/val/referral/0047.mp4 \
    --model_path resnet50-models/best_model.pth \
    --num_frames 32 \
    --fps 2
"""