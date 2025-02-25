import torch
import argparse
import json
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

from swin_video_classifier.models.swin3d import create_model
from swin_video_classifier.utils.logger import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Video Swin Transformer Inference')
    parser.add_argument('--video_path', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model checkpoint')
    parser.add_argument('--model_size', type=str, default='tiny',
                      choices=['tiny', 'small', 'base', 'base_in22k'],
                      help='Size of Swin Transformer model')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save inference logs')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='Number of frames to sample')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
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

def load_model(model_path, model_size, num_classes, device, logger):
    """Load the saved model."""
    logger.info(f"Loading model from {model_path}")
    
    # Create model
    model = create_model(logger, model_size=model_size, pretrained=False, num_classes=num_classes)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if model was saved with DataParallel
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists (from DataParallel)
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

def save_inference_result(exp_logger, video_path, class_name, confidence):
    """Save inference results to a JSON file."""
    # Prepare result dictionary
    result = {
        "video_path": str(video_path),
        "predicted_class": class_name,
        "confidence": confidence
    }
    
    # Create results directory if it doesn't exist
    results_dir = exp_logger.get_experiment_dir() / 'inference_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on video name
    video_filename = Path(video_path).stem
    result_path = results_dir / f'{video_filename}_result.json'
    
    # Save results to JSON
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    exp_logger.get_logger().info(f"Inference result saved to {result_path}")

def main():
    args = parse_args()
    
    # Use ExperimentLogger with a specific prefix for inference
    experiment_logger = ExperimentLogger(args.log_dir, prefix=f'inference-swin3d-{args.model_size}')
    logger = experiment_logger.get_logger()
    
    try:
        # Set device to GPU 1 specifically, can switch it back anytime if GPU 0 is available
        if torch.cuda.is_available():
            device = torch.device("cuda:1")  # Use GPU 1 specifically
            torch.cuda.set_device(1)  # Also set current device to ensure all operations use GPU 1
            logger.info(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")
            
        logger.info(f"Using device: {device}")
        
        # Create transform
        transform = create_video_transform(args.num_frames)
        
        # Load model
        model = load_model(args.model_path, args.model_size, args.num_classes, device, logger)
        
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
        
        # Save inference result
        save_inference_result(experiment_logger, args.video_path, class_name, confidence)
        
        print(f"\nResults:")
        print(f"Class: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

"""
python3 videoswintransformer/inference.py \
  --video_path artifacts/laryngeal_dataset_balanced:v0/dataset/val/referral/0047.mp4 \
  --model_path swin3d-models/20250225_162321_swin3d-tiny_best_model.pth \
  --model_size tiny \
  --num_frames 32 \
  --fps 8
"""