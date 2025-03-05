# videoswintransformer/inference.py

import torch
import argparse
import json
import random
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

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
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    parser.add_argument('--sampling_method', type=str, default='uniform',
                      choices=['uniform', 'random', 'random_window'],
                      help='Frame sampling method')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the frame sampling')
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

def get_sampling_indices(video_path, total_frames, num_frames, sampling_method, logger):
    """
    Get frame indices based on sampling method, with dynamic FPS for short videos.
    
    Args:
        video_path (str): Path to the video file
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        sampling_method (str): Sampling method ('uniform', 'random', 'random_window')
        logger: Logger instance
        
    Returns:
        list: Frame indices to sample
        float: Dynamic FPS used if applicable, None otherwise
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize dynamic FPS to None
    dynamic_fps = None
    
    # For videos with enough frames, use standard sampling
    if total_frames >= num_frames:
        if sampling_method == 'random':
            # Random sampling without replacement
            indices = sorted(random.sample(range(total_frames), num_frames))
        elif sampling_method == 'random_window':
            # Random window sampling
            window_size = total_frames / num_frames
            indices = []
            for i in range(num_frames):
                start = int(i * window_size)
                end = min(int((i + 1) * window_size), total_frames)
                end = max(end, start + 1)  # Ensure window has at least 1 frame
                frame_idx = random.randint(start, end - 1)
                indices.append(frame_idx)
        else:  # Default to uniform sampling
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                step = (total_frames - 1) / (num_frames - 1)
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    # For videos with fewer frames than requested, use dynamic FPS adjustment
    else:
        # Get original video FPS for proper time scaling
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / original_fps  # in seconds
        cap.release()
        
        # Calculate optimal sampling rate to get the exact number of frames requested
        dynamic_fps = num_frames / duration
        
        logger.info(f"Dynamic FPS adjustment: Video has {total_frames} frames, "
                  f"adjusted from {original_fps} to {dynamic_fps:.2f} fps to get {num_frames} frames.")
        
        # Use the sampling method with the adjusted parameters
        if sampling_method == 'random':
            # With dynamic FPS, we'll need to allow duplicates since total_frames < num_frames
            # Use random.choices which allows replacement
            indices = sorted(random.choices(range(total_frames), k=num_frames))
        elif sampling_method == 'random_window':
            # For random window with fewer frames, create virtual windows smaller than 1 frame
            indices = []
            window_size = total_frames / num_frames  # Will be < 1
            
            for i in range(num_frames):
                # Calculate virtual window boundaries
                virtual_start = i * window_size
                virtual_end = (i + 1) * window_size
                
                # Convert to actual frame indices with potential duplicates
                actual_index = min(int(np.floor(virtual_start + (virtual_end - virtual_start) * random.random())), 
                                  total_frames - 1)
                indices.append(actual_index)
        else:  # Uniform sampling
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                # Create evenly spaced indices that might include duplicates
                step = total_frames / num_frames
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    
    return indices, dynamic_fps

def visualize_sampling(video_path, frame_indices, sampling_method, dynamic_fps, save_path, logger):
    """
    Visualize frame sampling pattern on a video with dynamic FPS adjustment for short videos.
    
    Args:
        video_path (str): Path to the video file
        frame_indices (list): List of frame indices to visualize
        sampling_method (str): Sampling method used
        dynamic_fps (float): Dynamic FPS value if applicable, None otherwise
        save_path (str): Path to save the visualization
        logger: Logger instance
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
    method_name = f"{sampling_method.replace('_', ' ').title()}"
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
    if sampling_method == 'random_window':
        window_size = total_frames / len(frame_indices)
        for i in range(len(frame_indices)):
            start = int(i * window_size)
            end = int((i + 1) * window_size) if i < len(frame_indices) - 1 else total_frames
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
    logger.info(f"Frame sampling visualization saved to {save_path}")

def predict_video(video_path, model, transform, device, args, viz_dir, logger):
    """Make prediction on a video with visualization."""
    try:
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Get frame indices using our sampling method
        frame_indices, dynamic_fps = get_sampling_indices(
            video_path, 
            total_frames, 
            args.num_frames, 
            args.sampling_method,
            logger
        )
        
        # Visualize sampling if requested
        if args.visualize:
            viz_path = viz_dir / f'sampling_{Path(video_path).stem}_{args.sampling_method}.png'
            visualize_sampling(
                video_path, 
                frame_indices, 
                args.sampling_method,
                dynamic_fps,
                viz_path,
                logger
            )
        
        # Load video
        video = EncodedVideo.from_path(video_path)
        
        # Calculate clip duration from frame indices
        # This assumes constant framerate throughout the video
        if original_fps <= 0:  # Ensure FPS is valid
                # Use a default fallback FPS if needed
                logger.warning(f"Invalid FPS value ({original_fps}), using default 30 FPS")
                original_fps = 30.0
        start_sec = frame_indices[0] / original_fps
        end_sec = (frame_indices[-1] + 1) / original_fps
        
        # Extract the clip
        video_data = video.get_clip(start_sec, end_sec)
        
        # Apply transforms
        if transform is not None:
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
        
        result = {
            "video_path": str(video_path),
            "predicted_class": class_name,
            "confidence": confidence,
            "sampling_method": args.sampling_method,
            "frames_sampled": len(frame_indices),
            "total_frames": total_frames
        }
        
        if dynamic_fps:
            result["dynamic_fps"] = float(dynamic_fps)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def save_inference_result(exp_logger, result):
    """Save inference results to a JSON file."""
    # Create results directory if it doesn't exist
    results_dir = exp_logger.get_experiment_dir() / 'inference_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on video name
    video_filename = Path(result["video_path"]).stem
    result_path = results_dir / f'{video_filename}_result.json'
    
    # Save results to JSON
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    exp_logger.get_logger().info(f"Inference result saved to {result_path}")

def main():
    args = parse_args()
    
    # Use ExperimentLogger with a specific prefix for inference
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_logger = ExperimentLogger(args.log_dir, prefix=f'inference-swin3d-{args.model_size}')
    logger = exp_logger.get_logger()
    
    try:
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Use GPU 0 by default
            logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")
            
        logger.info(f"Using device: {device}")
        
        # Create transform
        transform = create_video_transform(args.num_frames)
        
        # Load model
        model = load_model(args.model_path, args.model_size, args.num_classes, device, logger)
        
        # Create visualization directory
        viz_dir = exp_logger.get_experiment_dir() / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Make prediction
        result = predict_video(
            args.video_path,
            model,
            transform,
            device,
            args,
            viz_dir,
            logger
        )
        
        # Save inference result
        save_inference_result(exp_logger, result)
        
        print(f"\nResults:")
        print(f"Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Sampling: {result['sampling_method']} with {result['frames_sampled']} frames")
        if 'dynamic_fps' in result:
            print(f"Dynamic FPS: {result['dynamic_fps']:.2f}")
        
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
  --sampling_method random_window \
  --visualize
"""