import torch
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import VivitImageProcessor
from pytorchvideo.data import make_clip_sampler
from vivit_classifier.utils.logger import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser(description='ViViT Transformer Inference')
    parser.add_argument('--video_path', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to saved model checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs',
                      help='Directory to save inference logs')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='Number of frames to sample')
    parser.add_argument('--sampling_method', type=str, default='uniform',
                      choices=['random', 'uniform', 'random_window'],
                      help='Method to sample frames from video')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    parser.add_argument('--save_viz', action='store_true',
                      help='Save visualization of sampled frames')
    return parser.parse_args()

def load_model(model_path, num_frames, num_classes, device, logger):
    """Load the saved model."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create VivitConfig object
        from transformers import VivitConfig
        
        # Get configuration from checkpoint
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            # Create config from dictionary
            config = VivitConfig.from_dict(checkpoint["config"])
        else:
            # Create new config
            config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
            config.num_classes = num_classes
            config.num_frames = num_frames
            
            # Get label mappings from checkpoint or use defaults
            id2label = checkpoint.get('id2label', {0: 'non-referral', 1: 'referral'})
            label2id = checkpoint.get('label2id', {'non-referral': 0, 'referral': 1})
            
            # Set mappings in config
            config.id2label = {str(i): c for i, c in id2label.items()}
            config.label2id = {c: str(i) for c, i in label2id.items()}
        
        # Create model with config
        from transformers import VivitForVideoClassification
        model = VivitForVideoClassification(config)
        
        # Load saved weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(device)
        model.eval()
        
        # Extract id2label for later use
        id2label = {int(k): v for k, v in config.id2label.items()}
        
        logger.info("Model loaded successfully")
        return model, id2label
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    
    if len(frames) == 0:
        raise ValueError(f"No frames found between indices {start_index} and {end_index}")
    
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def predict_video(video_path, model, device, num_frames, sampling_method, 
                   logger, save_viz=False):
    """Make prediction on a video using custom frame sampling."""
    try:
        # Load video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = float(total_frames) / video_fps
        
        logger.info(f"Loaded video: {video_path} with {total_frames} frames, duration: {video_duration:.2f}s")
        
        if total_frames == 0:
            logger.error(f"Video has no frames")
            return None, None
        
        # Create a dataset instance to use its sampling methods
        from vivit_classifier.data_config.dataset import VideoDataset
        temp_dataset = VideoDataset(
            root_dir=Path(video_path).parent.parent.parent,
            mode="inference",
            sampling_method=sampling_method,
            num_frames=num_frames,
            logger=logger
        )
        
        # Get frame indices using the dataset's sampling method
        frame_indices = temp_dataset.get_sampling_indices(video_path, total_frames)
        
        logger.info(f"Sampling {num_frames} frames using {sampling_method} sampling method")
        
        # Extract frames directly
        frames = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Could not read frame {frame_idx}, using placeholder")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        
        # Stack frames into a video tensor
        video = np.stack(frames)
        logger.info(f"Extracted {len(video)} frames with shape {video.shape}")
        
        # Save visualization if requested
        if save_viz:
            video_name = Path(video_path).stem
            save_dir = Path('visualization_outputs')
            save_dir.mkdir(exist_ok=True)
            
            from vivit_classifier.utils.visualization import TrainingVisualizer
            visualizer = TrainingVisualizer(save_dir)
            save_path = save_dir / f"{video_name}_{sampling_method}_sampling.png"
            
            visualizer.visualize_sampling(
                video_path,
                sampling_method,
                num_frames,
                save_path,
                "Inference"
            )
            
            logger.info(f"Saved frame visualization to {save_path}")
        
        # Process frames
        image_processor = VivitImageProcessor(
            num_frames=num_frames,
            image_size=224,
            patch_size=16,
        )
        
        inputs = image_processor(list(video), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class.item(), confidence
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def save_frame_visualization(video, video_name, num_frames_to_show=6):
    """Save a grid visualization of sampled frames."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Select evenly spaced frames to display
    step = len(video) // num_frames_to_show
    indices = [i*step for i in range(num_frames_to_show)]
    
    for i, idx in enumerate(indices):
        if i < len(axes) and idx < len(video):
            axes[i].imshow(video[idx])
            axes[i].set_title(f"Frame {idx}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{video_name}_sampled_frames.png")
    plt.close()

def save_inference_result(exp_logger, video_path, class_name, confidence, id2label):
    """Save inference results to a JSON file."""
    # Prepare result dictionary
    result = {
        "video_path": str(video_path),
        "predicted_class": class_name,
        "class_id": id2label.get(class_name, -1),
        "confidence": confidence,
        "class_mapping": id2label
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
    
    # Setup logger
    experiment_logger = ExperimentLogger(args.log_dir, prefix='vivit-inference')
    logger = experiment_logger.get_logger()
    
    try:
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU")
        
        # Load model
        model, id2label = load_model(
            args.model_path, 
            args.num_frames,
            args.num_classes, 
            device, 
            logger
        )
        
        # Make prediction using specified sampling method
        class_id, confidence = predict_video(
            args.video_path,
            model,
            device,
            args.num_frames,
            args.sampling_method,
            logger,
            args.save_viz
        )
        
        if class_id is not None:
            # Map class ID to name
            class_name = id2label.get(class_id, f"Unknown-{class_id}")
            
            # Save inference result
            save_inference_result(
                experiment_logger, 
                args.video_path, 
                class_name, 
                confidence,
                id2label
            )
            
            print(f"\nResults:")
            print(f"Class: {class_name}")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("\nCould not process video - insufficient frames")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

"""
Example usage:
python3 vivit_transformer/inference.py \
  --video_path artifacts/laryngeal_dataset_iqm_filtered:v0/dataset/val/referral/0088_processed.mp4 \
  --model_path vivit-models/20250228_123221_vivit-classifier_best_model.pth \
  --num_frames 32 \
  --sampling_method uniform \
  --save_viz
"""