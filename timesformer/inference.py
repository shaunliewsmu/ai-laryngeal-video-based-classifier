# timesformer/inference.py

import torch
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor
from timesformer_classifier.utils.logger import ExperimentLogger
from timesformer_classifier.data_config.dataset import VideoDataset

def parse_args():
    parser = argparse.ArgumentParser(description='TimeSformer Transformer Inference')
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
        
        # Create TimesformerConfig object
        from transformers import TimesformerConfig
        
        # Get configuration from checkpoint
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            # Create config from dictionary
            config = TimesformerConfig.from_dict(checkpoint["config"])
        else:
            # Create new config
            config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
            config.num_classes = num_classes
            config.num_frames = num_frames
            
            # Get label mappings from checkpoint or use defaults
            id2label = checkpoint.get('id2label', {0: 'non-referral', 1: 'referral'})
            label2id = checkpoint.get('label2id', {'non-referral': 0, 'referral': 1})
            
            # Set mappings in config
            config.id2label = {str(i): c for i, c in id2label.items()}
            config.label2id = {c: str(i) for c, i in label2id.items()}
        
        # Create model with config
        from transformers import TimesformerForVideoClassification
        model = TimesformerForVideoClassification(config)
        
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

def predict_video(video_path, model, device, num_frames, sampling_method, 
                  logger, save_viz=False):
    """Make prediction on a video using custom sampling method."""
    try:
        # Create a temporary dataset object to use its sampling methods
        temp_dataset = VideoDataset(
            root_dir=Path(video_path).parent.parent.parent,
            mode='inference',
            sampling_method=sampling_method,
            num_frames=num_frames,
            logger=logger
        )
        
        # Get video properties
        total_frames, fps, duration_sec, width, height = temp_dataset.get_video_properties(video_path)
        
        logger.info(f"Loaded video: {video_path} with {total_frames} frames, duration: {duration_sec:.2f}s")
        
        if total_frames == 0:
            logger.error(f"Video has no frames")
            return None, None
        
        # Get frame indices using the dataset's sampling method
        frame_indices = temp_dataset.get_sampling_indices(video_path, total_frames)
        
        logger.info(f"Sampling {num_frames} frames using {sampling_method} sampling method")
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(str(video_path))
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
            # Create visualization directory
            viz_dir = Path('visualization_outputs')
            viz_dir.mkdir(exist_ok=True)
            
            video_name = Path(video_path).stem
            save_path = viz_dir / f"{video_name}_{sampling_method}_sampling.png"
            
            from timesformer_classifier.utils.visualization import TrainingVisualizer
            visualizer = TrainingVisualizer(viz_dir)
            visualizer.visualize_sampling(
                video_path,
                sampling_method,
                num_frames,
                save_path,
                "Inference"
            )
            
            logger.info(f"Saved frame visualization to {save_path}")
        
        # Process frames
        image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        image_processor.size = {"height": 224, "width": 224}
        image_processor.crop_size = {"height": 224, "width": 224}

        inputs = image_processor(
            images=list(video),
            return_tensors="pt",
            do_resize=True,
            size={"height": 224, "width": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224}
        )
        
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
    experiment_logger = ExperimentLogger(args.log_dir, prefix='timesformer-inference')
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
python3 timesformer/inference.py \
  --video_path artifacts/laryngeal_dataset_iqm_filtered:v0/dataset/val/referral/0088_processed.mp4 \
  --model_path timesformer-models/20250305_165129_timesformer-classifier_best_model.pth \
  --num_frames 32 \
  --sampling_method random_window \
  --save_viz
"""