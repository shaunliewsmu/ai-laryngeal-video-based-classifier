import sys
from pathlib import Path

# Add parent directory to Python path
file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    
from tqdm import tqdm  # For progress bar
from src.config import WEIGHTS_PATH, DATA_PATH, DEVICE
from src.yolo_detector import YOLODetector

def process_dataset_videos(detector, input_root_dir, output_root_dir, conf_thres=0.25):
    """
    Process all videos in the dataset directory structure
    
    Args:
        detector: YOLODetector instance
        input_root_dir: Root directory containing dataset folders
        output_root_dir: Root directory for saving processed videos
        conf_thres: Confidence threshold for detections
    """
    # Convert to Path objects
    input_root = Path(input_root_dir)
    output_root = Path(output_root_dir)
    
    # Create output root directory if it doesn't exist
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get all video files in the dataset
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:  # Add more extensions if needed
        video_files.extend(list(input_root.rglob(f'*{ext}')))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            # Get relative path from input root
            rel_path = video_path.relative_to(input_root)
            
            # Create corresponding output directory
            output_dir = output_root / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output video name (remove extension and add _processed)
            output_name = video_path.stem
            
            print(f"\nProcessing: {rel_path}")
            
            # Process the video
            results = detector.process_video_detected_only_raw(
                video_path=str(video_path),
                output_video_name=output_name,
                output_dir=output_dir, 
                conf_thres=conf_thres
            )
            
            # Print results for this video
            print(f"Results for {rel_path}:")
            print(f"Total frames: {results['total_frames']}")
            print(f"Detected frames: {results['detected_frame_count']}")
            if results['detected_frame_count'] > 0:
                avg_conf = sum(f['confidence'] for f in results['detected_frames'])/len(results['detected_frames'])
                print(f"Average confidence: {avg_conf:.3f}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue


if __name__ == "__main__":
   # Initialize the YOLO detector
    detector = YOLODetector(
        weights_path=WEIGHTS_PATH,
        data_path=DATA_PATH,
        device=DEVICE
    )
    
    # Define directories
    dataset_dir = "/home/shaunliew/ai-laryngeal-video-based-classifier/artifacts/laryngeal_dataset_balanced:v0/dataset"
    output_dir = "/home/shaunliew/ai-laryngeal-video-based-classifier/iqm_filtered_dataset"
    
    # Process all videos
    process_dataset_videos(
        detector=detector,
        input_root_dir=dataset_dir,
        output_root_dir=output_dir,
        conf_thres=0.25
    )