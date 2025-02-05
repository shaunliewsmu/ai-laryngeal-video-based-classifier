import sys
from pathlib import Path

# Add parent directory to Python path
file_path = Path(__file__).resolve()
root_path = file_path.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    
from src.config import WEIGHTS_PATH, DATA_PATH, DEVICE
from src.yolo_detector import YOLODetector
if __name__ == "__main__":

    # Initialize the YOLO detector
    detector = YOLODetector(
        weights_path=WEIGHTS_PATH,
        data_path=DATA_PATH,
        device=DEVICE
    )
    
    VIDEO_NUMBER = '0076'
    INPUT_VIDEO = f'/home/shaunliew/ai-laryngeal-video-based-classifier/artifacts/laryngeal_dataset_balanced:v0/dataset/test/referral/{VIDEO_NUMBER}.mp4'
    
    # Process video example
    video_results = detector.process_video(
        video_path=INPUT_VIDEO,
        output_video_name = VIDEO_NUMBER
    )
    
    # Print results
    print(f"Processed {video_results['total_frames']} frames")
    print(f"Found {len(video_results['good_frames'])} frames with confident glottis detections")
    
    # Process video saving only detected frames
    try:
        results = detector.process_video_detected_only(
            video_path=INPUT_VIDEO,
            conf_thres=0.25,  # Confidence threshold
            output_video_name = VIDEO_NUMBER
        )
        
        # Print results
        print(f"Processed {results['total_frames']} total frames")
        print(f"Found {len(results['detected_frames'])} frames with glottis")
        print(f"Output video saved to: {results['video_url']}")
        
        # Access detection details if needed
        for detection in results['frame_detections']:
            frame_idx = detection['frame_idx']
            detections = detection['detections']
            print(f"Frame {frame_idx} detections:")
            for det in detections:
                print(f"- Confidence: {det['confidence']:.3f}")
                print(f"- Bounding box: {det['bbox']}")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        
        
    # save the detected frames into video and with the raw frames  
    try:
        # Process video to save raw detected frames with automatic fps calculation
        results = detector.process_video_detected_only_raw(
            video_path=INPUT_VIDEO,
            conf_thres=0.25,  # Confidence threshold
            output_video_name = VIDEO_NUMBER
        )
        
        # Print results
        print(f"Video saved to: {results['video_url']}")
        print(f"Total frames processed: {results['total_frames']}")
        print(f"Frames with detections saved: {results['detected_frame_count']}")
        print(f"Input FPS: {results['input_fps']}")
        print(f"Output FPS: {results['output_fps']}")
        
        # Access frame-level detection details if needed
        for frame in results['detected_frames']:
            print(f"Frame {frame['frame_idx']} (output frame {frame['output_frame_idx']}):")
            print(f"Confidence: {frame['confidence']:.3f}")

    except Exception as e:
        print(f"Error processing video: {str(e)}")