import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import yaml
from datetime import datetime
import os
# Add YOLOv5 directory to path
YOLOV5_DIR = Path(__file__).parent.parent / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.append(str(YOLOV5_DIR))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class YOLODetector:
    def __init__(self, weights_path, data_path, device='0'):
        """
        Initialize YOLOv5 detector
        
        Args:
            weights_path (str): Path to model weights
            data_path (str): Path to data.yaml file
            device (str): Device to run inference on ('cpu' or GPU index)
        """
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights_path, device=self.device, data=data_path)
        self.stride = self.model.stride
        self.names = ['glottis'] 
        self.model.names = self.names
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.model.eval()

    def process_image(self, image_array, conf_thres=0.25, iou_thres=0.45, max_det=1000):
        """
        Process a single image array
        
        Args:
            image_array (np.ndarray): Input image array
            conf_thres (float): Confidence threshold
            iou_thres (float): NMS IoU threshold
            max_det (int): Maximum detections per image
            
        Returns:
            dict: Contains detections and annotated image
        """
        # Padded resize
        im = letterbox(image_array, self.imgsz, stride=self.stride)[0]
        
        # Convert
        im = im.transpose((2, 0, 1))[::-1].copy()
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        # Process predictions
        detections = []
        annotator = Annotator(image_array.copy(), line_width=3)
        
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image_array.shape).round()

                # Convert detections to list of dicts
                for *xyxy, conf, cls in reversed(det):
                    detection = {
                        "confidence": float(conf),
                        "bbox": [int(x) for x in xyxy],  # Convert to int for JSON serialization
                        "class": self.names[int(cls)]
                    }
                    detections.append(detection)
                    
                    # Add box to image
                    label = f'glottis {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(0, True))

        return {
            "detections": detections,
            "annotated_image": annotator.result()
        }

    def process_video(self, video_path, output_video_name, save_path=None, conf_thres=0.25):
        try:
            # Create output directory if it doesn't exist
            OUTPUT_DIR = Path("output")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{output_video_name}_processed_video_{timestamp}.mp4"
            save_path = OUTPUT_DIR / output_filename
            print(f"Will save output to: {save_path}")

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {video_path}")
                
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Try different codecs in order of preference
            codecs = ['avc1', 'mp4v', 'H264', 'XVID']
            out = None
            
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(
                        str(save_path),
                        fourcc,
                        fps,
                        (frame_width, frame_height),
                        True  # isColor
                    )
                    if out.isOpened():
                        print(f"Successfully initialized VideoWriter with codec: {codec}")
                        break
                except Exception as e:
                    print(f"Failed to initialize VideoWriter with codec {codec}: {e}")
                    if out is not None:
                        out.release()
                    continue
            
            if out is None or not out.isOpened():
                raise Exception("Failed to initialize video writer with any codec")

            good_frames = []
            frame_detections = []
            frame_idx = 0

            print(f"Starting video processing: {total_frames} frames total")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                results = self.process_image(frame, conf_thres=conf_thres)
                
                # If we have detections, write frame with bounding boxes
                if results["detections"]:
                    max_conf = max(d["confidence"] for d in results["detections"])
                    if max_conf >= conf_thres:
                        print(f"Frame {frame_idx}: Detected glottis with confidence {max_conf:.3f}")
                        good_frames.append({
                            "frame_idx": frame_idx,
                            "frame": frame,
                            "confidence": max_conf
                        })
                        frame_detections.append({
                            "frame_idx": frame_idx,
                            "detections": results["detections"]
                        })
                        success = out.write(results["annotated_image"])
                    else:
                        success = out.write(frame)
                else:
                    success = out.write(frame)
                    
                if not success:
                    print(f"Warning: Failed to write frame {frame_idx}")
                
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Processing: {progress:.1f}% complete")

            # Release resources
            cap.release()
            out.release()

            # Verify the output video
            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f"Video saved successfully. Size: {file_size} bytes")
                
                if file_size == 0:
                    raise Exception("Output video file is empty")
            else:
                raise Exception("Output video file was not created")

            # Convert to web-compatible format using FFmpeg
            try:
                import ffmpeg
                print("Converting video to web-compatible format...")
                
                temp_output = OUTPUT_DIR / f"temp_{output_filename}"
                stream = ffmpeg.input(str(save_path))
                stream = ffmpeg.output(
                    stream, 
                    str(temp_output),
                    vcodec='libx264',
                    acodec='aac',
                    **{'b:v': '2M'}  # 2 Mbps bitrate
                )
                ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
                
                # Replace original with converted file
                os.replace(temp_output, save_path)
                print("Video conversion completed successfully")
                
            except ImportError:
                print("FFmpeg Python bindings not available, skipping conversion")
            except Exception as e:
                print(f"FFmpeg conversion failed: {str(e)}")
                # If conversion fails, keep the original file

            print(f"\nProcessing complete:")
            print(f"Total frames processed: {frame_idx}")
            print(f"Frames with glottis detection: {len(good_frames)}")
            if good_frames:
                print(f"Average confidence on detected frames: {sum(f['confidence'] for f in good_frames)/len(good_frames):.3f}")

            return {
                "good_frames": good_frames,
                "frame_detections": frame_detections,
                "total_frames": total_frames,
                "fps": fps,
                "video_url": f"/output/{output_filename}"  # Return URL-friendly path
            }

        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            # Cleanup
            if 'cap' in locals():
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()
            raise e
        
        
    def process_video_detected_only(self, video_path, output_video_name, save_path=None, conf_thres=0.25, target_fps=None):
        """
        Process video and save only frames that contain glottis detections above confidence threshold.
        
        Args:
            video_path (str): Path to input video
            save_path (str): Optional path to save output video
            conf_thres (float): Confidence threshold for detections
            target_fps (int): Optional target fps for output video. If None, will calculate based on detection rate
            
        Returns:
            dict: Processing results including detected frames info and output path
        """
        try:
            # Create output directory
            OUTPUT_DIR = Path("output")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Generate output filename with timestamp 
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{output_video_name}_selected_frames_{timestamp}.mp4"
            save_path = OUTPUT_DIR / output_filename
            print(f"Will save output to: {save_path}")

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {video_path}")
                
            # Get video properties
            input_fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # First pass: count detections to estimate output fps if not specified
            if target_fps is None:
                print("Calculating appropriate output FPS based on detection rate...")
                detection_count = 0
                frame_indices = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    results = self.process_image(frame, conf_thres=conf_thres)
                    if results["detections"]:
                        max_conf = max(d["confidence"] for d in results["detections"])
                        if max_conf >= conf_thres:
                            detection_count += 1
                            frame_indices.append(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                
                # Calculate appropriate output fps
                if len(frame_indices) > 1:
                    avg_frame_gap = np.mean(np.diff(frame_indices))
                    detection_rate = detection_count / total_frames
                    # Scale fps based on detection rate to maintain approximate real-time playback
                    output_fps = int(input_fps * detection_rate)
                    # Ensure minimum reasonable fps
                    output_fps = max(output_fps, 5)
                else:
                    output_fps = input_fps
                    
                # Reset video capture for second pass
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                output_fps = target_fps

            print(f"Input FPS: {input_fps}")
            print(f"Output FPS: {output_fps}")

            # Try different codecs in order of preference
            codecs = ['avc1', 'mp4v', 'H264', 'XVID']
            out = None
            
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(
                        str(save_path),
                        fourcc,
                        output_fps,
                        (frame_width, frame_height),
                        True  # isColor
                    )
                    if out.isOpened():
                        print(f"Successfully initialized VideoWriter with codec: {codec}")
                        break
                except Exception as e:
                    print(f"Failed to initialize VideoWriter with codec {codec}: {e}")
                    if out is not None:
                        out.release()
                    continue
            
            if out is None or not out.isOpened():
                raise Exception("Failed to initialize video writer with any codec")

            detected_frames = []
            frame_detections = []
            frame_idx = 0
            frames_written = 0

            print(f"Starting video processing: {total_frames} frames total")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                results = self.process_image(frame, conf_thres=conf_thres)
                
                # Only save frames with valid detections above threshold
                if results["detections"]:
                    max_conf = max(d["confidence"] for d in results["detections"])
                    if max_conf >= conf_thres:
                        print(f"Frame {frame_idx}: Detected glottis with confidence {max_conf:.3f}")
                        detected_frames.append({
                            "frame_idx": frame_idx,
                            "confidence": max_conf,
                            "output_frame_idx": frames_written
                        })
                        frame_detections.append({
                            "frame_idx": frame_idx,
                            "detections": results["detections"]
                        })
                        # Write frame with bounding boxes
                        success = out.write(results["annotated_image"])
                        if not success:
                            print(f"Warning: Failed to write frame {frame_idx}")
                        else:
                            frames_written += 1
                
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Processing: {progress:.1f}% complete")

            # Release resources
            cap.release()
            out.release()

            # Verify the output video exists and is not empty
            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f"Video saved successfully. Size: {file_size} bytes")
                if file_size == 0:
                    raise Exception("Output video file is empty")
            else:
                raise Exception("Output video file was not created")

            # Try to convert to web-compatible format using FFmpeg
            try:
                import ffmpeg
                print("Converting video to web-compatible format...")
                
                temp_output = OUTPUT_DIR / f"temp_{output_filename}"
                stream = ffmpeg.input(str(save_path))
                stream = ffmpeg.output(
                    stream,
                    str(temp_output),
                    vcodec='libx264',
                    acodec='aac',
                    **{'b:v': '2M'}  # 2 Mbps bitrate
                )
                ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
                
                # Replace original with converted file
                os.replace(temp_output, save_path)
                print("Video conversion completed successfully")
                
            except ImportError:
                print("FFmpeg Python bindings not available, skipping conversion")
            except Exception as e:
                print(f"FFmpeg conversion failed: {str(e)}")
                # Keep original file if conversion fails

            print(f"\nProcessing complete:")
            print(f"Total frames processed: {frame_idx}")
            print(f"Frames with glottis detection: {len(detected_frames)}")
            if detected_frames:
                avg_conf = sum(f['confidence'] for f in detected_frames)/len(detected_frames)
                print(f"Average confidence on detected frames: {avg_conf:.3f}")

            return {
                "detected_frames": detected_frames,
                "frame_detections": frame_detections,
                "total_frames": frame_idx,
                "detected_frame_count": frames_written,
                "input_fps": input_fps,
                "output_fps": output_fps,
                "video_dimensions": (frame_width, frame_height),
                "video_url": f"/output/{output_filename}"
            }

        except Exception as e:
            print(f"Error in process_video_detected_only: {str(e)}")
            # Cleanup
            if 'cap' in locals():
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()
            raise e
        
    def process_video_detected_only_raw(self, video_path, output_video_name, output_dir, save_path=None, conf_thres=0.25, target_fps=None, force_reprocess=False):
        """
        Process video and save only raw frames (without annotations) that contain glottis detections above confidence threshold.
        
        Args:
            video_path (str): Path to input video
            output_video_name (str): Base name for output video file
            output_dir (str): Directory to save output video
            save_path (str): Optional path to save output video
            conf_thres (float): Confidence threshold for detections
            target_fps (int): Optional target fps for output video. If None, will calculate based on detection rate
            force_reprocess (bool): If True, will reprocess even if output file exists
            
        Returns:
            dict: Processing results including detected frames info and output path
        """
        try:
            # Create output directory
            OUTPUT_DIR = Path(output_dir)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # Generate output filename with timestamp 
            output_filename = f"{output_video_name}_processed.mp4"
            save_path = OUTPUT_DIR / output_filename
            
            # Check if output file already exists and skip processing if it does
            if save_path.exists() and not force_reprocess:
                print(f"Output file already exists: {save_path}")
                print("Skipping processing. Use force_reprocess=True to override.")
                return {
                    "detected_frames": [],
                    "frame_detections": [],
                    "total_frames": 0,
                    "detected_frame_count": 0,
                    "input_fps": 0,
                    "output_fps": 0,
                    "video_dimensions": (0, 0),
                    "video_url": f"/output/{output_filename}",
                    "status": "skipped_existing_file"
                }

            print(f"Will save output to: {save_path}")

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {video_path}")
                
            # Get video properties
            input_fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # First pass: count detections to estimate output fps if not specified
            if target_fps is None:
                print("Calculating appropriate output FPS based on detection rate...")
                detection_count = 0
                frame_indices = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    results = self.process_image(frame, conf_thres=conf_thres)
                    if results["detections"]:
                        max_conf = max(d["confidence"] for d in results["detections"])
                        if max_conf >= conf_thres:
                            detection_count += 1
                            frame_indices.append(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
                
                # Calculate appropriate output fps
                if len(frame_indices) > 1:
                    avg_frame_gap = np.mean(np.diff(frame_indices))
                    detection_rate = detection_count / total_frames
                    output_fps = int(input_fps * detection_rate)
                    output_fps = max(output_fps, 5)  # Ensure minimum reasonable fps
                else:
                    output_fps = input_fps
                    
                # Reset video capture for second pass
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                output_fps = target_fps

            print(f"Input FPS: {input_fps}")
            print(f"Output FPS: {output_fps}")

            # Try different codecs in order of preference
            codecs = ['avc1', 'mp4v', 'H264', 'XVID']
            out = None
            
            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(
                        str(save_path),
                        fourcc,
                        output_fps,
                        (frame_width, frame_height),
                        True  # isColor
                    )
                    if out.isOpened():
                        print(f"Successfully initialized VideoWriter with codec: {codec}")
                        break
                except Exception as e:
                    print(f"Failed to initialize VideoWriter with codec {codec}: {e}")
                    if out is not None:
                        out.release()
                    continue
            
            if out is None or not out.isOpened():
                raise Exception("Failed to initialize video writer with any codec")

            detected_frames = []
            frame_detections = []
            frame_idx = 0
            frames_written = 0

            print(f"Starting video processing: {total_frames} frames total")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame for detection but save raw frame
                results = self.process_image(frame, conf_thres=conf_thres)
                
                # Only save frames with valid detections above threshold
                if results["detections"]:
                    max_conf = max(d["confidence"] for d in results["detections"])
                    if max_conf >= conf_thres:
                        print(f"Frame {frame_idx}: Detected glottis with confidence {max_conf:.3f}")
                        detected_frames.append({
                            "frame_idx": frame_idx,
                            "confidence": max_conf,
                            "output_frame_idx": frames_written
                        })
                        frame_detections.append({
                            "frame_idx": frame_idx,
                            "detections": results["detections"]
                        })
                        # Write raw frame without bounding boxes
                        success = out.write(frame)
                        if not success:
                            print(f"Warning: Failed to write frame {frame_idx}")
                        else:
                            frames_written += 1
                
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Processing: {progress:.1f}% complete")

            # Release resources
            cap.release()
            out.release()

            # Verify the output video exists and is not empty
            if save_path.exists():
                file_size = save_path.stat().st_size
                print(f"Video saved successfully. Size: {file_size} bytes")
                if file_size == 0:
                    raise Exception("Output video file is empty")
            else:
                raise Exception("Output video file was not created")

            # Try to convert to web-compatible format using FFmpeg
            try:
                import ffmpeg
                print("Converting video to web-compatible format...")
                
                temp_output = OUTPUT_DIR / f"temp_{output_filename}"
                stream = ffmpeg.input(str(save_path))
                stream = ffmpeg.output(
                    stream,
                    str(temp_output),
                    vcodec='libx264',
                    acodec='aac',
                    **{'b:v': '2M'}  # 2 Mbps bitrate
                )
                ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
                
                # Replace original with converted file
                os.replace(temp_output, save_path)
                print("Video conversion completed successfully")
                
            except ImportError:
                print("FFmpeg Python bindings not available, skipping conversion")
            except Exception as e:
                print(f"FFmpeg conversion failed: {str(e)}")

            print(f"\nProcessing complete:")
            print(f"Total frames processed: {frame_idx}")
            print(f"Frames with glottis detection: {len(detected_frames)}")
            if detected_frames:
                avg_conf = sum(f['confidence'] for f in detected_frames)/len(detected_frames)
                print(f"Average confidence on detected frames: {avg_conf:.3f}")

            return {
                "detected_frames": detected_frames,
                "frame_detections": frame_detections,
                "total_frames": frame_idx,
                "detected_frame_count": frames_written,
                "input_fps": input_fps,
                "output_fps": output_fps,
                "video_dimensions": (frame_width, frame_height),
                "video_url": f"/output/{output_filename}",
                "status": "processed_successfully"
            }

        except Exception as e:
            print(f"Error in process_video_detected_only_raw: {str(e)}")
            # Cleanup
            if 'cap' in locals():
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()
            raise e