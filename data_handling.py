import os
import numpy as np
import av
from pathlib import Path


def generate_all_files(root: Path, only_files: bool = True):
    for p in root.rglob("*"):
        if only_files and not p.is_file():
            continue
        yield p
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    
    print(f"Reading frames from {start_index} to {end_index}")
    
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            reformatted_frame = frame.reformat(width=224, height=224)
            frames.append(reformatted_frame)
            
    print(f"Number of frames extracted: {len(frames)}")
    
    if len(frames) == 0:
        raise ValueError("No frames were extracted!")
        
    new = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    print(f"Final video array shape: {new.shape}")
    
    return new


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def frames_convert_and_create_dataset_dictionary(directory, number_of_frames=10):
    class_labels = []
    all_videos = []
    sizes = []
    
    # Convert directory to Path object
    dir_path = Path(directory)
    
    # Walk through dataset directory structure 
    for split in ['train', 'test', 'val']:
        split_path = dir_path / 'dataset' / split
        if not split_path.exists():
            continue
            
        # Iterate through class folders
        for class_path in split_path.iterdir():
            if not class_path.is_dir():
                continue
                
            cls = class_path.name
            if cls not in class_labels:
                class_labels.append(cls)
            
            # Process video files in class folder
            for video_file in class_path.glob('*.mp4'):
                try:
                    # Open video file
                    container = av.open(str(video_file))
                    num_frames = container.streams.video[0].frames
                    print(f"Processing file {video_file} number of Frames: {num_frames}")
                    
                    # Sample frames
                    indices = sample_frame_indices(clip_len=number_of_frames, 
                                                frame_sample_rate=1,
                                                seg_len=num_frames)
                    video = read_video_pyav(container=container, indices=indices)
                    
                    # Store video data
                    all_videos.append({
                        'video': video,
                        'labels': cls,
                        'split': split,
                        'path': str(video_file)
                    })
                    sizes.append(num_frames)
                    
                except Exception as e:
                    print(f"Error processing {video_file}: {str(e)}")
                finally:
                    container.close()
                    
    sizes = np.array(sizes)
    print(f"Min number frames {sizes.min()}")
    
    return all_videos, class_labels
