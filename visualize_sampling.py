import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import cv2
from pathlib import Path

def get_video_properties(video_path):
    """
    Get video properties using OpenCV with verification of actual frame count
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (total_frames, fps, duration_sec, width, height)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get reported frame count and fps
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Verify actual frame count by seeking to the last frame
    # This is more reliable than using CAP_PROP_FRAME_COUNT
    # Some codecs report incorrect frame counts
    actual_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_frames += 1
    
    print(f"Reported frames: {reported_frames}, Actual frames: {actual_frames}")
    total_frames = actual_frames
    duration_sec = total_frames / fps
    
    cap.release()
    return total_frames, fps, duration_sec, width, height

def get_frame_from_video(video_path, frame_idx):
    """
    Extract a specific frame from a video with fallback mechanism
    
    Args:
        video_path (str): Path to the video file
        frame_idx (int): Index of the frame to extract
        
    Returns:
        numpy.ndarray: The extracted frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Try setting position by frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    # If frame reading failed, try a safer approach
    if not ret:
        print(f"Warning: Could not directly access frame {frame_idx}, using sequential reading")
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        # Read frames sequentially until we reach our target frame
        current_frame = 0
        while current_frame < frame_idx:
            ret = cap.grab()  # Faster than cap.read() as we don't decode the frame
            if not ret:
                break
            current_frame += 1
        
        # Now read the actual frame we want
        ret, frame = cap.read()
    
    cap.release()
    
    if not ret:
        # If we still can't get the frame, use the last available frame or error
        # Reopen the video to get the last available frame
        cap = cv2.VideoCapture(video_path)
        last_frame = None
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            last_frame = current_frame
        cap.release()
        
        if last_frame is not None:
            print(f"Warning: Using last available frame instead of frame {frame_idx}")
            frame = last_frame
        else:
            raise ValueError(f"Could not read any frames from the video")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def random_sampling(total_frames, num_frames):
    """
    Randomly sample frames from the entire video
    
    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        
    Returns:
        list: Indices of sampled frames
    """
    # Ensure num_frames doesn't exceed total_frames
    num_frames = min(num_frames, total_frames)
    
    # Random sampling without replacement
    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    return frame_indices

def uniform_sampling(total_frames, num_frames):
    """
    Sample frames at regular intervals across the video
    
    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        
    Returns:
        list: Indices of sampled frames
    """
    # Ensure num_frames doesn't exceed total_frames
    num_frames = min(num_frames, total_frames)
    
    if num_frames == 1:
        return [total_frames // 2]  # Middle frame
    
    # Calculate step size - use (total_frames - 1) to ensure we don't go beyond last available frame
    step = (total_frames - 1) / (num_frames - 1)
    
    # Generate evenly spaced indices, making sure the last index doesn't exceed total_frames - 1
    frame_indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
    return frame_indices

def random_window_sampling(total_frames, num_frames):
    """
    Divide the video into equal windows and randomly sample one frame from each window
    
    Args:
        total_frames (int): Total number of frames in the video
        num_frames (int): Number of frames to sample
        
    Returns:
        list: Indices of sampled frames
    """
    # Ensure num_frames doesn't exceed total_frames
    num_frames = min(num_frames, total_frames)
    
    # Calculate window size
    window_size = total_frames / num_frames
    
    frame_indices = []
    for i in range(num_frames):
        # Calculate window boundaries, ensuring we don't exceed total_frames
        start = int(i * window_size)
        end = min(int((i + 1) * window_size), total_frames)
        
        # Ensure end is at least start+1 to avoid empty windows
        end = max(end, start + 1)
        
        # Randomly select a frame from this window
        frame_idx = random.randint(start, end - 1)
        
        frame_indices.append(frame_idx)
    
    return frame_indices

def visualize_sampling_methods(video_path, num_frames=8, figsize=(16, 12)):
    """
    Visualize different sampling methods on a video
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to sample
        figsize (tuple): Figure size for the plot
    """
    # Get video properties
    total_frames, fps, duration_sec, width, height = get_video_properties(video_path)
    print(f"Video has {total_frames} frames, duration: {duration_sec:.2f}s, FPS: {fps}")
    
    # Define sampling methods
    sampling_methods = {
        'Random Sampling': random_sampling,
        'Uniform Sampling': uniform_sampling,
        'Random Window Sampling': random_window_sampling
    }
    
    # Generate frame indices for each method
    sampled_indices = {}
    for method_name, method_func in sampling_methods.items():
        sampled_indices[method_name] = method_func(total_frames, num_frames)
        print(f"{method_name}: {sampled_indices[method_name]}")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # First row: sampling pattern visualization
    pattern_axes = [plt.subplot(2, 3, i+1) for i in range(3)]
    
    # Second row: actual frame samples
    frame_axes = [plt.subplot(2, 3, i+4) for i in range(3)]
    
    # Visualize sampling patterns
    for idx, (method_name, indices) in enumerate(sampled_indices.items()):
        ax = pattern_axes[idx]
        
        # Plot timeline
        ax.plot([0, total_frames], [0, 0], 'k-', linewidth=2)
        ax.plot([0, 0], [-0.2, 0.2], 'k-', linewidth=2)
        ax.plot([total_frames, total_frames], [-0.2, 0.2], 'k-', linewidth=2)
        
        # Mark windows for Random Window Sampling method
        if method_name == 'Random Window Sampling':
            window_size = total_frames / num_frames
            for i in range(num_frames):
                start = int(i * window_size)
                end = int((i + 1) * window_size) if i < num_frames - 1 else total_frames
                ax.axvspan(start, end, alpha=0.1, color='gray')
                ax.plot([start, start], [-0.2, 0.2], 'k--', alpha=0.5)
        
        # Mark sampled frames
        for i, frame_idx in enumerate(indices):
            color = plt.cm.rainbow(i / len(indices))
            ax.plot([frame_idx, frame_idx], [-0.3, 0.3], '-', color=color, linewidth=2)
            ax.plot(frame_idx, 0, 'o', color=color, markersize=8)
            ax.text(frame_idx, 0.1, f'{frame_idx}', 
                   horizontalalignment='center', color=color, fontsize=9)
        
        # Customize plot
        ax.set_title(f'{method_name}\n{len(indices)} frames')
        ax.set_xlabel('Frame Index')
        ax.set_xlim(-total_frames*0.05, total_frames*1.05)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        
        # Also display time for some equally spaced frames
        time_marks = 5
        for i in range(time_marks + 1):
            frame = int(i * total_frames / time_marks)
            time = frame / fps
            ax.text(frame, -0.2, f'{time:.1f}s', 
                   horizontalalignment='center', color='black', fontsize=8)
        
        # Display the sampled frames
        try:
            ax = frame_axes[idx]
            
            # Create a grid for the frames
            cols = min(4, len(indices))
            rows = (len(indices) + cols - 1) // cols
            
            for i, frame_idx in enumerate(indices):
                frame = get_frame_from_video(video_path, frame_idx)
                
                # Calculate position in grid
                row, col = i // cols, i % cols
                
                # Create subplot
                sub_ax = ax.inset_axes([col/cols, 1-(row+1)/rows, 1/cols, 1/rows])
                
                # Display frame
                sub_ax.imshow(frame)
                sub_ax.axis('off')
                sub_ax.set_title(f'Frame {frame_idx}\n({frame_idx/fps:.2f}s)', fontsize=9)
            
            ax.set_title(f'Frames sampled using {method_name}')
            ax.axis('off')
            
        except Exception as e:
            print(f"Error plotting frames for {method_name}: {str(e)}")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Use the same path as in your original script
    video_path = "artifacts/laryngeal_dataset_balanced:v0/dataset/test/non_referral/0023.mp4"
    
    # Number of frames to sample - using 32 frames as in your error message
    num_frames = 32
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create visualization
    fig = visualize_sampling_methods(video_path, num_frames)
    
    # Save figure
    fig.savefig('custom_sampling_methods_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Sampling methods comparison saved as 'custom_sampling_methods_comparison.png'")