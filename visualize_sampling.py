import matplotlib.pyplot as plt
import numpy as np
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.encoded_video import EncodedVideo
import torch
from pathlib import Path

def plot_all_sampling_methods(video_path, num_frames=32, fps=30, stride=0.5, figsize=(24, 10)):
    """
    Compare different sampling methods in a single figure.
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to sample
        fps (int): Frames per second
        stride (float): Stride fraction for sliding window sampling
        figsize (tuple): Figure size for the plot
    """
    sampling_methods = ['uniform', 'random', 'sliding']
    
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    
    # First row: sampling pattern visualization
    pattern_axes = [plt.subplot(2, 3, i+1) for i in range(3)]
    
    # Second row: actual frame samples
    frame_axes = [plt.subplot(2, 3, i+4) for i in range(3)]
    
    # Load video
    video = EncodedVideo.from_path(video_path)
    duration = float(video.duration)
    clip_duration = float(num_frames) / float(fps)
    
    # Plot for each sampling method
    for idx, method in enumerate(sampling_methods):
        # Create clip sampler
        if method == 'sliding':
            stride_time = float(clip_duration * stride)
            clip_sampler = make_clip_sampler('uniform', clip_duration, stride_time)
        else:
            clip_sampler = make_clip_sampler(method, clip_duration)
        
        # Get all clips
        clip_start_sec = 0.0
        clips = []
        
        while True:
            clip_info = clip_sampler(clip_start_sec, duration, None)
            if clip_info is None:
                break
                
            start_sec = float(clip_info.clip_start_sec)
            end_sec = float(clip_info.clip_end_sec)
            clips.append((start_sec, end_sec))
            
            clip_start_sec = end_sec
            
            if clip_info.is_last_clip:
                break
        
        # Plot sampling pattern
        ax = pattern_axes[idx]
        
        # Plot timeline
        ax.plot([0, duration], [0, 0], 'k-', linewidth=2)
        ax.plot([0, 0], [-0.2, 0.2], 'k-', linewidth=2)
        ax.plot([duration, duration], [-0.2, 0.2], 'k-', linewidth=2)
        
        # Plot clips with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clips)))
        for i, ((start, end), color) in enumerate(zip(clips, colors)):
            # Plot clip region
            ax.axvspan(start, end, alpha=0.3, color=color)
            ax.plot([start, start], [-0.2, 0.2], '--', color=color)
            ax.plot([end, end], [-0.2, 0.2], '--', color=color)
            ax.text((start + end)/2, 0.1, f'Clip {i+1}', 
                   horizontalalignment='center', color=color)
        
        # Customize plot
        title = f'{method.title()} Sampling'
        if method == 'sliding':
            title += f'\nOverlap: {(1-stride)*100:.0f}%'
        ax.set_title(title)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
        # Plot sample frames from first clip
        try:
            # Extract first clip
            video_data = video.get_clip(clips[0][0], clips[0][1])
            
            # Get frames and normalize
            frames = video_data["video"].permute(1, 2, 3, 0).numpy()  # T, H, W, C
            frames = frames.astype(np.float32) / 255.0
            
            # Select 8 evenly spaced frames
            num_sample_frames = 8
            frame_indices = np.linspace(0, len(frames)-1, num_sample_frames, dtype=int)
            selected_frames = frames[frame_indices]
            
            # Create sub-subplots for frames
            ax = frame_axes[idx]
            for i, frame in enumerate(selected_frames):
                if i == 0:
                    # For first frame, create a new axis that takes up 1/4 of the space
                    sub_ax = ax.inset_axes([i/num_sample_frames, 0, 1/num_sample_frames, 1])
                else:
                    # For subsequent frames, create axes next to the previous one
                    sub_ax = ax.inset_axes([i/num_sample_frames, 0, 1/num_sample_frames, 1])
                
                sub_ax.imshow(frame)
                sub_ax.axis('off')
                sub_ax.set_title(f'Frame {frame_indices[i]}')
            
            ax.set_title(f'Sample Frames from Clip 1\n({clips[0][0]:.1f}s - {clips[0][1]:.1f}s)')
            ax.axis('off')
            
        except Exception as e:
            print(f"Error plotting frames for {method} sampling: {str(e)}")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Replace with your video path
    video_path = "artifacts/laryngeal_dataset_balanced:v0/dataset/test/non_referral/0023.mp4"
    
    # Create visualization
    fig = plot_all_sampling_methods(video_path)
    
    # Save figure
    fig.savefig('sampling_methods_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Sampling methods comparison saved as 'sampling_methods_comparison.png'")
