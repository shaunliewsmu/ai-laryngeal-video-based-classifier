import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    """Plot confusion matrix using seaborn with string labels"""
    # Convert numeric predictions to string labels
    y_pred_labels = ['non_referral' if pred < 0.5 else 'referral' for pred in y_pred]
    y_true_labels = ['non_referral' if label == 0 else 'referral' for label in y_true]
    # Create confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non_referral', 'referral'],
                yticklabels=['non_referral', 'referral'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_auroc_curve(epochs, train_aurocs, val_aurocs, save_path):
    """Plot training and validation AUROC curves"""
    # Get actual number of epochs trained
    actual_epochs = len(train_aurocs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_aurocs, label='Train AUROC', marker='o')
    plt.plot(range(1, actual_epochs + 1), val_aurocs, label='Validation AUROC', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_sampled_frames(video_path, clip_info, save_path):
    """Plot and save frames from a sampled clip"""
    from pytorchvideo.data.encoded_video import EncodedVideo
    
    # Load video
    video = EncodedVideo.from_path(video_path)
    
    # Get clip
    video_data = video.get_clip(clip_info.clip_start_sec, clip_info.clip_end_sec)
    frames = video_data["video"]
    
    # Convert from tensor [C, T, H, W] to numpy [T, H, W, C]
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    
    # Create a figure with subplots
    n_cols = 8
    n_rows = (frames.shape[0] + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    # Plot each frame
    for i, frame in enumerate(frames):
        if i < len(axes):
            axes[i].imshow(frame)
            axes[i].axis('off')
            axes[i].set_title(f'Frame {i}')
    
    # Turn off remaining empty subplots
    for i in range(frames.shape[0], len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_frames(frames, frame_indices, save_path, title='Sampled Frames'):
    """Plot and save frames with their original indices from the video
    
    Args:
        frames (numpy.ndarray): Frames in [T, H, W, C] format
        frame_indices (list): Original frame indices from the video
        save_path (str): Path to save the visualization
        title (str): Title for the plot
    """
    # Create a figure with subplots
    n_cols = 8
    n_rows = (len(frames) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    # Plot each frame
    for i, (frame, original_idx) in enumerate(zip(frames, frame_indices)):
        if i < len(axes):
            # Normalize to [0,1] range to avoid the warning
            if frame.max() > 1.0:
                frame = frame / 255.0
            
            axes[i].imshow(frame)
            axes[i].axis('off')
            axes[i].set_title(f'Frame {original_idx}')
    
    # Turn off remaining empty subplots
    for i in range(len(frames), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_clip_visualization(video_path, clip_info, save_path, title, fps=30):
    """Plot and visualize a clip from the video
    
    Args:
        video_path (str): Path to the video
        clip_info (ClipInfo): Information about the clip
        save_path (str): Where to save the visualization
        title (str): Title for the plot
        fps (int): Frames per second
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices from clip_info
    start_frame = int(clip_info.clip_start_sec * fps)
    end_frame = int(clip_info.clip_end_sec * fps)
    
    # Create a visualization showing where the clip is positioned in the video
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(0, total_frames, color='lightgray')
    ax.barh(0, end_frame - start_frame, left=start_frame, color='green')
    
    ax.set_xlim(0, total_frames)
    ax.set_yticks([])
    ax.set_xlabel('Frame Number')
    ax.text(start_frame, 0, f"Start: {start_frame}", 
            horizontalalignment='right', verticalalignment='center')
    ax.text(end_frame, 0, f"End: {end_frame}", 
            horizontalalignment='left', verticalalignment='center')
    
    plt.suptitle(f"{title}\nSampled {end_frame-start_frame} frames from positions {start_frame}-{end_frame}")
    plt.tight_layout()
    
    # Save the position visualization
    position_save_path = save_path.replace('.png', '_position.png')
    plt.savefig(position_save_path)
    plt.close()
    
    # Now extract and show some frames from the clip
    frames = []
    indices = np.linspace(start_frame, end_frame-1, min(11, end_frame-start_frame), dtype=int)
    
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append((frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    
    cap.release()
    
    # Plot the sampled frames
    if frames:
        n_cols = 4
        n_rows = (len(frames) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten()
        
        for i, (frame_idx, frame) in enumerate(frames):
            if i < len(axes):
                if frame.max() > 1.0:  # Normalize to avoid warnings
                    frame = frame / 255.0
                    
                axes[i].imshow(frame)
                axes[i].axis('off')
                axes[i].set_title(f'Frame {frame_idx}')
        
        # Turn off remaining empty subplots
        for i in range(len(frames), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()