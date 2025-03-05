import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import cv2
import random

class TrainingVisualizer:
    """Class for creating visualizations of training metrics and results."""
    
    def __init__(self, save_dir):
        """
        Initialize the visualizer.
        
        Args:
            save_dir (str): Directory to save visualization plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_training_history(self, history):
        """
        Plot training and validation metrics over epochs.
        
        Args:
            history (dict): Dictionary containing training history
                Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()
        
    def plot_confusion_matrix(self, conf_matrix, class_names=None):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix (np.ndarray): Confusion matrix
            class_names (list): List of class names
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
    def plot_sample_predictions(self, images, true_labels, pred_labels, class_names=None):
        """
        Plot sample video frames with their true and predicted labels.
        
        Args:
            images (torch.Tensor): Batch of video frames (B, C, T, H, W)
            true_labels (list): True labels
            pred_labels (list): Predicted labels
            class_names (list): List of class names
        """
        num_samples = min(len(images), 5)  # Show up to 5 samples
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4*num_samples))
        
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            try:
                # Get middle frame from the video clip
                # Expecting shape (C, T, H, W)
                video = images[i]
                middle_frame_idx = video.shape[1] // 2
                middle_frame = video[:, middle_frame_idx, :, :]  # Shape: (C, H, W)
                
                # Convert to numpy and rescale to [0,1]
                middle_frame = middle_frame.cpu().numpy()
                middle_frame = (middle_frame - middle_frame.min()) / (middle_frame.max() - middle_frame.min())
                
                # Transpose from (C,H,W) to (H,W,C) for matplotlib
                middle_frame = middle_frame.transpose(1, 2, 0)
                
                axes[i].imshow(middle_frame)
                
                true_label = class_names[true_labels[i]] if class_names else true_labels[i]
                pred_label = class_names[pred_labels[i]] if class_names else pred_labels[i]
                
                title = f'True: {true_label} | Predicted: {pred_label}'
                color = 'green' if true_labels[i] == pred_labels[i] else 'red'
                axes[i].set_title(title, color=color)
                axes[i].axis('off')
                
            except Exception as e:
                print(f"Error plotting sample {i}: {str(e)}")
                continue
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'sample_predictions.png')
        plt.close()
    
    def visualize_sampling(self, video_path, sampling_method, num_frames, save_path, title=''):
        """
        Visualize frame sampling pattern on a video with dynamic FPS adjustment for short videos.
        
        Args:
            video_path (str): Path to the video file
            sampling_method (str): 'uniform', 'random', or 'random_window'
            num_frames (int): Number of frames to sample
            save_path (str): Path to save the visualization
            title (str): Title for the visualization
        """
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / original_fps  # in seconds
        
        # Check if we need dynamic FPS adjustment
        dynamic_fps = False
        fps_to_use = 30  # Default FPS
        
        if total_frames < num_frames:
            dynamic_fps = True
            # Calculate optimal FPS to get exactly num_frames
            fps_to_use = num_frames / video_duration
            method_name = f"{sampling_method.replace('_', ' ').title()} (Dynamic FPS: {fps_to_use:.2f})"
        else:
            method_name = f"{sampling_method.replace('_', ' ').title()}"
        
        # Sample frames using specified method
        if sampling_method == 'random':
            if total_frames >= num_frames:
                # Normal random sampling
                indices = sorted(random.sample(range(total_frames), num_frames))
            else:
                # Random sampling with replacement
                indices = sorted(random.choices(range(total_frames), k=num_frames))
        elif sampling_method == 'random_window':
            # Random window sampling (works with both normal and dynamic FPS)
            window_size = total_frames / num_frames
            indices = []
            for i in range(num_frames):
                virtual_start = i * window_size
                virtual_end = (i + 1) * window_size
                
                # Convert to actual frame indices
                if total_frames >= num_frames:
                    start = int(virtual_start)
                    end = min(int(virtual_end), total_frames)
                    end = max(end, start + 1)  # Ensure window has at least 1 frame
                    frame_idx = random.randint(start, end - 1)
                else:
                    # For shorter videos, we might have virtual windows smaller than 1 frame
                    actual_index = min(int(np.floor(virtual_start + (virtual_end - virtual_start) * random.random())), 
                                      total_frames - 1)
                    frame_idx = actual_index
                
                indices.append(frame_idx)
        else:  # uniform
            # Uniform sampling (works with both normal and dynamic FPS)
            if num_frames == 1:
                indices = [total_frames // 2]  # Middle frame
            else:
                step = total_frames / num_frames
                indices = [min(int(i * step), total_frames - 1) for i in range(num_frames)]
        
        # Create figure with 2 rows
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                       gridspec_kw={'height_ratios': [1, 3]})
        
        # Top row: sampling pattern visualization
        ax1.plot([0, total_frames], [0, 0], 'k-', linewidth=2)
        ax1.plot([0, 0], [-0.2, 0.2], 'k-', linewidth=2)
        ax1.plot([total_frames, total_frames], [-0.2, 0.2], 'k-', linewidth=2)
        
        # Mark windows for Random Window Sampling method
        if sampling_method == 'random_window':
            window_size = total_frames / num_frames
            for i in range(num_frames):
                start = int(i * window_size)
                end = int((i + 1) * window_size) if i < num_frames - 1 else total_frames
                ax1.axvspan(start, end, alpha=0.1, color='gray')
                ax1.plot([start, start], [-0.2, 0.2], 'k--', alpha=0.5)
        
        # Mark sampled frames
        for i, frame_idx in enumerate(indices):
            color = plt.cm.rainbow(i / len(indices))
            ax1.plot([frame_idx, frame_idx], [-0.3, 0.3], '-', color=color, linewidth=2)
            ax1.plot(frame_idx, 0, 'o', color=color, markersize=8)
            ax1.text(frame_idx, 0.1, f'{frame_idx}', 
                   horizontalalignment='center', color=color, fontsize=9)
        
        # Customize plot
        frames_info = f"{len(indices)} frames from {total_frames} total"
        if dynamic_fps:
            frames_info += f" (Dynamic FPS: {fps_to_use:.2f})"
            
        full_title = f"{title} - {method_name}\n{frames_info}"
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
        cols = min(8, len(indices))
        rows = (len(indices) + cols - 1) // cols
        
        # Extract frames and display in grid
        for i, frame_idx in enumerate(indices):
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