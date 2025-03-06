import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
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
        ax1.plot(history['train_loss'], label='Training Loss', color='#FF5733')
        ax1.plot(history['val_loss'], label='Validation Loss', color='#33A2FF')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add early stopping and best model markers if epochs > 1
        if len(history['val_loss']) > 1:
            best_epoch = np.argmin(history['val_loss'])
            ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, 
                       label=f'Best model (epoch {best_epoch+1})')
            ax1.plot(best_epoch, history['val_loss'][best_epoch], 'go', markersize=8)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy', color='#FF5733')
        ax2.plot(history['val_acc'], label='Validation Accuracy', color='#33A2FF')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add best accuracy marker if epochs > 1
        if len(history['val_acc']) > 1:
            best_acc_epoch = np.argmax(history['val_acc'])
            ax2.axvline(x=best_acc_epoch, color='green', linestyle='--', alpha=0.5,
                       label=f'Best accuracy (epoch {best_acc_epoch+1})')
            ax2.plot(best_acc_epoch, history['val_acc'][best_acc_epoch], 'go', markersize=8)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=150)
        plt.close()
        
    def plot_confusion_matrix(self, conf_matrix, class_names=None):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix (np.ndarray or list): Confusion matrix
            class_names (list): List of class names
        """
        # Convert to numpy array if it's a list
        if isinstance(conf_matrix, list):
            conf_matrix = np.array(conf_matrix)
            
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages for annotations
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Replace NaN with 0
        
        # Create the heatmap
        ax = sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5
        )
        
        # Add percentage annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                if conf_matrix[i, j] > 0:
                    ax.text(j + 0.5, i + 0.7, f'({conf_matrix_norm[i, j]:.1%})', 
                        ha='center', va='center', color='black', fontsize=9)
        
        # Calculate and display metrics
        if conf_matrix.shape == (2, 2):  # Binary classification
            try:
                tn, fp, fn, tp = conf_matrix.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                plt.figtext(0.5, 0.01, 
                        f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}", 
                        ha="center", fontsize=10, 
                        bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            except Exception as e:
                print(f"Error calculating metrics: {e}")
        
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
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
        
        # Create a figure with a row for each sample
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4*num_samples))
        fig.suptitle('Sample Predictions', fontsize=16, y=0.98)
        
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            try:
                # Get multiple frames from the video clip to show temporal aspect
                # Expecting shape (C, T, H, W)
                video = images[i]
                T = video.shape[1]
                
                # Choose frames to display (start, middle, end)
                frame_indices = [0, T//4, T//2, 3*T//4, T-1]
                frame_indices = [idx for idx in frame_indices if idx < T]
                
                # Create a grid of frames
                n_frames = len(frame_indices)
                for j, frame_idx in enumerate(frame_indices):
                    # Get the frame
                    frame = video[:, frame_idx, :, :]  # Shape: (C, H, W)
                    
                    # Convert to numpy and rescale to [0,1]
                    frame = frame.cpu().numpy()
                    frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                    
                    # Transpose from (C,H,W) to (H,W,C) for matplotlib
                    frame = frame.transpose(1, 2, 0)
                    
                    # Create subplot within the sample's row
                    sub_ax = axes[i].inset_axes([j/n_frames, 0, 1/n_frames, 1])
                    sub_ax.imshow(frame)
                    sub_ax.axis('off')
                    sub_ax.set_title(f'Frame {frame_idx}', fontsize=9)
                
                # Get labels
                true_label = class_names[true_labels[i]] if class_names else true_labels[i]
                pred_label = class_names[pred_labels[i]] if class_names else pred_labels[i]
                
                # Set title for the row
                title = f'Video {i+1}: True: {true_label} | Predicted: {pred_label}'
                color = 'green' if true_labels[i] == pred_labels[i] else 'red'
                axes[i].set_title(title, color=color, fontsize=12)
                axes[i].axis('off')
                
            except Exception as e:
                print(f"Error plotting sample {i}: {str(e)}")
                axes[i].text(0.5, 0.5, f"Error displaying sample: {str(e)}", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[i].transAxes, color='red')
                axes[i].axis('off')
                continue
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'sample_predictions.png', dpi=150, bbox_inches='tight')
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