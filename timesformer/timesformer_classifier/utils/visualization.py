import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import random
from pathlib import Path

class TrainingVisualizer:
    """Class for creating visualizations of training metrics and results."""
    
    def __init__(self, save_dir):
        """
        Initialize the visualizer.
        
        Args:
            save_dir (str or Path): Directory to save visualization plots
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
        
    def plot_confusion_matrix(self, conf_matrix, class_names=None, title="Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix (np.ndarray): Confusion matrix
            class_names (list): List of class names
            title (str): Title for the plot
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
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Create filename from title
        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.savefig(self.save_dir / f'{filename}.png')
        plt.close()
        
    def plot_roc_curve(self, y_true, y_score, class_names=None, title="ROC Curve"):
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_score (np.ndarray): Predicted probabilities
            class_names (list): List of class names
            title (str): Title for the plot
        """
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(8, 6))
        
        # Binary classification
        if len(class_names) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
        # Multi-class classification
        else:
            from sklearn.preprocessing import label_binarize
            
            # Binarize labels
            classes = np.arange(len(class_names))
            y_bin = label_binarize(y_true, classes=classes)
            
            # Compute ROC curve and ROC area for each class
            fpr = {}
            tpr = {}
            roc_auc = {}
            
            for i, class_name in enumerate(class_names):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(
                    fpr[i], tpr[i], lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})'
                )
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Create filename from title
        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.savefig(self.save_dir / f'{filename}.png')
        plt.close()
        
    def visualize_sampling(self, video_path, sampling_method, num_frames, save_path, title=''):
        """
        Visualize frame sampling pattern on a video.
        
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
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / original_fps  # in seconds
        
        # Sample frames using specified method
        if sampling_method == 'random':
            if total_frames >= num_frames:
                # Normal random sampling
                indices = sorted(random.sample(range(total_frames), num_frames))
            else:
                # Random sampling with replacement
                indices = sorted(random.choices(range(total_frames), k=num_frames))
        elif sampling_method == 'random_window':
            # random_window sampling
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
            # Uniform sampling
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
        
        # Mark windows for random_window sampling method
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
        if total_frames < num_frames:
            frames_info += f" (Short video - frame duplication applied)"
            
        full_title = f"{title} - {sampling_method.capitalize()} Sampling\n{frames_info}"
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