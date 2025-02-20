import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch

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