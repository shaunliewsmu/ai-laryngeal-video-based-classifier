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