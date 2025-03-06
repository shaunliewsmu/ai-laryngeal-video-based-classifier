import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import random
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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
        # Handle empty confusion matrix
        if conf_matrix is None or len(conf_matrix) == 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No data available for confusion matrix", 
                    horizontalalignment='center', fontsize=12)
            plt.title(title)
            plt.axis('off')
            
            filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            plt.savefig(self.save_dir / f'{filename}.png')
            plt.close()
            return
        
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
        Plot ROC curve with thresholds and optimal point.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_score (np.ndarray): Target scores (probability estimates of the positive class)
            class_names (list): List of class names
            title (str): Title for the plot
        """
        try:
            # For binary classification
            if len(class_names) == 2:
                plt.figure(figsize=(8, 6))
                
                # Use y_score for the positive class if it's a 2D array
                if len(y_score.shape) > 1 and y_score.shape[1] > 1:
                    y_score_binary = y_score[:, 1]
                else:
                    y_score_binary = y_score
                
                # Calculate ROC curve and ROC area
                fpr, tpr, thresholds = roc_curve(y_true, y_score_binary)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (area = {roc_auc:.3f})')
                
                # Plot chance level (diagonal line)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                
                # Plot styling
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(title)
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                
                # Add threshold labels at intervals
                num_thresholds = len(thresholds)
                if num_thresholds > 5:  # Ensure enough thresholds to sample
                    threshold_indices = [int(num_thresholds * i / 5) for i in range(6)]
                    
                    for idx in threshold_indices:
                        if idx < num_thresholds:
                            # Ensure idx is within bounds
                            threshold = thresholds[idx]
                            x, y = fpr[idx], tpr[idx]
                            plt.annotate(f'{threshold:.2f}', 
                                      (x, y), 
                                      textcoords="offset points",
                                      xytext=(0, 10),
                                      ha='center',
                                      fontsize=8,
                                      alpha=0.7)
                
                # Add optimal threshold point (closest to top-left corner)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')
                plt.annotate(f'Optimal threshold: {optimal_threshold:.2f}',
                          (fpr[optimal_idx], tpr[optimal_idx]),
                          textcoords="offset points",
                          xytext=(10, -10),
                          ha='left',
                          fontsize=9,
                          color='red')
            
            # For multi-class classification
            else:
                plt.figure(figsize=(8, 6))
                
                # If we have one-hot encoded probabilities
                if len(y_score.shape) > 1 and y_score.shape[1] > 1:
                    from sklearn.preprocessing import label_binarize
                    # Binarize labels for multi-class ROC
                    classes = np.arange(len(class_names))
                    y_bin = label_binarize(y_true, classes=classes)
                    
                    # Plot ROC curve for each class
                    for i, class_name in enumerate(class_names):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2,
                                label=f'{class_name} (AUC = {roc_auc:.3f})')
                else:
                    # Simple binary classification
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, 
                            label=f'ROC curve (AUC = {roc_auc:.3f})')
                
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
            
        except Exception as e:
            print(f"Error plotting ROC curve: {str(e)}")
            # Create a basic error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating ROC curve: {str(e)}", 
                    horizontalalignment='center', fontsize=10, wrap=True)
            plt.axis('off')
            plt.savefig(self.save_dir / 'roc_curve_error.png')
            plt.close()
    
    def plot_evaluation_metrics(self, y_true, y_pred, y_score, class_names, sampling_method=None, title=None):
        """
        Create a comprehensive visualization of evaluation metrics including:
        - ROC curve
        - Precision-Recall curve
        - Confusion matrix
        - Key metrics summary
        
        Args:
            y_true (array-like): True binary labels
            y_pred (array-like): Predicted binary labels
            y_score (array-like): Target scores (probability estimates of the positive class)
            class_names (list): List of class labels
            sampling_method (str, optional): Sampling method used for evaluation
            title (str, optional): Title for the plot
        
        Returns:
            str: Path to the saved metrics visualization
        """
        try:
            # Convert inputs to numpy arrays if they aren't already
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_score = np.array(y_score)
            
            # Create figure with 2x2 subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1. ROC Curve (top-left)
            if len(y_score.shape) > 1 and y_score.shape[1] > 1:
                # Use scores for the positive class for binary classification
                y_score_binary = y_score[:, 1]
            else:
                y_score_binary = y_score
                
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_score_binary)
            roc_auc = auc(fpr, tpr)
            
            axs[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC curve (area = {roc_auc:.3f})')
            axs[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[0, 0].set_xlim([0.0, 1.0])
            axs[0, 0].set_ylim([0.0, 1.05])
            axs[0, 0].set_xlabel('False Positive Rate')
            axs[0, 0].set_ylabel('True Positive Rate')
            axs[0, 0].set_title('ROC Curve')
            axs[0, 0].legend(loc="lower right")
            axs[0, 0].grid(True, alpha=0.3)
            
            # Add optimal threshold point
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds_roc[optimal_idx]
            axs[0, 0].plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')
            axs[0, 0].annotate(f'Optimal: {optimal_threshold:.2f}',
                             (fpr[optimal_idx], tpr[optimal_idx]),
                             textcoords="offset points",
                             xytext=(10, -10),
                             ha='left',
                             fontsize=9,
                             color='red')
            
            # 2. Precision-Recall Curve (top-right)
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score_binary)
            avg_precision = average_precision_score(y_true, y_score_binary)
            
            axs[0, 1].plot(recall, precision, color='green', lw=2,
                          label=f'PR curve (AP = {avg_precision:.3f})')
            axs[0, 1].set_xlim([0.0, 1.0])
            axs[0, 1].set_ylim([0.0, 1.05])
            axs[0, 1].set_xlabel('Recall')
            axs[0, 1].set_ylabel('Precision')
            axs[0, 1].set_title('Precision-Recall Curve')
            axs[0, 1].legend(loc="lower left")
            axs[0, 1].grid(True, alpha=0.3)
            
            # Add F1 optimal point if there are thresholds
            if len(thresholds_pr) > 0:
                # Calculate F1 scores for different thresholds
                f1_scores = []
                for i in range(len(precision) - 1):  # precision has one more element than thresholds
                    if precision[i] + recall[i] > 0:  # avoid division by zero
                        f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
                    else:
                        f1 = 0
                    f1_scores.append(f1)
                
                # Find threshold with highest F1 score
                best_f1_idx = np.argmax(f1_scores)
                if best_f1_idx < len(thresholds_pr):
                    best_f1_threshold = thresholds_pr[best_f1_idx]
                    best_precision = precision[best_f1_idx]
                    best_recall = recall[best_f1_idx]
                    
                    axs[0, 1].plot(best_recall, best_precision, 'ro')
                    axs[0, 1].annotate(f'Best F1: {best_f1_threshold:.2f}',
                                     (best_recall, best_precision),
                                     textcoords="offset points",
                                     xytext=(10, -10),
                                     ha='left',
                                     fontsize=9,
                                     color='red')
            
            # 3. Confusion Matrix (bottom-left)
            cm = np.zeros((len(class_names), len(class_names)), dtype=int)
            for i in range(len(y_true)):
                cm[y_true[i]][y_pred[i]] += 1
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, 
                       yticklabels=class_names,
                       ax=axs[1, 0])
            axs[1, 0].set_xlabel('Predicted Label')
            axs[1, 0].set_ylabel('True Label')
            axs[1, 0].set_title('Confusion Matrix')
            
            # 4. Metrics Summary (bottom-right)
            axs[1, 1].axis('off')
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Create a table for metrics
            metrics_table = [
                ['Metric', 'Value'],
                ['Accuracy', f'{accuracy:.3f}'],
                ['Precision', f'{precision:.3f}'],
                ['Recall', f'{recall:.3f}'],
                ['F1 Score', f'{f1:.3f}'],
                ['AUROC', f'{roc_auc:.3f}'],
                ['Avg Precision', f'{avg_precision:.3f}']
            ]
            
            axs[1, 1].table(
                cellText=metrics_table,
                cellLoc='center',
                loc='center',
                colWidths=[0.4, 0.3]
            )
            axs[1, 1].set_title('Performance Metrics')
            
            # Add class distribution
            class_counts = np.bincount(y_true, minlength=len(class_names))
            class_distribution = "\n".join([f"{class_names[i]}: {class_counts[i]}" for i in range(len(class_names))])
            
            axs[1, 1].text(0.5, 0.6, 
                          f"Class Distribution:\n{class_distribution}",
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.2))
            
            # Add model info
            model_info = "ViViT Model"
            if sampling_method:
                model_info += f"\nSampling: {sampling_method}"
                
            axs[1, 1].text(0.5, 0.3, 
                          model_info,
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.2))
            
            # Add title to the figure
            if title:
                plt.suptitle(title, fontsize=16)
            else:
                plt.suptitle('Laryngeal Cancer Screening - Model Evaluation Metrics', fontsize=16)
                
            plt.tight_layout()
            
            # Save the figure
            metrics_path = self.save_dir / 'evaluation_metrics.png'
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return str(metrics_path)
            
        except Exception as e:
            print(f"Error creating metrics visualization: {str(e)}")
            # Create a basic error plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating evaluation metrics: {str(e)}", 
                    horizontalalignment='center', fontsize=10, wrap=True)
            plt.axis('off')
            plt.savefig(self.save_dir / 'evaluation_metrics_error.png')
            plt.close()
            return None
    
    def plot_sample_predictions(self, images, true_labels, pred_labels, class_names=None):
        """
        Plot sample video frames with their true and predicted labels.
        
        Args:
            images (list): List of video frames arrays
            true_labels (list): True labels
            pred_labels (list): Predicted labels
            class_names (list): List of class names
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_samples = min(len(images), 5)  # Show up to 5 samples
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4*num_samples))
        
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            try:
                # Get middle frame from the video clip
                video = images[i]
                
                # Handle different input formats
                if isinstance(video, np.ndarray):
                    # For numpy array inputs
                    if len(video.shape) == 4:  # (T, H, W, C)
                        middle_frame_idx = video.shape[0] // 2
                        middle_frame = video[middle_frame_idx]
                    else:
                        print(f"Unexpected video shape: {video.shape}")
                        continue
                else:
                    # Skip non-numpy inputs
                    continue
                
                # Normalize frame for display if needed
                if middle_frame.dtype != np.uint8:
                    middle_frame = (middle_frame - middle_frame.min()) / max(1e-8, middle_frame.max() - middle_frame.min())
                    middle_frame = (middle_frame * 255).astype(np.uint8)
                
                # Display frame
                axes[i].imshow(middle_frame)
                
                # Add labels
                true_label = class_names[true_labels[i]] if class_names else f"Class {true_labels[i]}"
                pred_label = class_names[pred_labels[i]] if class_names else f"Class {pred_labels[i]}"
                
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