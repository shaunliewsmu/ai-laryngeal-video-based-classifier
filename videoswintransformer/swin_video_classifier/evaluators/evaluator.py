import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from swin_video_classifier.utils.visualization import TrainingVisualizer

class ModelEvaluator:
    def __init__(self, model, dataloader, device, args, exp_logger):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.logger = exp_logger.get_logger()
        self.exp_dir = exp_logger.get_experiment_dir()
        self.visualizer = TrainingVisualizer(self.exp_dir)
        self.class_names = ['non-referral', 'referral']
        
    def _save_metrics(self, metrics):
        """Save evaluation metrics."""
        metrics_path = self.exp_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Saved detailed metrics to {metrics_path}")
        return metrics
    
    def plot_roc_curve(self, y_true, y_score, save_path=None):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        
        Args:
            y_true (array-like): True binary labels
            y_score (array-like): Target scores (probability estimates of the positive class)
            save_path (str, optional): Path to save the plot. If None, defaults to 'roc_curve.png'
            
        Returns:
            float: Area Under the ROC Curve (AUROC)
        """
        try:
            # Calculate ROC curve and ROC area
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            
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
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Add threshold labels at intervals
            num_thresholds = len(thresholds)
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
            
            # Save figure
            if save_path is None:
                save_path = self.exp_dir / 'roc_curve.png'
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"ROC curve plot saved to {save_path}")
            return roc_auc
            
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            return 0.5  # Default AUROC for random classifier
            
    def plot_evaluation_metrics(self, y_true, y_pred, y_score):
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
        
        Returns:
            str: Path to the saved metrics visualization
        """
        try:
            # Create figure with 2x2 subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # 1. ROC Curve (top-left)
            fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
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
            precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)
            
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
                for threshold in thresholds_pr:
                    y_pred_threshold = (y_score >= threshold).astype(int)
                    f1 = f1_score(y_true, y_pred_threshold, zero_division=0)
                    f1_scores.append(f1)
                
                # Find threshold with highest F1 score
                best_f1_idx = np.argmax(f1_scores)
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
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=axs[1, 0])
            axs[1, 0].set_xlabel('Predicted Label')
            axs[1, 0].set_ylabel('True Label')
            axs[1, 0].set_title('Confusion Matrix')
            
            # 4. Metrics Summary (bottom-right)
            axs[1, 1].axis('off')
            
            # Calculate metrics
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
            class_counts = np.bincount(y_true, minlength=2)
            axs[1, 1].text(0.5, 0.6, 
                          f"Class Distribution:\n{self.class_names[0]}: {class_counts[0]}\n{self.class_names[1]}: {class_counts[1]}",
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.2))
            
            # Add model info
            axs[1, 1].text(0.5, 0.3, 
                          f"Model: {self.args.model_size.capitalize()} Swin3D\nSampling: {self.args.test_sampling}",
                          ha='center', va='center', fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.2))
            
            # Add title to the figure
            plt.suptitle('Laryngeal Cancer Screening - Model Evaluation Metrics', fontsize=16)
            plt.tight_layout()
            
            # Save the figure
            metrics_path = self.exp_dir / 'evaluation_metrics.png'
            plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Comprehensive evaluation metrics visualization saved to {metrics_path}")
            return str(metrics_path)
            
        except Exception as e:
            self.logger.error(f"Error creating metrics visualization: {str(e)}")
            return None
    
    def evaluate(self):
        """Evaluate the model with robust error handling."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        sample_inputs = []
        sample_labels = []
        sample_preds = []
        
        self.logger.info("Starting model evaluation...")
        
        try:
            # Create progress bar for evaluation
            eval_pbar = tqdm(enumerate(self.dataloader), desc='Evaluating',
                          total=len(self.dataloader), unit='batch', position=0, leave=True)
            
            with torch.no_grad():
                for i, batch in eval_pbar:
                    try:
                        inputs, labels = batch
                        
                        # Handle multiple clips per video
                        batch_size, num_clips, c, t, h, w = inputs.shape
                        inputs = inputs.view(-1, c, t, h, w)
                        labels = labels.view(-1)
                        
                        inputs = inputs.to(self.device)
                        outputs = self.model(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        
                        # Average predictions across clips
                        preds = preds.view(batch_size, num_clips).float().mean(dim=1).round().long()
                        probs = probs[:, 1].view(batch_size, num_clips).mean(dim=1)
                        labels = labels.view(batch_size, num_clips)[:, 0]
                        
                        # Store predictions and labels
                        batch_preds = preds.cpu().numpy()
                        batch_labels = labels.cpu().numpy()
                        
                        all_preds.extend(batch_preds)
                        all_labels.extend(batch_labels)
                        all_probs.extend(probs.cpu().numpy())
                        
                        # Update progress bar with batch accuracy
                        batch_acc = (batch_preds == batch_labels).mean()
                        eval_pbar.set_postfix({'batch_acc': f'{batch_acc:.4f}'})
                        
                        # Store some samples for visualization
                        if i == 0 and len(sample_inputs) < 5:  # Store up to 5 samples from first batch
                            for j in range(min(5 - len(sample_inputs), len(inputs))):
                                sample_inputs.append(inputs[j].cpu())
                                sample_labels.append(labels[j].cpu())
                                sample_preds.append(preds[j].cpu())
                    
                    except Exception as e:
                        self.logger.error(f"Error processing batch {i}: {str(e)}")
                        # Continue with next batch instead of crashing
                        continue
            
            eval_pbar.close()
            
            # Check if we have any valid results
            if len(all_preds) == 0:
                self.logger.error("No valid predictions were made during evaluation.")
                return 0.5, 0.0, [[0, 0], [0, 0]]
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
            
            # Create visualizations
            try:
                # Plot confusion matrix
                self.visualizer.plot_confusion_matrix(
                    metrics['confusion_matrix'], 
                    class_names=self.class_names
                )
                
                # Plot ROC curve
                roc_curve_path = self.exp_dir / 'roc_curve.png'
                self.plot_roc_curve(all_labels, all_probs, roc_curve_path)
                
                # Create comprehensive metrics visualization
                self.plot_evaluation_metrics(all_labels, all_preds, all_probs)
                
                # Plot sample predictions if we have any
                if sample_inputs:
                    self.visualizer.plot_sample_predictions(
                        sample_inputs,
                        sample_labels,
                        sample_preds,
                        class_names=self.class_names
                    )
            except Exception as e:
                self.logger.error(f"Error creating visualizations: {str(e)}")
            
            return metrics['auroc'], metrics['f1_score'], metrics['confusion_matrix']
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            # Return default values to avoid crashing the pipeline
            return 0.5, 0.0, [[0, 0], [0, 0]]
            
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate evaluation metrics with error handling."""
        try:
            auroc = roc_auc_score(labels, probs)
        except Exception as e:
            self.logger.error(f"Error calculating AUROC: {str(e)}")
            auroc = 0.5  # Default for random classifier
            
        try:
            f1 = f1_score(labels, preds)
        except Exception as e:
            self.logger.error(f"Error calculating F1 score: {str(e)}")
            f1 = 0.0  # Default
            
        try:
            conf_matrix = confusion_matrix(labels, preds)
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {str(e)}")
            conf_matrix = [[0, 0], [0, 0]]  # Default empty matrix
        
        self.logger.info(f'AUROC: {auroc:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Confusion Matrix:\n{conf_matrix}')
        
        metrics = {
            'auroc': float(auroc),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Save metrics
        try:
            metrics_path = self.exp_dir / 'test_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Saved detailed metrics to {metrics_path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
        
        return metrics