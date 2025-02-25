import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from tqdm import tqdm
import os
import json
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
    
    def evaluate(self):
        """Evaluate the model."""
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
            eval_pbar = tqdm(self.dataloader, desc='Evaluating',
                           unit='batch', position=0, leave=True)
            
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(eval_pbar):
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
                    batch_labels = labels.numpy()
                    
                    all_preds.extend(batch_preds)
                    all_labels.extend(batch_labels)
                    all_probs.extend(probs.cpu().numpy())
                    
                    # Update progress bar with batch accuracy
                    batch_acc = (batch_preds == batch_labels).mean()
                    eval_pbar.set_postfix({'batch_acc': f'{batch_acc:.4f}'})
                    
                    # Store some samples for visualization
                    if i == 0:  # Store first batch
                        sample_inputs.extend(inputs.cpu()[:5])  # Store up to 5 samples
                        sample_labels.extend(labels.cpu()[:5])
                        sample_preds.extend(preds.cpu()[:5])
            
            eval_pbar.close()
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
            
            # Create visualizations
            self.visualizer.plot_confusion_matrix(
                metrics['confusion_matrix'], 
                class_names=self.class_names
            )
            
            # Plot sample predictions
            self.visualizer.plot_sample_predictions(
                sample_inputs,
                sample_labels,
                sample_preds,
                class_names=self.class_names
            )
            
            return metrics['auroc'], metrics['f1_score'], metrics['confusion_matrix']
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
            
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate evaluation metrics."""
        auroc = roc_auc_score(labels, probs)
        f1 = f1_score(labels, preds)
        conf_matrix = confusion_matrix(labels, preds)
        
        self.logger.info(f'AUROC: {auroc:.4f}')
        self.logger.info(f'F1 Score: {f1:.4f}')
        self.logger.info(f'Confusion Matrix:\n{conf_matrix}')
        
        metrics = {
            'auroc': float(auroc),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Save metrics
        metrics_path = os.path.join(self.args.log_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Saved detailed metrics to {metrics_path}")
        
        return metrics