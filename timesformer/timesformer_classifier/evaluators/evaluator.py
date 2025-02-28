import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import json
import os
from transformers import AutoImageProcessor

class ModelEvaluator:
    def __init__(self, model, dataloader, device, args, exp_logger):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.args = args
        self.logger = exp_logger.get_logger()
        self.exp_dir = exp_logger.get_experiment_dir()
        self.sampling_method = args.test_sampling
        
        # Initialize image processor with proper settings
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.image_processor.size = {"height": 224, "width": 224}
        self.image_processor.crop_size = {"height": 224, "width": 224}
        
        # Get class labels from model
        if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
            self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        else:
            self.id2label = {0: 'non-referral', 1: 'referral'}
            
        self.class_names = [self.id2label[i] for i in range(len(self.id2label))]
        
    def _process_batch(self, batch):
        """Process batch for TimeSformer model."""
        # For TimeSformer, we need to preprocess the frames
        video_frames = batch['pixel_values']
        labels = batch['labels']
        
        # Process with image processor
        inputs = self.image_processor(
            images=list(video_frames),
            return_tensors="pt",
            do_resize=True,
            size={"height": 224, "width": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224}
        )
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = labels.to(self.device)
        
        return inputs, labels
    
    def _save_metrics(self, metrics):
        """Save evaluation metrics to JSON file."""
        metrics_path = os.path.join(self.exp_dir, f'test_metrics_{self.sampling_method}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
                
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        self.logger.info(f"Saved evaluation metrics to {metrics_path}")
        
    def evaluate(self):
        """Evaluate the model on the test set."""
        self.logger.info(f"Starting model evaluation using {self.sampling_method} sampling...")
        
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Create progress bar
        test_pbar = tqdm(self.dataloader, desc=f"Evaluating [{self.sampling_method}]")
        
        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_pbar:
                try:
                    # Process batch
                    inputs, labels = self._process_batch(batch)
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    loss = criterion(outputs.logits, labels)
                    
                    # Get predictions
                    probs = torch.softmax(outputs.logits, dim=1)
                    _, predicted = torch.max(outputs.logits, 1)
                    
                    # Update metrics
                    batch_size = labels.size(0)
                    batch_correct = (predicted == labels).sum().item()
                    
                    test_loss += loss.item() * batch_size
                    test_correct += batch_correct
                    test_total += batch_size
                    
                    # Store predictions and labels
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    # Update progress bar
                    batch_acc = batch_correct / batch_size
                    test_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{batch_acc:.4f}'
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch: {str(e)}")
                    continue
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Calculate overall metrics
        accuracy = test_correct / test_total if test_total > 0 else 0
        test_loss = test_loss / test_total if test_total > 0 else 0
        
        self.logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f} using {self.sampling_method} sampling")
        
        # Calculate detailed metrics
        metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Add sampling method information to metrics
        metrics['sampling_method'] = self.sampling_method
        
        # Save metrics
        self._save_metrics(metrics)
        
        # Create visualizations
        from timesformer_classifier.utils.visualization import TrainingVisualizer
        visualizer = TrainingVisualizer(self.exp_dir)
        
        visualizer.plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=self.class_names,
            title=f"Confusion Matrix ({self.sampling_method} sampling)"
        )
        
        visualizer.plot_roc_curve(
            all_labels,
            all_probs,
            class_names=self.class_names,
            title=f"ROC Curve ({self.sampling_method} sampling)"
        )
        
        # Return key metrics
        return metrics['auroc'], metrics['f1_score'], metrics['confusion_matrix']
    
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate detailed evaluation metrics."""
        metrics = {}
        
        # Calculate accuracy
        metrics['accuracy'] = (preds == labels).mean()
        
        # Calculate confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, preds)
        
        # For binary classification
        if len(self.class_names) == 2:
            metrics['f1_score'] = f1_score(labels, preds)
            
            # ROC AUC
            if len(np.unique(labels)) > 1:  # Check if we have multiple classes in the test set
                metrics['auroc'] = roc_auc_score(labels, probs[:, 1])
                
                # ROC curve
                fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
                metrics['pr_curve'] = {'precision': precision, 'recall': recall}
                metrics['average_precision'] = average_precision_score(labels, probs[:, 1])
            else:
                self.logger.warning("Only one class present in test set, skipping ROC AUC calculation")
                metrics['auroc'] = 0.0
                metrics['roc_curve'] = {'fpr': [0, 1], 'tpr': [0, 1]}
                metrics['pr_curve'] = {'precision': [0, 1], 'recall': [0, 1]}
                metrics['average_precision'] = 0.0
        
        # For multi-class classification
        else:
            metrics['f1_score'] = f1_score(labels, preds, average='weighted')
            
            # One-vs-rest ROC AUC
            metrics['class_auroc'] = {}
            
            # Convert labels to one-hot encoding
            from sklearn.preprocessing import label_binarize
            classes = np.arange(len(self.class_names))
            y_bin = label_binarize(labels, classes=classes)
            
            if len(classes) > 2:
                metrics['auroc'] = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
                
                # Class-wise AUROC
                for i, class_name in enumerate(self.class_names):
                    try:
                        metrics['class_auroc'][class_name] = roc_auc_score(y_bin[:, i], probs[:, i])
                    except:
                        metrics['class_auroc'][class_name] = 0.0
            else:
                metrics['auroc'] = roc_auc_score(labels, probs[:, 1])
        
        # Log metrics
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        self.logger.info(f"AUROC: {metrics['auroc']:.4f}")
        self.logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics