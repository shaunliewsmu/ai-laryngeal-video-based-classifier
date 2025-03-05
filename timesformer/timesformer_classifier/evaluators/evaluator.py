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
        
        try:
            # Process each video in the batch separately
            all_inputs = []
            for frames in video_frames:
                # Handle extra dimensions by squeezing if necessary
                if isinstance(frames, np.ndarray):
                    # Check for extra dimensions that need to be removed
                    if frames.shape == (1, 1, 224, 3) or len(frames.shape) > 4:
                        frames = frames.squeeze()  # Remove extra dimensions
                    
                    # Ensure it's uint8 for consistent processing
                    frames = frames.astype(np.uint8)
                    
                    # Process frames
                    inputs = self.image_processor(
                        images=list(frames),  # Convert to list of frames
                        return_tensors="pt",
                        do_resize=True,
                        size={"height": 224, "width": 224},
                        do_center_crop=True,
                        crop_size={"height": 224, "width": 224}
                    )
                    all_inputs.append(inputs)
                else:
                    self.logger.warning(f"Unexpected frame type: {type(frames)}")
            
            # Combine all inputs into a single batch
            if all_inputs:
                combined_inputs = {}
                for key in all_inputs[0].keys():
                    combined_inputs[key] = torch.cat([inp[key] for inp in all_inputs], dim=0)
                
                # Move to device
                combined_inputs = {k: v.to(self.device) for k, v in combined_inputs.items()}
                labels = labels.to(self.device)
                
                return combined_inputs, labels
            else:
                self.logger.error("No valid frames to process in batch")
                raise ValueError("No valid frames to process")
                
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _save_metrics(self, metrics):
        """Save evaluation metrics to JSON file."""
        metrics_path = os.path.join(self.exp_dir, f'test_metrics_{self.sampling_method}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            return obj
        
        # Create a copy of metrics to avoid modifying the original
        serializable_metrics = convert_numpy_to_list(metrics)
                
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
            
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
        all_probs = np.array(all_probs) if all_probs else np.array([])
        
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
        
        try:
            visualizer.plot_confusion_matrix(
                metrics['confusion_matrix'],
                class_names=self.class_names,
                title=f"Confusion Matrix ({self.sampling_method} sampling)"
            )
            
            if len(all_labels) > 0 and len(all_probs) > 0:
                visualizer.plot_roc_curve(
                    all_labels,
                    all_probs,
                    class_names=self.class_names,
                    title=f"ROC Curve ({self.sampling_method} sampling)"
                )
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")
        
        # Return key metrics
        return metrics.get('auroc', 0.0), metrics.get('f1_score', 0.0), metrics.get('confusion_matrix', np.array([]))
    
    def _calculate_metrics(self, labels, preds, probs):
        """Calculate detailed evaluation metrics."""
        metrics = {}
        
        # Handle empty predictions
        if len(labels) == 0 or len(preds) == 0:
            self.logger.warning("No valid predictions were made. Metrics cannot be calculated.")
            metrics['accuracy'] = 0.0
            metrics['f1_score'] = 0.0
            metrics['auroc'] = 0.0
            metrics['confusion_matrix'] = np.array([])
            return metrics
        
        # Calculate accuracy
        metrics['accuracy'] = (preds == labels).mean()
        
        try:
            # Calculate confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(labels, preds)
            
            # For binary classification
            if len(self.class_names) == 2:
                metrics['f1_score'] = f1_score(labels, preds)
                
                # ROC AUC
                if len(np.unique(labels)) > 1 and len(probs) > 0:  # Check if we have multiple classes in the test set
                    # Ensure probs is properly shaped for binary classification
                    if len(probs.shape) > 1 and probs.shape[1] > 1:
                        metrics['auroc'] = roc_auc_score(labels, probs[:, 1])
                        
                        # ROC curve
                        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
                        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                        
                        # Precision-Recall curve
                        precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
                        metrics['pr_curve'] = {'precision': precision, 'recall': recall}
                        metrics['average_precision'] = average_precision_score(labels, probs[:, 1])
                    else:
                        self.logger.warning("Probability array has incorrect shape for ROC calculation")
                        metrics['auroc'] = 0.0
                else:
                    self.logger.warning("Only one class present in test set or no probabilities, skipping ROC AUC calculation")
                    metrics['auroc'] = 0.0
                    metrics['roc_curve'] = {'fpr': [0, 1], 'tpr': [0, 1]}
                    metrics['pr_curve'] = {'precision': [0, 1], 'recall': [0, 1]}
                    metrics['average_precision'] = 0.0
            
            # For multi-class classification
            else:
                metrics['f1_score'] = f1_score(labels, preds, average='weighted')
                
                # One-vs-rest ROC AUC
                metrics['class_auroc'] = {}
                
                if len(probs) > 0:
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
                else:
                    metrics['auroc'] = 0.0
        
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            # Provide default values for required metrics
            metrics['f1_score'] = 0.0
            metrics['auroc'] = 0.0
            if 'confusion_matrix' not in metrics:
                metrics['confusion_matrix'] = np.array([])
        
        # Log metrics
        self.logger.info(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"F1 Score: {metrics.get('f1_score', 0.0):.4f}")
        self.logger.info(f"AUROC: {metrics.get('auroc', 0.0):.4f}")
        self.logger.info(f"Confusion Matrix:\n{metrics.get('confusion_matrix', np.array([]))}")
        
        return metrics