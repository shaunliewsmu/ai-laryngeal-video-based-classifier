import torch
import numpy as np
from tqdm import tqdm
import json
import wandb
import os
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from src.utils.metrics import calculate_metrics, print_class_metrics
from src.utils.visualization import EnhancedVisualizer

class EnhancedTrainer:
    """
    Enhanced training class with comprehensive metrics, visualization and early stopping
    """
    def __init__(self, model, train_loader, val_loader, device, config, exp_logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.logger = exp_logger.get_logger()
        self.exp_dir = Path(exp_logger.get_experiment_dir())
        
        # Create visualization directory
        self.viz_dir = self.exp_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = EnhancedVisualizer(self.viz_dir)
        
        # Calculate class weights for loss function
        train_dataset = train_loader.dataset
        num_class_1 = sum(train_dataset.labels)
        num_class_0 = len(train_dataset.labels) - num_class_1
        num_samples = len(train_dataset.labels)
        
        # Calculate positive weight for BCEWithLogitsLoss
        self.pos_weight = torch.tensor([num_samples/(2*num_class_1) / (num_samples/(2*num_class_0)) * 1.5]).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auroc': [],
            'val_auroc': []
        }
        
        # Initialize early stopping parameters
        self.early_stopping_patience = config.get('patience', 10)
        self.early_stopping_delta = config.get('early_stopping_delta', 0.001)
        self.early_stopping_counter = 0
        
        # Initialize metrics tracking
        self.best_composite_score = -float('inf')
        self.best_val_loss = float('inf')
        self.best_val_auroc = 0.0
        
        # Get model selection weighting
        self.loss_weight = config.get('loss_weight', 0.3)
        self.auroc_weight = 1.0 - self.loss_weight
        self.logger.info(f"Model selection weights - Loss: {self.loss_weight:.2f}, AUROC: {self.auroc_weight:.2f}")
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save training configuration to JSON file."""
        
        # Convert Path objects to strings for JSON serialization
        config_json = {}
        for key, value in self.config.items():
            if isinstance(value, Path):
                config_json[key] = str(value)
            else:
                config_json[key] = value
        
        config_path = self.exp_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_json, f, indent=4)
        self.logger.info(f"Saved training configuration to {config_path}")
        
    def _should_save_model(self, val_loss, val_auroc):
        """
        Determine if model should be saved based on a composite score.
        The composite score is weighted average of:
        1. Normalized validation loss (lower is better)
        2. Validation AUROC (higher is better)
        """
        # Normalize the loss to a [0,1] scale for composite scoring
        # This is necessary because loss and AUROC are on different scales
        # We use the best loss seen so far as a reference point
        best_val_loss = min(self.best_val_loss, val_loss)
        normalized_loss = best_val_loss / max(val_loss, 1e-10)  # Avoid division by zero
        
        # Compute composite score
        # Higher is better for both components (normalized_loss and AUROC)
        composite_score = (self.loss_weight * normalized_loss) + (self.auroc_weight * val_auroc)
        
        # Check if this is the best model so far
        if composite_score > self.best_composite_score:
            self.best_composite_score = composite_score
            self.best_val_loss = val_loss
            self.best_val_auroc = val_auroc
            return True
        return False
    
    def train(self):
        """Train the model with comprehensive metrics tracking and visualization."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_path = os.path.join(self.config['model_dir'], f'model_{timestamp}.pth')
        
        # Create model directory if it doesn't exist
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        # Start wandb logging if not already started
        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "laryngeal_cancer_video_classification"),
                name=f"resnet50-lstm-{timestamp}",
                config=self.config
            )
        
        # Main epoch progress bar
        epoch_pbar = tqdm(range(self.config['epochs']), desc='Training Progress', 
                         unit='epoch', position=0, leave=True)
        
        for epoch in epoch_pbar:
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_scores = []
            train_labels = []
            
            # Create progress bar for training batches
            train_pbar = tqdm(self.train_loader, desc='Training Batches',
                             unit='batch', position=1, leave=False)
            
            for videos, labels in train_pbar:
                try:
                    videos = videos.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, labels.unsqueeze(1))
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item() * videos.size(0)
                    scores = torch.sigmoid(outputs).cpu().detach().numpy()
                    preds = (scores >= 0.5).astype(int)
                    train_scores.extend(scores)
                    train_predictions.extend(preds)
                    train_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'batch_acc': f'{(preds.squeeze() == labels.cpu().numpy()).mean():.4f}'
                    })
                except Exception as e:
                    self.logger.error(f"Error in training batch: {str(e)}")
                    continue
            
            train_pbar.close()
            
            # Calculate training metrics
            train_predictions = np.array(train_predictions).squeeze()
            train_scores = np.array(train_scores).squeeze()
            train_labels = np.array(train_labels)
            
            train_loss = train_loss / len(train_labels)
            train_acc = accuracy_score(train_labels, train_predictions)
            train_auroc = roc_auc_score(train_labels, train_scores)
            train_f1 = f1_score(train_labels, train_predictions)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_scores = []
            val_labels = []
            
            # Create progress bar for validation batches
            val_pbar = tqdm(self.val_loader, desc='Validation Batches',
                           unit='batch', position=1, leave=False)
            
            with torch.no_grad():
                for videos, labels in val_pbar:
                    try:
                        videos = videos.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, labels.unsqueeze(1))
                        
                        # Update metrics
                        val_loss += loss.item() * videos.size(0)
                        scores = torch.sigmoid(outputs).cpu().numpy()
                        preds = (scores >= 0.5).astype(int)
                        val_scores.extend(scores)
                        val_predictions.extend(preds)
                        val_labels.extend(labels.cpu().numpy())
                        
                        # Update progress bar
                        val_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'batch_acc': f'{(preds.squeeze() == labels.cpu().numpy()).mean():.4f}'
                        })
                    except Exception as e:
                        self.logger.error(f"Error in validation batch: {str(e)}")
                        continue
            
            val_pbar.close()
            
            # Calculate validation metrics
            val_predictions = np.array(val_predictions).squeeze()
            val_scores = np.array(val_scores).squeeze()
            val_labels = np.array(val_labels)
            
            val_loss = val_loss / len(val_labels)
            val_acc = accuracy_score(val_labels, val_predictions)
            val_auroc = roc_auc_score(val_labels, val_scores)
            val_f1 = f1_score(val_labels, val_predictions)
            
            # Update learning rate scheduler
            self.scheduler.step(val_auroc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_auroc'].append(train_auroc)
            self.history['val_auroc'].append(val_auroc)
            
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_auroc': train_auroc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auroc': val_auroc,
                'val_f1': val_f1,
                'learning_rate': current_lr
            })
            
            # Create visualizations
            self.visualizer.plot_training_history(self.history)
            
            # Plot confusion matrices
            cm_train = confusion_matrix(train_labels, train_predictions)
            cm_val = confusion_matrix(val_labels, val_predictions)
            
            self.visualizer.plot_confusion_matrix(cm_train, class_names=['non_referral', 'referral'])
            self.visualizer.plot_confusion_matrix(cm_val, class_names=['non_referral', 'referral'])
            
            # Check for model saving
            if self._should_save_model(val_loss, val_auroc):
                self.logger.info(f"Saving best model at epoch {epoch+1} with val_loss: {val_loss:.4f}, val_auroc: {val_auroc:.4f}")
                torch.save(self.model.state_dict(), best_model_path)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                self.logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1}/{self.config['epochs']} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUROC: {train_auroc:.4f} | "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}', 
                'val_loss': f'{val_loss:.4f}',
                'val_auroc': f'{val_auroc:.4f}'
            })
        
        # End of training
        epoch_pbar.close()
        
        # Finalize and create summary visualizations
        if os.path.exists(best_model_path):
            self.logger.info(f"Loading best model from {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            self.logger.warning("No best model saved, using final model state")
            
        # Plot final ROC curves
        sample_videos = None
        sample_labels = None
        sample_predictions = None
        
        # Get some sample predictions for visualization
        try:
            self.model.eval()
            with torch.no_grad():
                for videos, labels in self.val_loader:
                    videos = videos.to(self.device)
                    outputs = self.model(videos)
                    scores = torch.sigmoid(outputs).cpu().numpy()
                    preds = (scores >= 0.5).astype(int)
                    
                    sample_videos = videos[:5].cpu()  # Get up to 5 samples
                    sample_labels = labels[:5].cpu().numpy()
                    sample_predictions = preds[:5]
                    break  # Just need one batch
                
            if sample_videos is not None:
                self.visualizer.plot_sample_predictions(
                    sample_videos, 
                    sample_labels, 
                    sample_predictions,
                    class_names=['non_referral', 'referral']
                )
        except Exception as e:
            self.logger.error(f"Error creating sample predictions visualization: {str(e)}")
            
        # Plot final AUROC curve
        try:
            self.model.eval()
            all_labels = []
            all_scores = []
            
            with torch.no_grad():
                for videos, labels in self.val_loader:
                    videos = videos.to(self.device)
                    outputs = self.model(videos)
                    scores = torch.sigmoid(outputs).cpu().numpy()
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(scores)
                    
            all_labels = np.array(all_labels)
            all_scores = np.array(all_scores).squeeze()
            
            self.visualizer.plot_auroc_curve(all_labels, all_scores)
        except Exception as e:
            self.logger.error(f"Error creating final ROC curve: {str(e)}")
        
        # Save final history
        import json
        try:
            history_path = self.exp_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                # Convert numpy values to native Python types for JSON serialization
                history_json = {k: [float(val) for val in v] for k, v in self.history.items()}
                json.dump(history_json, f, indent=4)
            self.logger.info(f"Saved training history to {history_path}")
        except Exception as e:
            self.logger.error(f"Error saving training history: {str(e)}")
        
        # Log best results
        self.logger.info(f"Training completed. Best validation results:")
        self.logger.info(f"Loss: {self.best_val_loss:.4f}")
        self.logger.info(f"AUROC: {self.best_val_auroc:.4f}")
        
        wandb.log({
            'best_val_loss': self.best_val_loss,
            'best_val_auroc': self.best_val_auroc
        })
        
        # Finalize wandb
        wandb.finish()
        
        return self.model, best_model_path