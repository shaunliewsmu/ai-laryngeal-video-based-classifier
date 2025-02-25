import torch
import os
import json
from tqdm import tqdm
from datetime import datetime
from video_classifier.utils.visualization import TrainingVisualizer
from video_classifier.utils.early_stopping import EarlyStopping

class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, args, exp_logger):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.logger = exp_logger.get_logger()
        self.exp_dir = exp_logger.get_experiment_dir()
        
        # Initialize visualizer with experiment directory
        self.visualizer = TrainingVisualizer(self.exp_dir)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Initialize best metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.acc_threshold = 0.02  # Allow 2% accuracy drop from best
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            verbose=True,
            delta=args.early_stopping_delta,
            path=self.exp_dir / f'{timestamp}_resnet50_best_model.pth',
            trace_func=self.logger.info
        )
        
        # Save training configuration
        self._save_config()
        
    def _save_config(self):
        """Save training configuration to JSON file."""
        config_path = self.exp_dir / 'training_config.json'
        config = vars(self.args)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Saved training configuration to {config_path}")
        
    def _should_save_model(self, val_loss, val_acc):
        """
        Determine if model should be saved based on validation metrics.
        Returns True if:
        1. Loss has improved AND accuracy is within threshold of best accuracy
        2. OR if this is the first validation run
        """
        if self.best_val_loss == float('inf'):  # First run
            return True
            
        acc_within_threshold = val_acc >= (self.best_val_acc - self.acc_threshold)
        loss_improved = val_loss < self.best_val_loss
        
        return loss_improved and acc_within_threshold
        
    def train(self):
        """Train the model."""
        best_model_weights = None
        
        # Main epoch progress bar
        epoch_pbar = tqdm(range(self.args.epochs), desc='Training Progress', 
                         unit='epoch', position=0, leave=True)
        
        for epoch in epoch_pbar:
            epoch_metrics = {'train_loss': 0.0, 'val_loss': 0.0, 
                           'train_acc': 0.0, 'val_acc': 0.0}
            
            # Training and validation phases
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                    
                running_loss = 0.0
                running_corrects = 0
                total_clips = 0
                
                # Batch progress bar for each phase
                batch_pbar = tqdm(self.dataloaders[phase], 
                                desc=f'{phase.capitalize()} Batches',
                                unit='batch',
                                position=1,
                                leave=False,
                                total=len(self.dataloaders[phase]))
                
                for inputs, labels in batch_pbar:
                    try:
                        # Handle input shape (B, num_clips, C, T, H, W)
                        b, n, c, t, h, w = inputs.shape
                        inputs = inputs.view(b * n, c, t, h, w)
                        labels = labels.view(-1)
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        self.optimizer.zero_grad()
                        
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)
                            
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                        
                        # Update metrics
                        batch_loss = loss.item() * inputs.size(0)
                        batch_corrects = torch.sum(preds == labels.data)
                        running_loss += batch_loss
                        running_corrects += batch_corrects
                        total_clips += inputs.size(0)
                        
                        # Update progress bar metrics
                        batch_acc = batch_corrects.double() / inputs.size(0)
                        batch_pbar.set_postfix({
                            'loss': f'{batch_loss/inputs.size(0):.4f}',
                            'acc': f'{batch_acc:.4f}'
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error during {phase} step: {str(e)}")
                        raise
                
                epoch_loss = running_loss / total_clips
                epoch_acc = running_corrects.double() / total_clips
                
                # Store metrics for visualization
                epoch_metrics[f'{phase}_loss'] = epoch_loss
                epoch_metrics[f'{phase}_acc'] = epoch_acc.item()
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    f'{phase}_loss': f'{epoch_loss:.4f}',
                    f'{phase}_acc': f'{epoch_acc:.4f}'
                })
                
                # Log metrics
                self.logger.info(
                    f'Epoch {epoch+1}/{self.args.epochs} | {phase.capitalize()} Loss: '
                    f'{epoch_loss:.4f} Acc: {epoch_acc:.4f}'
                )
                
                if phase == 'val':
                    # Check if model should be saved
                    if self._should_save_model(epoch_loss, epoch_acc):
                        self.best_val_loss = epoch_loss
                        self.best_val_acc = max(epoch_acc, self.best_val_acc)  # Keep highest accuracy
                        best_model_weights = self.model.state_dict().copy()
                        self._save_best_model(epoch, epoch_loss, epoch_acc, best_model_weights)
                        self.logger.info(f'New best model saved! Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
                
                batch_pbar.close()
                    
            # Update history and create visualization after each epoch
            for metric in self.history.keys():
                self.history[metric].append(epoch_metrics[metric])
                
            # Plot training history after each epoch
            self.visualizer.plot_training_history(self.history)
        
        self.logger.info(f'Best val Loss: {self.best_val_loss:.4f}, Best val Acc: {self.best_val_acc:4f}')
        self.model.load_state_dict(best_model_weights)
        return self.model
    
    def _save_best_model(self, epoch, val_loss, val_acc, best_model_weights):
        """Save the best model checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create model directory if it doesn't exist
        os.makedirs(self.args.model_dir, exist_ok=True)
        model_path = os.path.join(self.args.model_dir, f'{timestamp}_resnet50_best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_weights,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history  # Save training history with model
        }, model_path)
        self.logger.info(f"Saved best model to {model_path}")