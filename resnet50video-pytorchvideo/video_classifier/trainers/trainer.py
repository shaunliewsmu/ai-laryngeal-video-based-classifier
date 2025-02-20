import torch
import os
import json
from tqdm import tqdm
from video_classifier.utils.visualization import TrainingVisualizer

class ModelTrainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, args, logger):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.logger = logger
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(args.log_dir)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Create directories
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Save training configuration
        self._save_config()
        
    def _save_config(self):
        """Save training configuration to JSON file."""
        config_path = os.path.join(self.args.log_dir, 'training_config.json')
        config = vars(self.args)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        self.logger.info(f"Saved training configuration to {config_path}")
        
    def train(self):
        """Train the model."""
        best_val_acc = 0.0
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
                
                if phase == 'val' and epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_weights = self.model.state_dict().copy()
                    self._save_best_model(epoch, best_val_acc, best_model_weights)
                
                batch_pbar.close()
                    
            # Update history and create visualization after each epoch
            for metric in self.history.keys():
                self.history[metric].append(epoch_metrics[metric])
                
            # Plot training history after each epoch
            self.visualizer.plot_training_history(self.history)
        
        self.logger.info(f'Best val Acc: {best_val_acc:4f}')
        self.model.load_state_dict(best_model_weights)
        return self.model
    
    def _save_best_model(self, epoch, best_val_acc, best_model_weights):
        """Save the best model checkpoint."""
        model_path = os.path.join(self.args.model_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_weights,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': best_val_acc,
            'history': self.history  # Save training history with model
        }, model_path)
        self.logger.info(f"Saved best model to {model_path}")