import torch
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from timesformer_classifier.utils.early_stopping import EarlyStopping
from timesformer_classifier.utils.visualization import TrainingVisualizer
from transformers import AutoImageProcessor

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
        
        # Initialize image processor with proper settings
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.image_processor.size = {"height": 224, "width": 224}
        self.image_processor.crop_size = {"height": 224, "width": 224}
        
        # Initialize visualizer
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
        
        # Model saving path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_save_path = os.path.join(
            args.model_dir, 
            f"{timestamp}_timesformer-classifier_best_model.pth"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            verbose=True,
            delta=args.early_stopping_delta,
            path=self.model_save_path,
            trace_func=self.logger.info
        )
    
    def _process_batch(self, batch):
        """Process batch for TimeSformer model."""
        # Get video frames and labels
        video_frames = batch['pixel_values']
        labels = batch['labels'].to(self.device)  # Move labels to correct device
        batch_size = labels.size(0)
        
        try:
            # Process each video independently
            all_processed_inputs = []
            
            for i in range(batch_size):
                # Get frames for this item in the batch
                frames = video_frames[i]
                
                # Convert to numpy array if not already
                if isinstance(frames, torch.Tensor):
                    frames = frames.cpu().numpy()
                
                # Ensure frames have correct shape (T, H, W, C) and are uint8
                if isinstance(frames, np.ndarray):
                    # Check if we have extra dimensions that need to be removed
                    if frames.shape == (1, 1, 224, 3) or len(frames.shape) > 4:
                        frames = frames.squeeze()  # Remove extra dimensions
                    
                    # Ensure it's uint8 for consistent processing
                    frames = frames.astype(np.uint8)
                    
                    # Process this video with image processor
                    try:
                        # Process video frames - each video should be processed independently
                        inputs = self.image_processor(
                            images=list(frames),  # Convert to list of frames
                            return_tensors="pt",
                            do_resize=True, 
                            size={"height": 224, "width": 224},
                            do_center_crop=True,
                            crop_size={"height": 224, "width": 224}
                        )
                        
                        # Move tensors to device
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        all_processed_inputs.append(inputs)
                    except Exception as e:
                        self.logger.error(f"Error in image processor for item {i}: {str(e)}")
                        # Create placeholder inputs
                        placeholder = {
                            'pixel_values': torch.zeros((1, self.args.num_frames, 3, 224, 224), 
                                                    dtype=torch.float32, 
                                                    device=self.device)
                        }
                        all_processed_inputs.append(placeholder)
                else:
                    self.logger.error(f"Unsupported frame type for item {i}: {type(frames)}")
                    # Create placeholder
                    placeholder = {
                        'pixel_values': torch.zeros((1, self.args.num_frames, 3, 224, 224), 
                                                dtype=torch.float32, 
                                                device=self.device)
                    }
                    all_processed_inputs.append(placeholder)
            
            # Combine all inputs into a single batch
            combined_inputs = {}
            for key in all_processed_inputs[0].keys():
                combined_inputs[key] = torch.cat([inputs[key] for inputs in all_processed_inputs], dim=0)
            
            return combined_inputs, labels
        
        except Exception as e:
            self.logger.error(f"Error in processing batch: {str(e)}")
            # Return minimal placeholder inputs on correct device
            placeholder = {
                'pixel_values': torch.zeros((batch_size, self.args.num_frames, 3, 224, 224), 
                                            dtype=torch.float32, 
                                            device=self.device)
            }
            return placeholder, labels
    
    def train(self):
        """Train the model and return the best model."""
        self.logger.info(f"Starting training for {self.args.epochs} epochs")
        self.logger.info(f"Using sampling methods - Train: {self.args.train_sampling}, Val: {self.args.val_sampling}")
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(self.args.epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            # Training metrics for this epoch
            epoch_metrics = {
                'train_loss': 0.0, 'val_loss': 0.0,
                'train_acc': 0.0, 'val_acc': 0.0
            }
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Create progress bar for training batches
            train_pbar = tqdm(
                self.dataloaders['train'], 
                desc=f"Epoch {epoch+1}/{self.args.epochs} [Train - {self.args.train_sampling}]",
                leave=False
            )
            
            for batch in train_pbar:
                try:
                    # Process batch
                    inputs, labels = self._process_batch(batch)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(**inputs)
                    loss = self.criterion(outputs.logits, labels)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    # Calculate metrics
                    _, predicted = torch.max(outputs.logits, 1)
                    batch_correct = (predicted == labels).sum().item()
                    batch_size = labels.size(0)
                    
                    # Update counters
                    train_loss += loss.item() * batch_size
                    train_correct += batch_correct
                    train_total += batch_size
                    
                    # Update progress bar
                    batch_acc = batch_correct / batch_size
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{batch_acc:.4f}'
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error in training batch: {str(e)}")
                    continue
            
            # Calculate epoch training metrics
            epoch_train_loss = train_loss / train_total if train_total > 0 else 0
            epoch_train_acc = train_correct / train_total if train_total > 0 else 0
            
            epoch_metrics['train_loss'] = epoch_train_loss
            epoch_metrics['train_acc'] = epoch_train_acc
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} - Train Loss: {epoch_train_loss:.4f}, "
                f"Train Acc: {epoch_train_acc:.4f}"
            )
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Create progress bar for validation batches
            val_pbar = tqdm(
                self.dataloaders['val'], 
                desc=f"Epoch {epoch+1}/{self.args.epochs} [Val - {self.args.val_sampling}]",
                leave=False
            )
            
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        # Process batch
                        inputs, labels = self._process_batch(batch)
                        
                        # Forward pass
                        outputs = self.model(**inputs)
                        loss = self.criterion(outputs.logits, labels)
                        
                        # Calculate metrics
                        _, predicted = torch.max(outputs.logits, 1)
                        batch_correct = (predicted == labels).sum().item()
                        batch_size = labels.size(0)
                        
                        # Update counters
                        val_loss += loss.item() * batch_size
                        val_correct += batch_correct
                        val_total += batch_size
                        
                        # Update progress bar
                        batch_acc = batch_correct / batch_size
                        val_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'acc': f'{batch_acc:.4f}'
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # Calculate epoch validation metrics
            epoch_val_loss = val_loss / val_total if val_total > 0 else 0
            epoch_val_acc = val_correct / val_total if val_total > 0 else 0
            
            epoch_metrics['val_loss'] = epoch_val_loss
            epoch_metrics['val_acc'] = epoch_val_acc
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} - Val Loss: {epoch_val_loss:.4f}, "
                f"Val Acc: {epoch_val_acc:.4f}"
            )
            
            # Update history
            for metric in self.history.keys():
                self.history[metric].append(epoch_metrics[metric])
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{epoch_train_loss:.4f}',
                'val_loss': f'{epoch_val_loss:.4f}',
                'train_acc': f'{epoch_train_acc:.4f}',
                'val_acc': f'{epoch_val_acc:.4f}'
            })
            
            # Check for early stopping
            self.early_stopping(
                epoch_val_loss,
                self.model,
                self.optimizer,
                epoch,
                self.history
            )
            
            # Save best model
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_val_acc = epoch_val_acc
                self._save_best_model(epoch, epoch_val_loss, epoch_val_acc)
                
            # Create training plots
            self.visualizer.plot_training_history(self.history)
            
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping triggered!")
                break
        
        # Load best model
        best_checkpoint = torch.load(self.model_save_path)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        return self.model
    
    def _save_best_model(self, epoch, val_loss, val_acc):
        """Save the best model checkpoint."""
        # Get model config and label mappings
        if hasattr(self.model, 'config'):
            config = self.model.config.to_dict()
            id2label = self.model.config.id2label
            label2id = self.model.config.label2id
        else:
            config = None
            id2label = None
            label2id = None
        
        # Save model checkpoint with sampling method information
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': config,
            'id2label': id2label,
            'label2id': label2id,
            'num_frames': self.args.num_frames,
            'train_sampling': self.args.train_sampling,
            'val_sampling': self.args.val_sampling,
            'test_sampling': self.args.test_sampling,
        }, self.model_save_path)
        
        self.logger.info(f"Saved best model to {self.model_save_path}")