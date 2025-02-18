import torch
import numpy as np
from tqdm import tqdm
import logging
import wandb
import os
from datetime import datetime
from src.utils.metrics import calculate_metrics, print_class_metrics
from src.utils.visualization import plot_confusion_matrix, plot_auroc_curve
from src.config.config import MODEL_NAME, DATASET, PROJECT

def train_model(model, train_loader, val_loader, device, config):
    wandb.init(
        project=PROJECT,
        name=f"resnet50-lstm-duke-raw-dataset-{config['train_sampling']}",
        tags=[MODEL_NAME, DATASET, config['train_sampling']],
        entity="shaunliewsmu-singapore-management-university"
    )
    
    # Calculate class weights
    all_labels = [label for _, label in train_loader.dataset]
    num_samples = len(all_labels)
    num_class_0 = sum([1 for label in all_labels if label == 0])
    num_class_1 = num_samples - num_class_0
    
    # Calculate positive weight for BCEWithLogitsLoss
    pos_weight = torch.tensor([num_samples/(2*num_class_1) / (num_samples/(2*num_class_0))]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['learning_rate']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping parameters
    patience = 10
    no_improve_epochs = 0
    best_auroc = 0
    # Update model save path
    best_model_path = os.path.join(config['model_dir'], f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    
    train_aurocs = []
    val_aurocs = []
    
    for epoch in range(config['epochs']):
        # Training loop
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []
        
        for videos, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            try:
                videos = videos.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                # Apply sigmoid to get predictions
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                train_predictions.extend(predictions)
                train_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logging.warning(f"Error processing batch: {str(e)}")
                continue
        
        # Convert lists to numpy arrays
        train_predictions = np.array(train_predictions).squeeze()
        train_labels = np.array(train_labels)
        
        if len(train_predictions) == 0 or len(train_labels) == 0:
            logging.error("No training predictions or labels collected")
            continue
        
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(train_labels, train_predictions)
        train_aurocs.append(train_metrics['auroc'])
        
        # Plot training confusion matrix
        plot_confusion_matrix(
            train_labels, 
            train_predictions,
            os.path.join(config['viz_dir'], 'visualizations', f'train_confusion_matrix_epoch_{epoch+1}.png'),
            f'Training Confusion Matrix - Epoch {epoch+1}'
        )
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                # Apply sigmoid to get predictions
                predictions = torch.sigmoid(outputs).cpu().numpy()
                val_predictions.extend(predictions)
                val_labels.extend(labels.cpu().numpy())
                
        # Convert lists to numpy arrays
        val_predictions = np.array(val_predictions).squeeze()
        val_labels = np.array(val_labels)
        
        if len(val_predictions) == 0 or len(val_labels) == 0:
            logging.error("No validation predictions or labels collected")
            continue
        
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(val_labels, val_predictions)
        val_aurocs.append(val_metrics['auroc'])
        
        # Plot validation confusion matrix
        plot_confusion_matrix(
            val_labels, 
            val_predictions,
            os.path.join(config['viz_dir'], 'visualizations', f'val_confusion_matrix_epoch_{epoch+1}.png'),
            f'Validation Confusion Matrix - Epoch {epoch+1}'
        )
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['auroc'])
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current learning rate: {current_lr}')
        
        # Plot AUROC curves after each epoch
        plot_auroc_curve(
            epoch + 1,
            train_aurocs,
            val_aurocs,
            os.path.join(config['viz_dir'], 'visualizations', f'auroc_curves_epoch_{epoch+1}.png')
        )
        
        # Log metrics to wandb
        wandb.log({
            'train_loss': avg_train_loss,
            'train_auroc': train_metrics['auroc'],
            'train_accuracy': train_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'val_loss': avg_val_loss,
            'val_auroc': val_metrics['auroc'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'learning_rate': current_lr
        })
        
        # Model checkpointing and early stopping
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Saved best model with AUROC: {best_auroc:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Print detailed metrics
        logging.info(f'\nEpoch {epoch+1} metrics:')
        logging.info(f'Training - Loss: {avg_train_loss:.4f}, AUROC: {train_metrics["auroc"]:.4f}, '
                    f'Accuracy: {train_metrics["accuracy"]:.4f}, F1: {train_metrics["f1"]:.4f}')
        logging.info(f'Validation - Loss: {avg_val_loss:.4f}, AUROC: {val_metrics["auroc"]:.4f}, '
                    f'Accuracy: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # Print class-wise metrics
        print_class_metrics(train_labels, train_predictions, 'Training')
        print_class_metrics(val_labels, val_predictions, 'Validation')
    
    wandb.finish()
    return best_model_path, train_aurocs, val_aurocs