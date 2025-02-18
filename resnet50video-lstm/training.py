import os
import torch
import torch.nn as nn
import torchvision.models.video as models
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import logging
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = "resnet50 LSTM"
DATASET = "duke-raw-dataset"
PROJECT = "laryngeal_cancer_video_classification"
SEED = 42  # Global seed for reproducibility


def set_seed(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    """Plot confusion matrix using seaborn with string labels"""
    # Convert numeric predictions to string labels
    y_pred_labels = ['non_referral' if pred < 0.5 else 'referral' for pred in y_pred]
    y_true_labels = ['non_referral' if label == 0 else 'referral' for label in y_true]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non_referral', 'referral'],
                yticklabels=['non_referral', 'referral'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def print_class_metrics(y_true, y_pred, split=''):
    """Print class-wise metrics with string labels"""
    y_pred_binary = np.round(y_pred)
    
    logging.info(f'\n{split}:')
    for cls, label in [(0, 'non_referral'), (1, 'referral')]:
        mask = np.array(y_true) == cls
        if np.any(mask):
            cls_acc = accuracy_score(np.array(y_true)[mask], y_pred_binary[mask])
            logging.info(f'{label} accuracy: {cls_acc:.4f}')
            
def plot_sampled_frames(video_path, sampled_indices, save_path):
    """Plot and save sampled frames from a video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a figure with subplots
    n_cols = 8
    n_rows = (len(sampled_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(sampled_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(frame)
            axes[idx].axis('off')
            axes[idx].set_title(f'Frame {frame_idx}')
    
    # Turn off remaining empty subplots
    for idx in range(len(sampled_indices), len(axes)):
        axes[idx].axis('off')
    
    cap.release()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_auroc_curve(epochs, train_aurocs, val_aurocs, save_path):
    """Plot training and validation AUROC curves"""
    # Get actual number of epochs trained
    actual_epochs = len(train_aurocs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_epochs + 1), train_aurocs, label='Train AUROC', marker='o')
    plt.plot(range(1, actual_epochs + 1), val_aurocs, label='Validation AUROC', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

class VideoSampling:
    @staticmethod
    def uniform_sampling(total_frames, num_samples):
        """Uniformly sample frames across the video"""
        indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
        return indices
    
    @staticmethod
    def random_sampling(total_frames, num_samples):
        """Randomly sample frames from the video with reproducible results"""
        rng = np.random.RandomState(SEED)
        indices = sorted(rng.choice(total_frames, num_samples, replace=False))
        return indices
    
    @staticmethod
    def sliding_window_sampling(total_frames, num_samples, window_size=None):
        """Sample frames using sliding window approach"""
        if window_size is None:
            window_size = total_frames // num_samples
        stride = (total_frames - window_size) // (num_samples - 1) if num_samples > 1 else 1
        indices = [min(i * stride, total_frames - 1) for i in range(num_samples)]
        return indices

    @staticmethod
    def get_sampler(sampling_method):
        sampling_methods = {
            'uniform': VideoSampling.uniform_sampling,
            'random': VideoSampling.random_sampling,
            'sliding': VideoSampling.sliding_window_sampling
        }
        return sampling_methods.get(sampling_method, VideoSampling.uniform_sampling)

class VideoDataset(Dataset):
    def __init__(self, root_dir, split='train', sequence_length=32, transform=None, sampling_method='uniform'):
        self.root_dir = root_dir
        self.split = split
        self.sequence_length = sequence_length
        self.transform = transform
        self.sampling_method = sampling_method
        self.sampler = VideoSampling.get_sampler(sampling_method)
        
        # Get all video paths and labels
        self.video_paths = []
        self.labels = []
        
        # Get non-referral videos
        non_referral_path = os.path.join(root_dir, split, 'non_referral', '*.mp4')
        non_referral_videos = glob.glob(non_referral_path)
        self.video_paths.extend(non_referral_videos)
        self.labels.extend([0] * len(non_referral_videos))
        
        # Get referral videos
        referral_path = os.path.join(root_dir, split, 'referral', '*.mp4')
        referral_videos = glob.glob(referral_path)
        self.video_paths.extend(referral_videos)
        self.labels.extend([1] * len(referral_videos))
        
        logging.info(f"Found {len(self.video_paths)} videos for {split} using {sampling_method} sampling")
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self.sampler(total_frames, self.sequence_length)
        
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        frames = np.stack(frames, axis=0)  # [T, H, W, C]
        frames = frames.astype(np.float32) / 255.0
        
        if self.transform:
            frames = self.transform(frames)
        
        # Convert to tensor and rearrange dimensions to [C, T, H, W]
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.load_video(video_path)
        return frames, torch.tensor(label, dtype=torch.float32)

class VideoResNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.5):
        super(VideoResNetLSTM, self).__init__()
        
        # Load pretrained 3D ResNet
        self.video_resnet = models.r3d_18(
                weights=models.R3D_18_Weights.KINETICS400_V1
            )
        self.video_resnet.fc = nn.Identity()
        
        # Freeze video ResNet parameters
        for param in self.video_resnet.parameters():
            param.requires_grad = False
            
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.video_resnet(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

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
    class_weights = torch.FloatTensor([num_samples/(2*num_class_0), 
                                     num_samples/(2*num_class_1)]).to(device)
    
     # New way to handle class weights using pos_weight in BCEWithLogitsLoss
    pos_weight = torch.tensor([num_samples/(2*num_class_1) / (num_samples/(2*num_class_0))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    best_model_path = os.path.join(config['model_dir'], 'best_model.pth')
    
    train_aurocs = []
    val_aurocs = []
    
    for epoch in range(config['epochs']):
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
        train_auroc = roc_auc_score(train_labels, train_predictions)
        train_accuracy = accuracy_score(train_labels, np.round(train_predictions))
        train_f1 = f1_score(train_labels, np.round(train_predictions))
        train_aurocs.append(train_auroc)
        
        # Plot training confusion matrix
        plot_confusion_matrix(
            train_labels, 
            train_predictions,
            os.path.join(config['log_dir'], 'visualizations', f'train_confusion_matrix_epoch_{epoch+1}.png'),
            f'Training Confusion Matrix - Epoch {epoch+1}'
        )
        
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
        val_auroc = roc_auc_score(val_labels, val_predictions)
        val_accuracy = accuracy_score(val_labels, np.round(val_predictions))
        val_precision = precision_score(val_labels, np.round(val_predictions))
        val_recall = recall_score(val_labels, np.round(val_predictions))
        val_f1 = f1_score(val_labels, np.round(val_predictions))
        val_aurocs.append(val_auroc)
        
        # Plot validation confusion matrix
        plot_confusion_matrix(
            val_labels, 
            val_predictions,
            os.path.join(config['log_dir'], 'visualizations', f'val_confusion_matrix_epoch_{epoch+1}.png'),
            f'Validation Confusion Matrix - Epoch {epoch+1}'
        )
        
        scheduler.step(val_auroc)
        logging.info(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
        
        # Plot AUROC curves after each epoch
        plot_auroc_curve(
            epoch + 1,
            train_aurocs,
            val_aurocs,
            os.path.join(config['log_dir'], 'visualizations', f'auroc_curves_epoch_{epoch+1}.png')
        )
        
        # Log metrics
        wandb.log({
            'train_loss': avg_train_loss,
            'train_auroc': train_auroc,
            'train_accuracy': train_accuracy,
            'train_f1': train_f1,
            'val_loss': avg_val_loss,
            'val_auroc': val_auroc,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save model and check for early stopping
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Saved best model with AUROC: {best_auroc:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Print detailed metrics
        logging.info(f'Detailed metrics for epoch {epoch+1}:')
        logging.info(f'Training - Loss: {avg_train_loss:.4f}, AUROC: {train_auroc:.4f}, '
                    f'Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}')
        logging.info(f'Validation - Loss: {avg_val_loss:.4f}, AUROC: {val_auroc:.4f}, '
                    f'Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')
        
        # Print class-wise metrics
        logging.info('Class-wise metrics:')
        print_class_metrics(train_labels, train_predictions, 'Training')
        print_class_metrics(val_labels, val_predictions, 'Validation')
        for split, (y_true, y_pred) in [('Training', (train_labels, train_predictions)), 
                                       ('Validation', (val_labels, val_predictions))]:
            y_pred_binary = np.round(y_pred)
            logging.info(f'\n{split}:')
            for cls in [0, 1]:
                mask = np.array(y_true) == cls
                if np.any(mask):
                    cls_acc = accuracy_score(np.array(y_true)[mask], y_pred_binary[mask])
                    logging.info(f'Class {cls} accuracy: {cls_acc:.4f}')
    
    wandb.finish()
    return best_model_path, train_aurocs, val_aurocs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Path to model directory')
    parser.add_argument('--train_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method for training')
    parser.add_argument('--val_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method for validation')
    parser.add_argument('--test_sampling', type=str, default='uniform',
                        choices=['uniform', 'random', 'sliding'],
                        help='Frame sampling method for testing')
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(SEED)

    # Create directories
    create_directories(args.log_dir)
    create_directories(args.model_dir)
    viz_dir = os.path.join(args.log_dir, 'visualizations')
    create_directories(viz_dir)
    setup_logging(args.log_dir)

    config = {
        'sequence_length': 32,
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 30,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'data_dir': args.data_dir,
        'log_dir': args.log_dir,
        'model_dir': args.model_dir,
        'train_sampling': args.train_sampling,
        'val_sampling': args.val_sampling,
        'test_sampling': args.test_sampling
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = VideoDataset(
        root_dir=args.data_dir,
        split='train',
        sequence_length=config['sequence_length'],
        transform=transform,
        sampling_method=args.train_sampling
    )
    
    val_dataset = VideoDataset(
        root_dir=args.data_dir,
        split='val',
        sequence_length=config['sequence_length'],
        transform=transform,
        sampling_method=args.val_sampling
    )
    
    test_dataset = VideoDataset(
        root_dir=args.data_dir,
        split='test',
        sequence_length=config['sequence_length'],
        transform=transform,
        sampling_method=args.test_sampling
    )
    
    # Visualize sampling for one example video from each split
    for split, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        example_video = dataset.video_paths[0]
        cap = cv2.VideoCapture(example_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        sampled_indices = dataset.sampler(total_frames, config['sequence_length'])
        plot_sampled_frames(
            example_video,
            sampled_indices,
            os.path.join(viz_dir, f'sampled_frames_{split}_{args.train_sampling}.png')
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = VideoResNetLSTM(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    best_model_path, train_aurocs, val_aurocs = train_model(
        model, train_loader, val_loader, device, config
    )
    
    # Plot final AUROC curves
    plot_auroc_curve(
        len(train_aurocs),  # Use actual number of epochs instead of config['epochs']
        train_aurocs,
        val_aurocs,
        os.path.join(viz_dir, 'final_auroc_curves.png')
    )
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc='Testing'):
            videos = videos.to(device)
            outputs = model(videos)
            # Apply sigmoid to get predictions
            predictions = torch.sigmoid(outputs).cpu().numpy()
            test_predictions.extend(predictions)
            test_labels.extend(labels.numpy())
     # Convert lists to numpy arrays
    test_predictions = np.array(test_predictions).squeeze()
    test_labels = np.array(test_labels)
    
    if len(test_predictions) > 0 and len(test_labels) > 0:
        test_auroc = roc_auc_score(test_labels, test_predictions)
        test_accuracy = accuracy_score(test_labels, np.round(test_predictions))
        test_precision = precision_score(test_labels, np.round(test_predictions))
        test_recall = recall_score(test_labels, np.round(test_predictions))
        test_f1 = f1_score(test_labels, np.round(test_predictions))
        
        plot_confusion_matrix(
            test_labels, 
            np.round(test_predictions),
            os.path.join(viz_dir, 'test_confusion_matrix.png'),
            'Test Set Confusion Matrix'
        )
        
        logging.info('Test Results:')
        logging.info(f'AUROC: {test_auroc:.4f}')
        logging.info(f'Accuracy: {test_accuracy:.4f}')
        logging.info(f'Precision: {test_precision:.4f}')
        logging.info(f'Recall: {test_recall:.4f}')
        logging.info(f'F1 Score: {test_f1:.4f}')
        print_class_metrics(test_labels, test_predictions, 'Test')
    else:
        logging.error("No test predictions or labels collected")

if __name__ == "__main__":
    main()

"""
python3 resnet50video-lstm/training.py \
    --data_dir artifacts/laryngeal_dataset_balanced:v0/dataset \
    --log_dir logs \
    --model_dir resnet-models \
    --train_sampling uniform \
    --val_sampling uniform \
    --test_sampling uniform
"""