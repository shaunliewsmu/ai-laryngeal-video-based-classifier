import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

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

def plot_sampled_frames(video_path, sampled_indices, save_path):
    """Plot and save sampled frames from a video"""
    cap = cv2.VideoCapture(video_path)
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