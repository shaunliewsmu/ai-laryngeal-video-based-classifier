import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging

def print_class_metrics(y_true, y_pred, split=''):
    """Print class-wise metrics with string labels"""
    y_pred_binary = np.round(y_pred)
    
    logging.info(f'\n{split}:')
    for cls, label in [(0, 'non_referral'), (1, 'referral')]:
        mask = np.array(y_true) == cls
        if np.any(mask):
            cls_acc = accuracy_score(np.array(y_true)[mask], y_pred_binary[mask])
            logging.info(f'{label} accuracy: {cls_acc:.4f}')

def calculate_metrics(y_true, y_pred):
    y_pred_binary = np.round(y_pred)
    return {
        'auroc': roc_auc_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary)
    }