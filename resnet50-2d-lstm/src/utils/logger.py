import logging
import os
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    """Handles experiment logging and directory management."""
    
    def __init__(self, log_dir, prefix='resnet50-lstm-training'):
        """
        Initialize experiment logger.
        
        Args:
            log_dir (str): Base directory for experiment logs
            prefix (str, optional): Prefix for the log directory name. Defaults to 'resnet50-lstm-training'.
        """
        # Create timestamped directory for this run if none provided
        if os.path.exists(log_dir) and not os.path.isdir(log_dir):
            raise ValueError(f"Log directory path exists but is not a directory: {log_dir}")
        
        self.exp_dir = Path(log_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log filename based on prefix
        log_filename = f'{prefix}.log'
        
        # Setup logger
        self.logger = logging.getLogger('ResNet50LSTM')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Setup file handler
        file_handler = logging.FileHandler(self.exp_dir / log_filename)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Experiment directory created: {self.exp_dir}")
        
    def get_logger(self):
        """Get the logger instance."""
        return self.logger
    
    def get_experiment_dir(self):
        """Get the experiment directory path."""
        return self.exp_dir
    
    def get_visualization_dir(self):
        """Get the visualization directory path."""
        viz_dir = self.exp_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        return viz_dir
    
    def get_model_path(self, timestamp=None):
        """
        Get path for saving model checkpoints.
        
        Args:
            timestamp (str, optional): Timestamp to use in the filename. 
                                     If None, current time will be used.
        
        Returns:
            Path: Path to save the model
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return self.exp_dir / f'model_{timestamp}.pth'
    
    def get_config_path(self):
        """Get path for saving training configuration."""
        return self.exp_dir / 'training_config.json'
    
    def get_metrics_path(self):
        """Get path for saving evaluation metrics."""
        return self.exp_dir / 'test_metrics.json'

def setup_logger(log_dir, prefix='resnet50-lstm-training'):
    """
    Setup and return an ExperimentLogger instance.
    
    Args:
        log_dir (str): Directory for logs
        prefix (str, optional): Prefix for log directory. Defaults to 'resnet50-lstm-training'.
    
    Returns:
        ExperimentLogger: Configured logger instance
    """
    return ExperimentLogger(log_dir, prefix)