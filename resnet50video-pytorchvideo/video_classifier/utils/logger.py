import logging
import os
from datetime import datetime
from pathlib import Path

class ExperimentLogger:
    """Handles experiment logging and directory management."""
    
    def __init__(self, base_log_dir, prefix='resnet50-training'):
        """
        Initialize experiment logger.
        
        Args:
            base_log_dir (str): Base directory for all logs
            prefix (str, optional): Prefix for the log directory name. Defaults to 'resnet50-training'.
        """
        # Create timestamped directory for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(base_log_dir) / f'{prefix}-{self.timestamp}'
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine log filename based on prefix
        log_filename = f'{prefix}.log'
        
        # Setup logger
        self.logger = logging.getLogger('VideoClassifier')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        # Setup file handler
        file_handler = logging.FileHandler(self.exp_dir / log_filename)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        self.logger.addHandler(console_handler)
        
    def get_logger(self):
        """Get the logger instance."""
        return self.logger
    
    def get_experiment_dir(self):
        """Get the experiment directory path."""
        return self.exp_dir
    
    def get_model_path(self):
        """Get path for saving model checkpoints."""
        return self.exp_dir / 'best_model.pth'
    
    def get_config_path(self):
        """Get path for saving training configuration."""
        return self.exp_dir / 'training_config.json'
    
    def get_metrics_path(self):
        """Get path for saving evaluation metrics."""
        return self.exp_dir / 'test_metrics.json'

def setup_logger(base_log_dir, prefix='resnet50-training'):
    """
    Setup and return an ExperimentLogger instance.
    
    Args:
        base_log_dir (str): Base directory for logs
        prefix (str, optional): Prefix for log directory. Defaults to 'resnet50-training'.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    exp_logger = ExperimentLogger(base_log_dir, prefix)
    return exp_logger.get_logger()