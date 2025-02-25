import os
import logging
import numpy as np
import torch

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

# def setup_logging(log_dir):
#     log_file = os.path.join(log_dir, 'training.log')
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
def setup_logging(log_file):
    """Set up logging configuration with file creation if needed
    
    Args:
        log_file (str): Complete path to the log file
    """
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    create_directories(log_dir)
    
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file):
        open(log_file, 'a').close()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )