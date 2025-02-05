# src/config.py
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent  # Go up one level from src/
YOLOV5_DIR = BASE_DIR / "yolov5"

# Model paths
WEIGHTS_PATH = BASE_DIR / "model/yolov5.pt"
DATA_PATH = YOLOV5_DIR / "data/custom_data.yaml"

# Device configuration
DEVICE = "0"