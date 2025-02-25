MODEL_NAME = "resnet50 LSTM"
DATASET = "duke-raw-dataset"
PROJECT = "laryngeal_cancer_video_classification"
SEED = 42 # Global seed for reproducibility

DEFAULT_CONFIG = {
    'sequence_length': 32,
    'batch_size': 4,
    'learning_rate': 0.001,
    'epochs': 30,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.5
}
