import torch
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor

def create_model(model_name, num_classes, class_labels, num_frames, device, logger):
    """
    Create and initialize the ViViT model.
    
    Args:
        model_name (str): Model name or path (e.g., "google/vivit-b-16x2-kinetics400")
        num_classes (int): Number of output classes
        class_labels (list): List of class names
        num_frames (int): Number of frames to process
        device (torch.device): Device to place model on
        logger: Logger instance
        
    Returns:
        model: Initialized ViViT model
    """
    logger.info(f"Creating ViViT model based on {model_name}")
    
    try:
        # Create label mappings
        id2label = {i: label for i, label in enumerate(class_labels)}
        label2id = {label: i for i, label in enumerate(class_labels)}
        
        # Initialize configuration with our specific settings
        config = VivitConfig.from_pretrained(model_name)
        config.num_classes = num_classes
        config.id2label = {str(i): c for i, c in id2label.items()}
        config.label2id = {c: str(i) for c, i in label2id.items()}
        config.num_frames = num_frames
        
        logger.info(f"Model config: num_classes={num_classes}, num_frames={num_frames}")
        logger.info(f"Class mapping: {id2label}")
        
        # Create model with our configuration
        model = VivitForVideoClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Move model to device
        model = model.to(device)
        
        logger.info(f"Successfully created ViViT model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating ViViT model: {str(e)}")
        raise

def get_image_processor(num_frames):
    """Get the ViViT image processor with proper settings."""
    return VivitImageProcessor(
        num_frames=num_frames,
        image_size=224,
        patch_size=16,
    )