import torch
from transformers import TimesformerConfig, TimesformerForVideoClassification, AutoImageProcessor

def create_model(model_name, num_classes, class_labels, num_frames, device, logger):
    """
    Create and initialize the TimeSformer model.
    
    Args:
        model_name (str): Model name or path (e.g., "facebook/timesformer-base-finetuned-k400")
        num_classes (int): Number of output classes
        class_labels (list): List of class names
        num_frames (int): Number of frames to process
        device (torch.device): Device to place model on
        logger: Logger instance
        
    Returns:
        model: Initialized TimeSformer model
    """
    logger.info(f"Creating TimeSformer model based on {model_name}")
    
    try:
        # Create label mappings
        id2label = {i: label for i, label in enumerate(class_labels)}
        label2id = {label: i for i, label in enumerate(class_labels)}
        
        # Initialize configuration with our specific settings
        config = TimesformerConfig.from_pretrained(model_name)
        config.num_classes = num_classes
        config.id2label = {str(i): c for i, c in id2label.items()}
        config.label2id = {c: str(i) for c, i in label2id.items()}
        config.num_frames = num_frames
        config.video_size = [num_frames, 224, 224]
        
        logger.info(f"Model config: num_classes={num_classes}, num_frames={num_frames}")
        logger.info(f"Class mapping: {id2label}")
        
        # Create model with our configuration
        model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Move model to device
        model = model.to(device)
        
        logger.info(f"Successfully created TimeSformer model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating TimeSformer model: {str(e)}")
        raise

def get_image_processor():
    """Get the TimeSformer image processor with proper settings."""
    # Use AutoImageProcessor instead of TimesformerImageProcessor
    image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    
    # Apply custom configurations as shown in the notebook
    image_processor.size = {"height": 224, "width": 224}
    image_processor.crop_size = {"height": 224, "width": 224}
    
    return image_processor