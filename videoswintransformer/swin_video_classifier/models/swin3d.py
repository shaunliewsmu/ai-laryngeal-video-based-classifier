import torch.nn as nn
from torchvision.models.video import (
    swin3d_t, swin3d_s, swin3d_b,
    Swin3D_T_Weights, Swin3D_S_Weights, Swin3D_B_Weights
)

def create_model(logger, model_size="tiny", pretrained=True, num_classes=2):
    """Create a Video Swin Transformer model for video classification.
    
    Args:
        logger: Logger instance
        model_size: Model size ("tiny", "small", or "base")
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes (default: 2 for binary classification)
    
    Returns:
        model: Video Swin Transformer model
    """
    logger.info(f"Creating Video Swin Transformer ({model_size}) model...")
    
    # Select model and weights based on size
    if model_size == "tiny":
        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        model = swin3d_t(weights=weights)
        logger.info("Using Swin3D Tiny variant with 28.2M parameters")
    elif model_size == "small":
        weights = Swin3D_S_Weights.DEFAULT if pretrained else None
        model = swin3d_s(weights=weights)
        logger.info("Using Swin3D Small variant with 49.8M parameters")
    elif model_size == "base":
        weights = Swin3D_B_Weights.DEFAULT if pretrained else None
        model = swin3d_b(weights=weights)
        logger.info("Using Swin3D Base variant with 88.0M parameters")
    elif model_size == "base_in22k":
        # Base model pretrained on ImageNet-22K then Kinetics-400
        weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1 if pretrained else None
        model = swin3d_b(weights=weights)
        logger.info("Using Swin3D Base variant (ImageNet-22K pretrained) with 88.0M parameters")
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Modify the final classification head for your task
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    
    if pretrained:
        logger.info(f"Using pretrained weights from Kinetics-400")
    else:
        logger.info("Training from scratch (no pretrained weights)")
        
    logger.info(f"Modified classification head to output {num_classes} classes")
    
    return model