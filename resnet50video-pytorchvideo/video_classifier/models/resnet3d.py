import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet

def create_model(logger):
    """Create a 3D ResNet model for video classification."""
    logger.info("Creating 3D ResNet-50 model...")
    
    model = create_resnet(
        # Model configs
        model_depth=50,
        model_num_class=2,  # Binary classification
        dropout_rate=0.5,
        
        # Input clip configs
        input_channel=3,
        
        # Normalization configs
        norm=nn.BatchNorm3d,
        
        # Activation configs
        activation=nn.ReLU,
        
        # Stem configs
        stem_dim_out=64,
        stem_conv_kernel_size=(3, 7, 7),
        stem_conv_stride=(1, 2, 2),
        stem_pool=nn.MaxPool3d,
        stem_pool_kernel_size=(1, 3, 3),
        stem_pool_stride=(1, 2, 2),
        
        # Stage configs
        stage_conv_a_kernel_size=((1, 1, 1), (1, 1, 1), (3, 1, 1), (3, 1, 1)),
        stage_conv_b_kernel_size=((1, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)),
        stage_conv_b_num_groups=(1, 1, 1, 1),
        stage_conv_b_dilation=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        stage_spatial_h_stride=(1, 2, 2, 2),
        stage_spatial_w_stride=(1, 2, 2, 2),
        stage_temporal_stride=(1, 1, 1, 1),
        
        # Head configs
        head_pool=nn.AvgPool3d,
        head_pool_kernel_size=(4, 7, 7),
        head_output_size=(1, 1, 1),
        head_activation=None,
        head_output_with_global_average=True,
    )
    
    logger.info("Model created successfully")
    return model