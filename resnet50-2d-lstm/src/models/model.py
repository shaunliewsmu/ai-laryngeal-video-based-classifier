import torch.nn as nn
import torchvision.models as models
import torch

class VideoResNet50LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.5):
        super(VideoResNet50LSTM, self).__init__()
        
        # Load pretrained 2D ResNet50 (instead of 3D ResNet18)
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fully connected layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # Freeze ResNet50 parameters
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # LSTM for temporal processing    
        # Note: ResNet50 outputs 2048-dimensional features (vs 512 for ResNet18)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        # x shape: [batch_size, channels, frames, height, width]
        batch_size, C, T, H, W = x.size()
        
        # Reshape to process each frame individually
        x = x.permute(0, 2, 1, 3, 4)  # [batch_size, frames, channels, height, width]
        x = x.reshape(batch_size * T, C, H, W)  # [batch_size*frames, channels, height, width]
        
        # Extract features for each frame using 2D ResNet50
        x = self.resnet50(x)  # [batch_size*frames, 2048, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch_size*frames, 2048]
        
        # Reshape back to sequence form
        x = x.reshape(batch_size, T, -1)  # [batch_size, frames, 2048]
        
        # Process sequence with LSTM
        x, _ = self.lstm(x)  # [batch_size, frames, hidden_size]
        
        # Take the final time step output
        x = x[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply classification head
        x = self.classifier(x)
        
        return x