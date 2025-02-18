import torch.nn as nn
import torchvision.models.video as models

class VideoResNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.5):
        super(VideoResNetLSTM, self).__init__()
        
        # Load pretrained 3D ResNet
        self.video_resnet = models.r3d_18(
            weights=models.R3D_18_Weights.KINETICS400_V1
        )
        self.video_resnet.fc = nn.Identity()
        
        # Freeze video ResNet parameters
        for param in self.video_resnet.parameters():
            param.requires_grad = False
        
        # LSTM for temporal processing    
        self.lstm = nn.LSTM(
            input_size=512,
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
        batch_size = x.size(0)
        x = self.video_resnet(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x