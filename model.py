"""Custom multi-task CNN for traffic-sign class and occlusion prediction."""
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class RoadSignCNN(nn.Module):
    """Shared convolutional trunk with sign-classification and occlusion heads."""
    def __init__(self, num_classes: int = 43, dropout: float = 0.35):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(nn.Flatten(), nn.Linear(192, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.sign_head = nn.Linear(128, num_classes)
        self.occlusion_head = nn.Linear(128, 1)

    def forward(self, images):
        representation = self.shared(self.features(images))
        return {"sign_logits": self.sign_head(representation), "occlusion_logits": self.occlusion_head(representation).squeeze(1)}
