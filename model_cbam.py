import torch
import torch.nn as nn
from torchvision.models import resnet18

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg = self.avg_pool(x).squeeze(-1).squeeze(-1)
        max_ = self.max_pool(x).squeeze(-1).squeeze(-1)
        channel_attn = self.sigmoid_channel(self.shared_MLP(avg) + self.shared_MLP(max_)).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(spatial))
        x = x * spatial_attn
        return x

class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # keep up to layer4
        self.cbam = CBAM(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x).view(x.size(0), -1)
        out = self.fc(x)
        return out
