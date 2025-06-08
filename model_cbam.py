import torch
import torch.nn as nn
from torchvision.models import resnet18

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat([ca.mean(1, keepdim=True), ca.max(1, keepdim=True)[0]], dim=1))
        return sa * ca

class ResNet18_CBAM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            CBAMBlock(64),
            base.layer2,
            CBAMBlock(128),
            base.layer3,
            CBAMBlock(256),
            base.layer4,
            CBAMBlock(512),
            base.avgpool,
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 1)  # 你训练时就是1维SRSS回归输出

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
