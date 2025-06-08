import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.channel = ChannelAttention(in_planes)
        self.spatial = SpatialAttention()

    def forward(self, x):
        out = x * self.channel(x)
        out = out * self.spatial(out)
        return out

from torchvision.models import resnet18

class ResNet18_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(base.children())[:-2])  # 去掉 avgpool 和 fc
        self.cbam = CBAM(512)  # 对最后一个 block 后应用 CBAM（512 是通道数）
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        att_map = x.clone()  # 保存 attention map 用于引导
        x = self.pool(x).view(x.size(0), -1)
        out = self.fc(x)
        return out, att_map


