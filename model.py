import torch
import torch.nn as nn
import torchvision.models as models

class SRSSModel(nn.Module):
    def __init__(self):
        super(SRSSModel, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        # 修改最后全连接层输出为1个回归值
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # 返回形状 (batch,)
