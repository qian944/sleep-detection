import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import numpy as np
import cv2

# 加载模型并返回其主干和用于可视化的目标层
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.eval()

    # 获取某一中间层作为 target layer
    return model, model.layer4[1].conv2

# 得分预测函数
def predict_score(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()

# 简单版 Grad-CAM 手写实现
def generate_heatmap(model, target_layer, input_tensor, orig_pil):
    feature_map = {}
    grads = {}

    def forward_hook(module, input, output):
        feature_map["value"] = output

    def backward_hook(module, grad_in, grad_out):
        grads["value"] = grad_out[0]

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    output.backward()

    fmap = feature_map["value"][0]  # shape: [C, H, W]
    grad = grads["value"][0]        # shape: [C, H, W]

    weights = torch.mean(grad, dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * fmap, dim=0)
    cam = torch.relu(cam).cpu().numpy()

    # Normalize heatmap
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    cam = cv2.resize(cam, orig_pil.size)
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    orig_np = np.array(orig_pil)
    heatmap = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)

    handle_fw.remove()
    handle_bw.remove()

    return Image.fromarray(heatmap)
