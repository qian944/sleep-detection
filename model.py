import torch
import torch.nn as nn
from torchvision import models
from model_cbam import ResNet18_CBAM
import matplotlib.pyplot as plt
import numpy as np
from utils import apply_colormap_on_image

def load_model_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_CBAM()
    hf_url = "https://huggingface.co/qxliu/srss_model/resolve/main/model_final_cb2.pth"
    state_dict = torch.hub.load_state_dict_from_url(hf_url, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, device

def generate_gradcam(model, input_tensor, device):
    # 简易自定义 Grad-CAM
    feature_maps = []
    gradients = []

    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.features[7][-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output.backward(torch.ones_like(output))

    grads_val = gradients[0].cpu().data.numpy()[0]
    fmap = feature_maps[0].cpu().data.numpy()[0]
    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    input_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_img = np.clip(input_img, 0, 1)
    heatmap = apply_colormap_on_image(input_img, cam)

    fh.remove()
    bh.remove()

    return heatmap
