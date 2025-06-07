import torch
from torchvision import transforms
import numpy as np
import cv2

# 图像预处理（保持和训练时一致）
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess(image).unsqueeze(0)  # 增加batch维度

# 简单heatmap生成示例（基于灰度强度，演示用）
def generate_heatmap(image):
    # image: PIL Image，转换为灰度并归一化为0~1
    img_gray = np.array(image.convert("L"))
    heatmap = cv2.applyColorMap((img_gray / 255 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap  # numpy array (H,W,3)

# 加载huggingface模型权重（假设模型保存在本地或用transformers hub）
def load_model_weights(model, model_path_or_url):
    # 支持本地路径或网络url
    state_dict = torch.hub.load_state_dict_from_url(model_path_or_url, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model
