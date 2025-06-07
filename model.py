
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model_cbam import ResNet18_CBAM  # 如果是 ResNet18 + CBAM 模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18_CBAM()
model.load_state_dict(torch.load("model_final_cb2.pth", map_location=device))
model.to(device)
model.eval()

target_layer = model.features[7][-1]
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

def predict_and_gradcam(image: Image.Image):
    image = image.convert("RGB").resize((224, 224))
    rgb = np.array(image) / 255.0
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=tensor)[0]
    vis_img = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        pred = model(tensor).item()

    return vis_img, round(pred)
