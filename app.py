
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import requests

from model import YourModel  # 替换为你的模型类名
from utils import generate_gradcam  # 你自己的 Grad-CAM 函数

# 模型路径（Hugging Face 模型直链）
MODEL_URL = "https://huggingface.co/qxliu/srss_model/blob/main/model_final_cb2.pth"
MODEL_PATH = "model_final_cb2.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏬ 正在下载模型文件..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    model = YourModel()  # 替换为你的模型类构造
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def preprocess(image: Image.Image):
    # 将PIL图像裁剪上部60%
    img = np.array(image)
    h = img.shape[0]
    upper_img = img[:int(h * 0.6), :, :]
    image = Image.fromarray(upper_img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 根据训练时使用的尺寸修改
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0), upper_img

def main():
    st.title("🧠 基于人脸图像的 SRSS 睡眠评分预测系统")
    st.markdown("上传一张正脸照片，系统将预测你的睡眠质量评分（SRSS 0-50）并显示关注区域。")

    uploaded_file = st.file_uploader("请上传图片（jpg/png）", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="原始图像", use_column_width=True)

        input_tensor, cropped = preprocess(image)

        model = load_model()

        with torch.no_grad():
            output = model(input_tensor)
            srss_score = output.item()

        st.subheader(f"📊 预测 SRSS 睡眠评分：**{srss_score:.2f}**")

        # Grad-CAM 可视化
        st.subheader("🎯 模型关注区域（Grad-CAM）")
        gradcam_img = generate_gradcam(model, input_tensor, cropped)
        st.image(gradcam_img, caption="Grad-CAM 热力图", use_column_width=True)

if __name__ == "__main__":
    main()
