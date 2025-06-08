import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import os

from model import load_model_and_predict, generate_gradcam
from utils import crop_face

st.set_page_config(page_title="SRSS 睡眠质量评分", layout="centered")

st.title("💤 基于人脸图像的睡眠质量预测（SRSS）")
st.markdown("上传一张人脸照片，获取睡眠评分（0-50，分数越高表示睡眠质量越差）")

uploaded_file = st.file_uploader("请上传一张人脸照片（jpg/png）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始上传图像", use_column_width=True)

    # 裁剪并展示面部区域
    cropped = crop_face(image)
    if cropped is None:
        st.error("未检测到人脸，请上传清晰的正脸照片。")
    else:
        st.image(cropped, caption="检测到的面部区域", use_column_width=True)

        # 模型预测
        with st.spinner("模型加载中..."):
            model, device = load_model_and_predict()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            input_tensor = transform(cropped).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(input_tensor).item()
            st.success(f"预测 SRSS 分数为：**{score:.2f}**")

            # Grad-CAM
            st.subheader("可解释性分析（Grad-CAM）")
            heatmap = generate_gradcam(model, input_tensor, device)
            st.image(heatmap, caption="Grad-CAM 热力图（模拟）", use_column_width=True)
