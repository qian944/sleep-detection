import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from model import load_model, GradCAM, show_cam_on_image
from utils import load_image, detect_and_crop_face, default_crop, preprocess_image

# 初始化
st.set_page_config(page_title="SRSS 睡眠评分预测", layout="centered")
st.title("😴 基于人脸图像的 SRSS 睡眠评分预测系统")

# 加载模型
@st.cache_resource
def get_model():
    model = load_model('srss_webapp/model_final_cb2.pth')
    model.eval()
    return model

model = get_model()

# 获取 GradCAM 目标层
target_layer = model.features[7][-1] if hasattr(model, 'features') else list(model.children())[-1]

# 图片上传
uploaded_file = st.file_uploader("上传一张人脸照片（支持 JPG/PNG）", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="原始图像", use_column_width=True)

    # 人脸裁剪
    cropped = detect_and_crop_face(image)
    if cropped is None:
        st.warning("未检测到人脸，默认使用上部裁剪。")
        cropped = default_crop(image)
    else:
        st.success("人脸识别成功，已自动裁剪。")

    # 显示裁剪图像
    st.image(cropped, caption="裁剪后图像", use_column_width=True)

    # 预处理
    input_tensor = preprocess_image(cropped).to(next(model.parameters()).device)

    # Grad-CAM 可视化
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    vis_img = show_cam_on_image(np.array(cropped.resize((224, 224))) / 255.0, grayscale_cam, use_rgb=True)

    # 预测 SRSS 分数
    with torch.no_grad():
        pred = model(input_tensor).item()

    # 显示结果
    st.subheader(f"🌙 预测 SRSS 睡眠质量得分：**{pred:.1f}**")
    st.image(vis_img, caption="Grad-CAM 可视化（红色区域为模型关注区域）", use_column_width=True)

    # 下载预测结果
    st.download_button(
        label="📥 下载预测结果",
        data=f"SRSS Score: {pred:.1f}\n",
        file_name="srss_prediction.txt"
    )
