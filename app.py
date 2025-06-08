import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from model import load_model_from_hf, get_cam
from utils import crop_face


st.set_page_config(page_title="SRSS 睡眠质量预测", layout="wide")

st.title("😴 基于人脸图像的 SRSS 睡眠质量预测")

uploaded_file = st.file_uploader("上传一张面部图片（jpg/png）", type=['jpg', 'png'])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert('RGB')
    image = crop_face(raw_image)
    st.image(image, caption='上传图像', use_column_width=True)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with st.spinner("加载模型并预测中..."):
        model = load_model_from_hf()
        with torch.no_grad():
            prediction = model(input_tensor).item()
        
        st.subheader(f"预测的 SRSS 分数：`{prediction:.2f}`")

        # Grad-CAM 解释性图
        grayscale_cam = get_cam(model, input_tensor)
        input_image = np.array(image.resize((224, 224))) / 255.0
        cam_image = (255 * cv2.cvtColor(show_cam_on_image(input_image, grayscale_cam, use_rgb=True), cv2.COLOR_RGB2BGR)).astype(np.uint8)

        st.subheader("🧠 Grad-CAM 可视化")
        st.image(cam_image, caption="模型关注区域", use_column_width=True)
