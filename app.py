import streamlit as st
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import requests

from model import load_model, generate_heatmap, predict_score
from utils import load_image, detect_and_crop_face, default_crop, preprocess_image

# Hugging Face 模型 raw 文件链接，确保链接指向实际权重文件
MODEL_URL = "https://huggingface.co/qxliu/srss_model/resolve/main/model_final_cb2.pth"
MODEL_PATH = "model_final_cb2.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏬ 正在下载模型文件..."):
            r = requests.get(MODEL_URL)
            if r.status_code != 200:
                st.error("❌ 模型文件下载失败，请检查链接或稍后再试。")
                st.stop()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    model = YourModel()  # 替换为你的模型构造函数
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess(image: Image.Image):
    img = np.array(image)
    h = img.shape[0]
    upper_img = img[:int(h * 0.6), :, :]
    image = Image.fromarray(upper_img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0), upper_img

def main():
    st.title("🧠 基于人脸图像的 SRSS 睡眠评分预测系统")
    st.markdown("上传一张正脸照片，系统将预测你的睡眠质量评分（SRSS 0-50）。")

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

        st.image(cropped, caption="裁剪后的上部面部区域", use_column_width=True)

if __name__ == "__main__":
    main()

