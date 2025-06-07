import streamlit as st
import torch
import os
import requests
from torchvision import transforms
from PIL import Image
import numpy as np
from model import ResNet18_CBAM
from utils import predict_with_visualization
import tempfile

# -------------------------------
# 模型下载配置
MODEL_URL = "https://huggingface.co/qxliu/srss_model/blob/main/model_final_cb2.pth"
MODEL_PATH = "model_final_cb2.pth"

# -------------------------------
# 自动下载模型
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("正在从 Hugging Face 下载模型...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("模型下载完成！")

# -------------------------------
# 应用主入口
def main():
    st.set_page_config(page_title="睡眠评分预测系统", layout="centered")
    st.title("😴 基于人脸图像的 SRSS 睡眠质量评分")

    # 下载模型
    download_model()

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_CBAM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    uploaded_file = st.file_uploader("请上传一张人脸图像（支持jpg/png）", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='上传的图像', use_column_width=True)

        # 处理图像并预测
        with st.spinner('分析中...'):
            vis_img, score = predict_with_visualization(model, image, device)
            st.image(vis_img, caption=f"预测结果：SRSS = {score:.1f}", use_column_width=True)
            st.success("✅ 分析完成")

if __name__ == "__main__":
    main()
