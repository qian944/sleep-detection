import streamlit as st
from PIL import Image
import torch
from model import SRSSModel
from utils import preprocess_image, generate_heatmap, load_model_weights
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = SRSSModel()
    # 这里替换成你的huggingface模型文件的直链，比如：
    model_url = "https://huggingface.co/qxliu/srss_model/resolve/main/model_final_cb2.pth"
    model = load_model_weights(model, model_url)
    return model

def main():
    st.title("基于人脸图像的SRSS睡眠质量预测（无Grad-CAM）")

    uploaded_file = st.file_uploader("上传人脸图片", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="上传的图片", use_column_width=True)

        model = load_model()

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            pred = model(input_tensor).item()
        
        st.write(f"预测SRSS睡眠质量分数：{pred:.2f}")

        # 生成heatmap（示范：基于图像灰度强度的伪heatmap）
        heatmap = generate_heatmap(image)
        st.image(heatmap, caption="示例Heatmap（灰度伪热力图）", use_column_width=True)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

