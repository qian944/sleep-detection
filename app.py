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
    image = None # 初始化 image 为 None
    try:
        # 尝试打开图像文件
        # uploaded_file 是一个类文件对象，可以直接传递给 Image.open
        image_pil = Image.open(uploaded_file)

        # 转换为 RGB (如果图像是 RGBA, P (palette) 等格式)
        # 这一步也可能因为图像问题而出错，但通常在 open() 之后问题较少
        image = image_pil.convert("RGB")

    except UnidentifiedImageError:
        st.error("无法识别上传的文件。请确保它是一个有效的 JPG 或 PNG 图像文件，并且文件没有损坏。")
    except Exception as e:
        # 捕获其他可能的 PIL 相关错误或意外错误
        st.error(f"打开或处理图像时发生错误：请检查文件是否为标准图像格式。详情: {e}")
        # 打印详细错误到控制台供调试
        print(f"Error opening/processing image: {e}")
        import traceback
        traceback.print_exc()

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
                score = model(input_tensor)[0].item()
                if score is not None:
                        # --- 根据分数给出睡眠质量描述 ---
                        sleep_quality_description = ""
                        if score <= 10:
                            sleep_quality_description = "睡眠无问题 😊"
                        elif 11 <= score <= 20:
                            sleep_quality_description = "睡眠情况较好 🙂"
                        elif 21 <= score <= 30:
                            sleep_quality_description = "睡眠情况一般 😐"
                        elif 31 <= score <= 40:
                            sleep_quality_description = "睡眠情况较差 😟"
                        elif score >= 41: # 包含大于50的情况，如果分数范围是0-50，可以写成 41 <= score <= 50
                            sleep_quality_description = "睡眠问题严重 😫"
                        else: # 处理超出0-50范围的异常分数
                            sleep_quality_description = "分数异常，请检查模型或输入。"

                        st.success(f"预测 SRSS 分数为：**{score:.0f}**")
                        st.info(f"睡眠评估：**{sleep_quality_description}**")

            # Grad-CAM
            st.subheader("可解释性分析（Grad-CAM）")
            heatmap = generate_gradcam(model, input_tensor, device)
            st.image(heatmap, caption="Grad-CAM 热力图（模拟）", use_column_width=True)
