跳至内容
历史聊天记录

您说：
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
    main()根据这个修改app。py，承袭上个对话框记忆
ChatGPT 说：
你想要基于上一个版本的 app.py，在保留主流程的前提下，把 Grad-CAM 和模型结构按实际项目（带 CBAM 模块的 ResNet18）整合进来。我会：

使用你已有的 YourModel 类加载模型（如为 ResNet18_CBAM）。

保留 Grad-CAM，根据你项目中已经实现的 generate_gradcam() 函数调用。

添加错误处理与增强部署稳定性（防止 Hugging Face URL 报错）。

加入对 torch.hub.load_state_dict_from_url() 的替代（因你使用 Hugging Face raw blob 链接）。

这是改写后的 app.py：

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import requests

from model import YourModel  # 使用你自己的模型类，如 ResNet18_CBAM
from utils import generate_gradcam  # 你自定义的 Grad-CAM 可视化函数

# Hugging Face 模型直链（需确保为 raw 链接，否则 requests 获取的是HTML页面）
MODEL_URL = "https://huggingface.co/qxliu/srss_model/resolve/main/model_final_cb2.pth"
MODEL_PATH = "model_final_cb2.pth"

@st.cache_resource
def load_model():
    # 下载模型（若不存在）
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏬ 正在下载模型文件..."):
            r = requests.get(MODEL_URL)
            if r.status_code != 200:
                st.error("❌ 模型文件下载失败，请检查链接或稍后再试。")
                st.stop()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    # 初始化模型并加载权重
    model = YourModel()  # 替换为你的模型构造函数
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess(image: Image.Image):
    # 裁剪人脸上部区域
    img = np.array(image)
    h = img.shape[0]
    upper_img = img[:int(h * 0.6), :, :]
    image = Image.fromarray(upper_img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 确保与训练时输入一致
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0), upper_img

def main():
    st.set_page_config(page_title="SRSS 睡眠评分预测", layout="centered")
    st.title("🧠 基于人脸图像的 SRSS 睡眠评分预测系统")
    st.markdown("上传一张正脸照片，系统将预测你的睡眠质量评分（SRSS 0-50）并显示模型关注区域。")

    uploaded_file = st.file_uploader("📤 请上传图片（jpg/png）", type=["jpg", "jpeg", "png"])

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
⚠️ 请确保以下几点无误：
YourModel() 定义于 model.py，且与 model_final_cb2.pth 权重兼容；

generate_gradcam() 函数可接受 (model, input_tensor, original_img)，返回 np.ndarray 或 PIL.Image；

MODEL_URL 是 Hugging Face 上的 raw 文件链接（不是 blob）。

如果你需要我一并补上 generate_gradcam() 结构（如适配 CBAM 注意力模型的 hook 设置），可以告诉我，我会一起加上。

您说：
改成不依赖gradcam的！huggingface用不了！！！！！！
ChatGPT 说：
明白了！你想要的是一个不依赖 Grad-CAM，也不使用 Hugging Face 模型在线下载的版本，直接用本地模型权重，简单加载，做推断，展示预测结果和裁剪图像就行。

我帮你改成：

去掉所有 Grad-CAM 相关代码

去掉 Hugging Face 下载，直接本地加载模型文件（模型文件需手动放到项目根目录）

只保留上传图片，裁剪上部，预测，显示原图和裁剪图，并输出预测分数

这是修改后的 app.py 代码：

import streamlit as st
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os

from model import YourModel  # 替换为你的模型类名

MODEL_PATH = "model_final_cb2.pth"  # 请确保模型文件已放到本地此路径

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"模型文件 {MODEL_PATH} 不存在，请先把模型文件放到该路径。")
        st.stop()
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
使用说明：
请把你的模型权重文件 model_final_cb2.pth 放到和 app.py 同目录下

YourModel 替换成你项目里实际模型类名和构造方式

上传人脸图片后，页面显示原图、裁剪的脸部上部、预测分数

需要我帮你写一个简单示例的 YourModel 也可以告诉我！或者需要帮你改成加载 .pt 格式模型权重也没问题。

您说：
不对，我是让你改一个不依赖gradcam的，没让你把huggingface下载也去掉
ChatGPT 说：
明白了，你想保留模型从 Hugging Face 下载的功能（自动下载模型文件），但是不使用 Grad-CAM 相关代码，即只做预测和显示原图、裁剪图以及预测分数。

我给你改成如下版本：

保留从 Hugging Face 下载模型文件的逻辑（用 requests 下载 raw 文件）

去掉所有和 Grad-CAM 相关的部分（包括调用、显示等）

保留上传图片、裁剪上部、模型推断、展示结果功能

import streamlit as st
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import requests

from model import YourModel  # 替换为你的模型类名

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

