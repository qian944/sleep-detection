import cv2
import numpy as np
from PIL import Image

def crop_face(image: Image.Image) -> Image.Image:
    """从 PIL Image 中裁剪出人脸区域，返回裁剪后 PIL Image。"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 加载 HaarCascade 模型（OpenCV 内置）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image  # 如果没检测到人脸，返回原图

    # 取最大的人脸区域
    x, y, w, h = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[0]
    face = img_cv[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return Image.fromarray(face_rgb)
