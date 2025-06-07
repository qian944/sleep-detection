import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import torch
import torchvision.transforms as transforms

mp_face_detection = mp.solutions.face_detection
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def detect_and_crop_face(image):
    img_array = np.array(image)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img_array.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                cropped = img_array[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if cropped.size > 0:
                    return Image.fromarray(cropped)
    return None

def default_crop(image, ratio=0.6):
    w, h = image.size
    return image.crop((0, 0, w, int(h * ratio)))

def preprocess_image(image):
    return transform(image).unsqueeze(0)
