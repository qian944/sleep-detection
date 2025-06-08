import cv2
import numpy as np
from PIL import Image

def crop_face(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    return Image.fromarray(face_img)

def apply_colormap_on_image(org_img, activation_map, colormap_name='jet'):
    import matplotlib.cm as cm
    heatmap = cm.get_cmap(colormap_name)(activation_map)
    heatmap = np.delete(heatmap, 3, 2)  # Remove alpha channel
    heatmap = np.uint8(255 * heatmap)
    overlayed_img = heatmap * 0.4 + org_img * 255 * 0.6
    overlayed_img = np.uint8(overlayed_img)
    return Image.fromarray(overlayed_img)
