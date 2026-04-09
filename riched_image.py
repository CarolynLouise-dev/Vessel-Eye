import cv2
import os
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# PIPELINE TĂNG CƯỜNG + SEGMENTATION
# ==========================================

def get_enhanced_vessels(img):

    # ===== 1. Tạo FOV mask =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, fov_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel)

    # ===== 2. XÓA NỀN =====
    img_no_bg = cv2.bitwise_and(img, img, mask=fov_mask)

    # ===== 3. GREEN CHANNEL =====
    green = img_no_bg[:,:,1]

    # ===== 4. CLAHE =====
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    en_green = clahe.apply(green)

    # ===== CHẶN NGOÀI FOV NGAY =====
    en_green = cv2.bitwise_and(en_green, en_green, mask=fov_mask)

    # ===== 5. TOPHAT =====
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(en_green, cv2.MORPH_TOPHAT, kernel)

    en_green = cv2.addWeighted(en_green,0.8,tophat,0.4,0)

    # ===== 6. THRESHOLD =====
    denoise = cv2.medianBlur(en_green,3)

    vessel_mask = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        17,
        2
    )

    # ===== CHẶN NGOÀI FOV =====
    vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=fov_mask)

    # ===== 7. SKELETON =====
    from skimage.morphology import skeletonize

    skeleton = skeletonize(vessel_mask > 0).astype(np.uint8)*255

    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=fov_mask)

    return en_green, vessel_mask, skeleton, img_no_bg, fov_mask