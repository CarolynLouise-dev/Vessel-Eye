import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
import warnings


def _ensure_odd(value):
    value = int(round(value))
    return value if value % 2 == 1 else value + 1


def get_processing_steps(img, params):
    steps = []

    # ===== 1. BUILD FOV MASK =====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_blur = _ensure_odd(params['fov_blur'])
    blur_fov = cv2.GaussianBlur(gray, (f_blur, f_blur), 0)
    _, mask_low = cv2.threshold(blur_fov, params['fov_low_thr'], 255, cv2.THRESH_BINARY)
    _, mask_otsu = cv2.threshold(blur_fov, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fov_mask = cv2.bitwise_or(mask_low, mask_otsu)
    steps.append(fov_mask)  # Hiển thị Mask

    # ===== 2. GREEN CHANNEL + SMOOTHING =====
    # Trong riched_image.py: green = img[:,:,1], sau đó mới GaussianBlur
    green = img[:, :, 1]
    p_blur = _ensure_odd(params['pre_blur'])
    green_blur = cv2.GaussianBlur(green, (p_blur, p_blur), 0)
    # Áp mask ngay để các bước sau không bị nhiễu nền
    green_masked = cv2.bitwise_and(green_blur, green_blur, mask=fov_mask)
    steps.append(green_masked)

    # ===== 3. ILLUMINATION CORRECTION =====
    k = _ensure_odd(params['illu_k'])
    la_blur = cv2.GaussianBlur(green_masked, (k, k), 0)
    # Công thức: green * 1.0 + blur * -1.0 + 128
    en_green = cv2.addWeighted(green_masked, 1.0, la_blur, -1.0, 128)
    en_green = cv2.bitwise_and(en_green, en_green, mask=fov_mask)
    steps.append(en_green)

    # ===== 4. CLAHE ENHANCEMENT =====
    # Thao tác trên output của Illumination
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
    en_green_final = clahe.apply(en_green)
    en_green_final = cv2.bitwise_and(en_green_final, en_green_final, mask=fov_mask)
    steps.append(en_green_final)

    # ===== 5. FRANGI VESSELNESS =====
    resp = frangi(en_green_final,
                  sigmas=np.arange(1, params['fr_scale'] + 1, 1),
                  beta=params['fr_b1'],
                  gamma=params['fr_b2'],
                  black_ridges=True)
    resp_u8 = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=fov_mask)
    steps.append(resp_u8)

    # ===== 6. SEGMENTATION (BINARY) =====
    _, v_mask = cv2.threshold(resp_u8, params['hyst_low'], 255, cv2.THRESH_BINARY)
    # Lọc diện tích nhỏ (Small Components Removal)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
    v_clean = np.zeros_like(v_mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= params['min_area']:
            v_clean[labels == i] = 255
    steps.append(v_clean)

    # ===== 7. SKELETONIZE =====
    skeleton = (skeletonize(v_clean > 0).astype(np.uint8)) * 255
    steps.append(skeleton)

    # ===== 8. SKELETON PRUNING =====
    # Bước cuối cùng trong riched_image.py là loại bỏ các nhánh cụt ngắn
    n_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    sk_clean = np.zeros_like(skeleton)
    for i in range(1, n_s):
        if stats_s[i, cv2.CC_STAT_AREA] >= params['skel_prune']:
            sk_clean[lab_s == i] = 255
    steps.append(sk_clean)

    return steps