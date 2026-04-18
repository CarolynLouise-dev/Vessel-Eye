import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize


def _ensure_odd(value):
    value = int(round(value))
    return value if value % 2 == 1 else value + 1


def get_processing_steps(img, params):
    steps = []

    # STEP 1: FOV MASK
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_blur = _ensure_odd(params['fov_blur'])
    blur_fov = cv2.GaussianBlur(gray, (f_blur, f_blur), 0)
    _, mask_low = cv2.threshold(blur_fov, params['fov_low_thr'], 255, cv2.THRESH_BINARY)
    _, mask_otsu = cv2.threshold(blur_fov, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fov_mask = cv2.bitwise_or(mask_low, mask_otsu)
    steps.append(fov_mask)

    # STEP 2: GREEN CHANNEL (Lấy gốc từ ảnh nạp vào)
    green = img[:, :, 1]
    green_masked = cv2.bitwise_and(green, green, mask=fov_mask)
    steps.append(green_masked)

    # STEP 3: ILLUMINATION CORRECTION (Làm phẳng sáng trước khi tăng tương phản)
    k = _ensure_odd(params['illu_k'])
    la_blur = cv2.GaussianBlur(green_masked, (k, k), 0)
    illu_fixed = cv2.addWeighted(green_masked, 1.0, la_blur, -1.0, 128)
    illu_fixed = cv2.bitwise_and(illu_fixed, illu_fixed, mask=fov_mask)
    steps.append(illu_fixed)

    # STEP 4: CLAHE (Tăng chi tiết mạch máu trên nền đã phẳng sáng)
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip'], tileGridSize=(8, 8))
    en_green = clahe.apply(illu_fixed)
    en_green = cv2.bitwise_and(en_green, en_green, mask=fov_mask)
    steps.append(en_green)

    # STEP 5: GAUSSIAN BLUR (Làm mượt để khử nhiễu hạt tạo ra bởi CLAHE)
    p_blur = _ensure_odd(params['pre_blur'])
    smooth_final = cv2.GaussianBlur(en_green, (p_blur, p_blur), 0)
    smooth_final = cv2.bitwise_and(smooth_final, smooth_final, mask=fov_mask)
    steps.append(smooth_final)

    # STEP 6: FRANGI FILTER
    resp = frangi(smooth_final,
                  sigmas=np.arange(1, params['fr_scale'] + 1, 1),
                  beta=params['fr_b1'],
                  gamma=params['fr_b2'],
                  black_ridges=True)
    resp_u8 = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=fov_mask)
    steps.append(resp_u8)

    # STEP 7: SEGMENTATION & CLEANING
    _, v_mask = cv2.threshold(resp_u8, params['hyst_low'], 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(v_mask, connectivity=8)
    v_clean = np.zeros_like(v_mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= params['min_area']:
            v_clean[labels == i] = 255
    steps.append(v_clean)

    # STEP 8: FINAL SKELETON (Đã qua Pruning)
    skeleton = (skeletonize(v_clean > 0).astype(np.uint8)) * 255
    n_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    sk_clean = np.zeros_like(skeleton)
    for i in range(1, n_s):
        if stats_s[i, cv2.CC_STAT_AREA] >= params['skel_prune']:
            sk_clean[lab_s == i] = 255
    steps.append(sk_clean)

    return steps