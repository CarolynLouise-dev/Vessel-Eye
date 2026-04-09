import cv2
import numpy as np
from constant import THRESHOLD_TORT_LOCAL, THRESHOLD_NARROW_LOCAL


def draw_feature_map(img_disp, vessel_mask, en_green, regions, img_no_bg=None, fov_mask=None):
    """
    img_disp   : ảnh gốc (để hiển thị nếu cần)
    vessel_mask: mask mạch máu
    en_green   : ảnh green đã enhance
    regions    : regionprops từ feature_extract
    img_no_bg  : ảnh đã loại nền
    fov_mask   : mask retina (từ riched_image.py)
    """

    if img_no_bg is None:
        img_no_bg = img_disp

    h, w = img_no_bg.shape[:2]

    # nếu chưa có fov_mask thì tạo tạm
    if fov_mask is None:
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # ======== Chuẩn bị ảnh vẽ ========
    result_img = img_no_bg.copy()
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # ======== Tính brightness threshold ========
    vessel_pixels = en_green[vessel_mask > 0]

    if len(vessel_pixels) == 0:
        return result_img

    brightness_threshold = np.median(vessel_pixels)

    # ======== Tính đường kính tĩnh mạch trung bình ========
    v_diams = []

    for r in regions:
        if r.area < 100:
            continue

        coords = r.coords
        intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])

        if intensity <= brightness_threshold:
            v_diams.append(r.axis_minor_length)

    avg_v_diam = np.mean(v_diams) if len(v_diams) > 0 else 2.0

    # ======== Vẽ vessel ========
    for reg in regions:

        if reg.area < 100:
            continue

        coords = reg.coords

        center_y, center_x = map(int, reg.centroid)

        if fov_mask[center_y, center_x] == 0:
            continue

        avg_intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])

        # artery / vein
        if avg_intensity > brightness_threshold:
            color = (0, 0, 255)   # artery - red
            is_artery = True
        else:
            color = (255, 0, 0)   # vein - blue
            is_artery = False

        rr, cc = coords[:, 0], coords[:, 1]
        color_mask[rr, cc] = color

        # ===== Tortuosity warning =====
        if reg.area > 0:
            t_score = (reg.perimeter ** 2) / (4 * np.pi * reg.area)
        else:
            t_score = 1.0

        if t_score > THRESHOLD_TORT_LOCAL:
            cv2.circle(result_img, (center_x, center_y), 12, (0, 255, 0), 2)

        # ===== Narrow artery warning =====
        if is_artery and (reg.axis_minor_length / avg_v_diam) < THRESHOLD_NARROW_LOCAL:
            cv2.circle(result_img, (center_x, center_y), 10, (255, 255, 0), 2)

    # ===== Áp dụng FOV mask =====
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=fov_mask)

    # ===== Overlay =====
    result_img = cv2.addWeighted(result_img, 1.0, color_mask, 0.3, 0)

    return result_img