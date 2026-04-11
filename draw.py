import cv2
import numpy as np
from constant import THRESHOLD_TORT_LOCAL, THRESHOLD_NARROW_LOCAL

def draw_feature_map(img_disp, vessel_mask, en_green, regions, img_no_bg=None, fov_mask=None, abnormal_junctions=None):
    """
    img_disp   : ảnh gốc (để hiển thị nếu cần)
    vessel_mask: mask mạch máu
    en_green   : ảnh green đã enhance
    regions    : regionprops từ feature_extract
    img_no_bg  : ảnh đã loại nền
    fov_mask   : mask retina (từ riched_image.py)
    abnormal_junctions: list tọa độ (x, y) của các ngã ba phân nhánh bất thường
    """

    if img_no_bg is None:
        img_no_bg = img_disp

    h, w = img_no_bg.shape[:2]

    # Nếu chưa có fov_mask thì tạo tạm
    if fov_mask is None:
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # ======== Chuẩn bị ảnh vẽ ========
    result_img = img_no_bg.copy()
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # ======== Tính brightness threshold ========
    vessel_pixels = en_green[vessel_mask > 0]
    brightness_threshold = np.median(vessel_pixels) if len(vessel_pixels) > 0 else 127

    # ======== Vẽ vessel ========
    for reg in regions:
        if reg.area < 100:
            continue

        coords = reg.coords
        center_y, center_x = map(int, reg.centroid)

        # Bỏ qua nếu nằm ngoài vùng võng mạc
        if fov_mask[center_y, center_x] == 0:
            continue

        avg_intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])

        # Phân loại Động mạch (Đỏ) / Tĩnh mạch (Xanh)
        if avg_intensity > brightness_threshold:
            color = (0, 0, 255)   # artery - red
        else:
            color = (255, 0, 0)   # vein - blue

        rr, cc = coords[:, 0], coords[:, 1]
        color_mask[rr, cc] = color

        # ===== Cảnh báo xoắn vặn (Tortuosity) =====
        if reg.axis_major_length > 0:
            arc_len = reg.perimeter / 2.0
            chord_len = reg.axis_major_length
            t_score = arc_len / chord_len
        else:
            t_score = 1.0

        if t_score > THRESHOLD_TORT_LOCAL:
            # Vẽ vòng tròn Xanh lục cho các đoạn xoắn vặn
            cv2.circle(result_img, (center_x, center_y), 12, (0, 255, 0), 2)

    # Trộn màu overlay mạch máu lên ảnh gốc
    alpha = 0.6
    mask_indices = color_mask[:, :, 0] > 0
    result_img[mask_indices] = cv2.addWeighted(
        result_img, 1 - alpha, color_mask, alpha, 0
    )[mask_indices]

    # ======== Vẽ ngã ba bất thường (Abnormal Junctions) ========
    if abnormal_junctions is not None:
        for (jx, jy) in abnormal_junctions:
            # Dùng cv2.drawMarker để vẽ dấu chữ Thập (+) màu Vàng nổi bật tại ngã ba
            cv2.drawMarker(
                result_img,
                (int(jx), int(jy)),
                color=(0, 255, 255), # Màu vàng (Cyan-Yellow)
                markerType=cv2.MARKER_CROSS,
                markerSize=15,
                thickness=2
            )

    return result_img