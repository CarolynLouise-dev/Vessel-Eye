import cv2
import numpy as np
from constant import THRESHOLD_TORT_LOCAL


def draw_feature_map(img_color, vessel_mask, en_green, regions, img_no_bg=None, fov_mask=None, abnormal_junctions=None,
                     breakages=None):
    # Lấy kích thước từ ảnh gốc
    h, w = img_color.shape[:2]

    if fov_mask is None:
        if img_no_bg is None:
            img_no_bg = img_color
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # ========================================================
    # SỬA Ở ĐÂY: Dùng ảnh gốc (img_color) làm nền (canvas)
    # thay vì ảnh đã xóa nền (img_no_bg)
    # ========================================================
    result_img = img_color.copy()
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Phân tích R/G Ratio để lấy màu Động/Tĩnh mạch
    red_ch = img_color[:, :, 2].astype(np.float32)
    green_ch = img_color[:, :, 1].astype(np.float32)
    rg_ratio_map = red_ch / (green_ch + 1e-5)

    vessel_rg_ratios = rg_ratio_map[vessel_mask > 0]
    median_rg_ratio = np.median(vessel_rg_ratios) if len(vessel_rg_ratios) > 0 else 1.0

    # Lặp qua các vùng mạch máu để tô màu & kiểm tra xoắn vặn
    for reg in regions:
        if reg.area < 100: continue

        center_y, center_x = map(int, reg.centroid)

        # Bỏ qua nếu nằm ngoài vùng võng mạc (FOV)
        if fov_mask[center_y, center_x] == 0: continue

        rr, cc = reg.coords[:, 0], reg.coords[:, 1]
        local_rg_ratio = np.mean(rg_ratio_map[rr, cc])

        # Động mạch (Đỏ) / Tĩnh mạch (Xanh biển)
        if local_rg_ratio > median_rg_ratio:
            color = (0, 0, 255)  # Artery -> Red
        else:
            color = (255, 0, 0)  # Vein -> Blue

        color_mask[rr, cc] = color

        # Kiểm tra Xoắn vặn (Tortuosity)
        if reg.axis_major_length > 0:
            t_score = (reg.perimeter / 2.0) / reg.axis_major_length
        else:
            t_score = 1.0

        if t_score > THRESHOLD_TORT_LOCAL:
            # Vẽ vòng tròn Xanh lục khoanh vùng đoạn mạch xoắn vặn
            cv2.circle(result_img, (center_x, center_y), 12, (0, 255, 0), 2)

    # Trộn (Overlay) mạng lưới mạch máu có màu sắc lên ảnh gốc
    alpha = 0.6
    mask_indices = color_mask[:, :, 0] > 0
    result_img[mask_indices] = cv2.addWeighted(result_img, 1 - alpha, color_mask, alpha, 0)[mask_indices]

    # Vẽ ngã ba bất thường (Dấu thập Vàng)
    if abnormal_junctions is not None:
        for (jx, jy) in abnormal_junctions:
            cv2.drawMarker(result_img, (int(jx), int(jy)), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

    # Vẽ các điểm đứt gãy mạch máu (Dấu X Cam)
    if breakages is not None:
        for x, y in breakages:
            cv2.drawMarker(result_img, (int(x), int(y)), (0, 165, 255), cv2.MARKER_CROSS, 12, 2)

    return result_img