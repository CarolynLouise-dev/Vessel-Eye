import numpy as np
from skimage.measure import label, regionprops
import cv2
import math


def extract_features(binary_mask, en_green, skeleton):
    """
    Hàm trung tâm: Trích xuất toàn bộ đặc trưng y sinh từ các bộ lọc ảnh.
    Đầu vào: mask mạch máu, ảnh xanh cường độ cao, khung xương mạch máu.
    Đầu ra: [Danh sách 6 chỉ số AI], regions (để vẽ), abnormal_junctions (để vẽ)
    """
    label_img = label(binary_mask)
    regions = regionprops(label_img)

    vessel_pixels = en_green[binary_mask > 0]

    # Nếu ảnh quá nhiễu, không có mạch máu
    if len(vessel_pixels) < 10:
        return [1.0, 1.0, 0.0, 0.0, 75.0, 0], regions, []

    v_pix_mean = np.median(vessel_pixels)
    a_diams, v_diams, torts = [], [], []

    # 1. Trích xuất đặc trưng từ RegionProps (Mask & Green Channel)
    for reg in regions:
        if reg.area < 50: continue

        avg_int = np.mean(en_green[reg.coords[:, 0], reg.coords[:, 1]])

        # Công thức tính xoắn vặn chuẩn Toán học: Tỷ lệ Cung / Dây cung
        arc_length = reg.perimeter / 2.0
        chord_length = reg.axis_major_length

        if chord_length > 0:
            t_score = arc_length / chord_length
        else:
            t_score = 1.0

        if reg.area > 100:
            torts.append(t_score)

        if avg_int > v_pix_mean:
            a_diams.append(reg.axis_minor_length)
        else:
            v_diams.append(reg.axis_minor_length)

    avg_a = np.mean(a_diams) if a_diams else 1.0
    avg_v = np.mean(v_diams) if v_diams else 1.5
    av_ratio = avg_a / avg_v

    valid_torts = [t for t in torts if t > 1.02]
    avg_tort = np.mean(valid_torts) if valid_torts else 1.0
    std_tort = np.std(valid_torts) if valid_torts else 0.0
    density = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1])

    # 2. Trích xuất đặc trưng từ Skeleton (Góc & Số ngã ba)
    avg_angle, abnormal_junctions, total_branches = get_abnormal_branching(skeleton)

    # 3. Đóng gói toàn bộ 6 Features cho AI
    features = [av_ratio, avg_tort, std_tort, density, avg_angle, total_branches]

    return features, regions, abnormal_junctions


def get_abnormal_branching(skeleton):
    """
    Tính góc phân nhánh bằng vector lượng giác và đếm số lượng ngã ba.
    Đã fix lỗi đếm lặp ngã ba (Overcounting) bằng Spatial Filter.
    """
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 10, 1]], dtype=np.float32)
    skel_norm = (skeleton > 0).astype(np.float32)
    filtered = cv2.filter2D(skel_norm, -1, kernel)

    junctions = np.argwhere(filtered >= 13)

    raw_abnormal_junctions = []
    raw_all_junctions = []  # THÊM MỚI: Mảng chứa tất cả pixel ngã ba
    valid_angles = []

    h, w = skeleton.shape
    r = 8

    for j in junctions:
        y, x = j[0], j[1]
        if y < r or y >= h - r or x < r or x >= w - r:
            continue

        roi = skel_norm[y - r:y + r + 1, x - r:x + r + 1]
        y_coords, x_coords = np.where(roi > 0)
        dists = np.sqrt((x_coords - r) ** 2 + (y_coords - r) ** 2)

        branch_ends = []
        for idx in range(len(dists)):
            if dists[idx] >= r - 1.5:
                branch_ends.append((x_coords[idx] - r, y_coords[idx] - r))

        if len(branch_ends) < 2:
            continue

        angles_rad = [math.atan2(by, bx) for bx, by in branch_ends]
        angles_rad.sort()

        clusters = []
        if angles_rad:
            curr_cluster = [angles_rad[0]]
            for a in angles_rad[1:]:
                if a - curr_cluster[-1] < 0.8:
                    curr_cluster.append(a)
                else:
                    clusters.append(np.mean(curr_cluster))
                    curr_cluster = [a]
            clusters.append(np.mean(curr_cluster))

            if len(clusters) > 1 and (2 * math.pi + clusters[0] - clusters[-1]) < 0.8:
                clusters[0] = np.mean([clusters[0], clusters[-1] - 2 * math.pi])
                clusters.pop()

        if len(clusters) >= 2:
            # Lưu TẤT CẢ các pixel ngã ba hợp lệ vào mảng thô
            raw_all_junctions.append((x, y))

            is_abnormal_here = False

            for i in range(len(clusters)):
                for k in range(i + 1, len(clusters)):
                    ang_diff = abs(clusters[i] - clusters[k]) * 180.0 / math.pi
                    if ang_diff > 180:
                        ang_diff = 360 - ang_diff

                    if 20 < ang_diff < 160:
                        valid_angles.append(ang_diff)

                        if ang_diff < 35 or ang_diff > 125:
                            is_abnormal_here = True

            if is_abnormal_here:
                raw_abnormal_junctions.append((x, y))

    # --- 1. LỌC NHIỄU ĐẾM TỔNG SỐ NGÃ BA (TRÁNH OVERCOUNTING) ---
    all_junctions_filtered = []
    min_dist_count = 15  # Khoảng cách 15 pixel để gộp các pixel nhiễu thành 1 ngã ba thực tế

    for pt in raw_all_junctions:
        if not all_junctions_filtered:
            all_junctions_filtered.append(pt)
        else:
            dists = [math.hypot(pt[0] - ex[0], pt[1] - ex[1]) for ex in all_junctions_filtered]
            if min(dists) >= min_dist_count:
                all_junctions_filtered.append(pt)

    true_branch_count = len(all_junctions_filtered)  # Số đếm chính xác cuối cùng!

    # --- 2. LỌC NHIỄU CHO CÁC ĐIỂM CẢNH BÁO (VẼ CHỮ THẬP VÀNG) ---
    abnormal_junctions = []
    min_dist_draw = 35  # Giữ nguyên mức 35 pixel để ảnh không bị rối

    for pt in raw_abnormal_junctions:
        if not abnormal_junctions:
            abnormal_junctions.append(pt)
        else:
            dists = [math.hypot(pt[0] - ex[0], pt[1] - ex[1]) for ex in abnormal_junctions]
            if min(dists) >= min_dist_draw:
                abnormal_junctions.append(pt)

    avg_angle = np.mean(valid_angles) if valid_angles else 75.0
    return avg_angle, abnormal_junctions, true_branch_count