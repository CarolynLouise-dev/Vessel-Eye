import numpy as np
from skimage.measure import label, regionprops
import cv2
import math


def extract_features(img_color, binary_mask, en_green, skeleton, fov_mask):
    """
    Trích xuất đặc trưng với thuật toán "Phân mảnh vi mạch" (Vessel Segmentation)
    giúp triệt tiêu lỗi Overflow và tăng độ chuẩn xác đo lường lâm sàng.
    """
    vessel_pixels = en_green[binary_mask > 0]
    if len(vessel_pixels) < 10:
        return [1.0, 1.0, 0.0, 0.0, 75.0, 0, 0], [], [], {"a_count": 0, "v_count": 0, "phong_mach": 0, "vo_mach": 0,
                                                          "dut_mach": 0}, []

    # 1. TÍNH BẢN ĐỒ KHOẢNG CÁCH (Lấy độ dày thực tế của mạch máu)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)

    # ====================================================================
    # 2. PHÂN MẢNH MẠNG LƯỚI (KHẮC PHỤC LỖI OVERFLOW CỦA NUMPY)
    # ====================================================================
    skel_norm = (skeleton > 0).astype(np.float32)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    filtered = cv2.filter2D(skel_norm, -1, kernel)
    junctions = np.argwhere(filtered >= 13)

    mask_segmented = binary_mask.copy()

    for pt in junctions:
        # Ép kiểu an toàn qua float rồi mới sang int
        y = int(float(pt[0]))
        x = int(float(pt[1]))

        # Tính bán kính cắt mạch máu
        r_val = float(dist_transform[y, x])
        r_cut = int(r_val) + 1

        # 🛡️ CHỐT CHẶN AN TOÀN: Bán kính mạch máu không thể lớn hơn 30 pixel
        # Nếu lớn hơn 30 (do lỗi toán học/nhiễu), tự động ép về 30 để OpenCV không bị sập
        if r_cut > 30:
            r_cut = 30
        if r_cut < 1:
            r_cut = 1

        # Vẽ điểm cắt (Ép tuple nguyên thủy cho OpenCV)
        cv2.circle(mask_segmented, (x, y), r_cut, (0, 0, 0), -1)

    # Đưa mask đã băm nhỏ vào regionprops
    label_img = label(mask_segmented)
    regions = regionprops(label_img)
    # ====================================================================

    # 3. TÍNH KHÔNG GIAN MÀU R/G ĐỂ TÌM ĐỘNG/TĨNH MẠCH
    red_ch = img_color[:, :, 2].astype(np.float32)
    green_ch = img_color[:, :, 1].astype(np.float32)
    rg_ratio_map = red_ch / (green_ch + 1e-5)
    median_rg_ratio = np.median(rg_ratio_map[binary_mask > 0])

    a_count, v_count, phong_mach, vo_mach = 0, 0, 0, 0
    avg_thickness = np.mean(dist_transform[binary_mask > 0])

    a_diams, v_diams, torts = [], [], []

    for reg in regions:
        if reg.area < 20: continue

        rr, cc = reg.coords[:, 0], reg.coords[:, 1]
        local_rg_ratio = np.mean(rg_ratio_map[rr, cc])

        # Tính toán hình học cơ bản
        major = reg.axis_major_length
        minor = reg.axis_minor_length
        aspect_ratio = major / (minor + 1e-5)  # Tỷ lệ Dài / Rộng

        # Phân loại nhánh Động mạch / Tĩnh mạch
        if local_rg_ratio > median_rg_ratio:
            a_diams.append(minor)
            a_count += 1
        else:
            v_diams.append(minor)
            v_count += 1

        # =======================================================
        # LOGIC MỚI CHO CHUẨN ĐOÁN LÂM SÀNG NON-AI
        # =======================================================

        # 1. Cảnh báo xuất huyết / vỡ mạch (vo_mach)
        # Đặc điểm y khoa: Là mảng vỡ lan rộng, diện tích lớn, không thon dài (aspect_ratio thấp)
        if reg.area > 100 and aspect_ratio < 2.5 and reg.solidity > 0.7:
            vo_mach += 1

        # 2. Cảnh báo phình mạch cục bộ (phong_mach)
        # Đặc điểm y khoa: Dạng bầu dục/tròn lồi lên (aspect_ratio < 4.0) và độ dày phải vượt trội
        mask_reg = np.zeros_like(binary_mask)
        mask_reg[rr, cc] = 255
        local_dist = cv2.bitwise_and(dist_transform, dist_transform, mask=mask_reg)
        max_local_thick = np.max(local_dist)

        if max_local_thick > avg_thickness * 2.5 and aspect_ratio < 4.0:
            phong_mach += 1

        # =======================================================

        # Đo độ xoắn vặn (Tortuosity) cực kỳ chính xác trên từng nhánh đơn
        if reg.axis_major_length > 0:
            torts.append((reg.perimeter / 2.0) / reg.axis_major_length)

    # 4. TỔNG HỢP CHỈ SỐ AI VÀ NON-AI
    dut_mach, breakages_pts = detect_breakages(skeleton, fov_mask)
    avg_angle, abnormal_junctions, total_branches = get_abnormal_branching(skeleton)

    non_ai_stats = {
        "a_count": a_count, "v_count": v_count,
        "phong_mach": phong_mach, "vo_mach": vo_mach, "dut_mach": dut_mach
    }

    avg_a = np.mean(a_diams) if a_diams else 1.0
    avg_v = np.mean(v_diams) if v_diams else 1.5
    av_ratio = avg_a / avg_v

    valid_torts = [t for t in torts if t > 1.02]
    avg_tort = np.mean(valid_torts) if valid_torts else 1.0
    std_tort = np.std(valid_torts) if valid_torts else 0.0
    density = np.sum(binary_mask > 0) / (binary_mask.shape[0] * binary_mask.shape[1])

    ai_features = [av_ratio, avg_tort, std_tort, density, avg_angle, total_branches, dut_mach]

    return ai_features, regions, abnormal_junctions, non_ai_stats, breakages_pts


def detect_breakages(skeleton, fov_mask):
    """
    Dò tìm điểm đứt gãy (breakages) dựa trên khoảng cách giữa các điểm cụt.
    """
    # 1. KHỬ NHIỄU KHUNG XƯƠNG: Chỉ giữ lại các nhánh dài (>= 30 pixels)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    clean_skel = np.zeros_like(skeleton, dtype=np.float32)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            clean_skel[labels == i] = 1.0

    # 2. TÌM TẤT CẢ CÁC ĐIỂM CỤT (Endpoints)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    filtered = cv2.filter2D(clean_skel, -1, kernel)
    endpoints = np.argwhere(filtered == 11)

    # 3. LỌC VÙNG RÌA (Bỏ qua các điểm cụt nằm sát viền võng mạc)
    kernel_erode = np.ones((60, 60), np.uint8)
    inner_fov = cv2.erode(fov_mask, kernel_erode, iterations=1)

    valid_endpoints = []
    for y, x in endpoints:
        if inner_fov[y, x] > 0:
            valid_endpoints.append((x, y))

    # 4. TÌM CẶP ĐIỂM ĐỨT GÃY THỰC SỰ (Logic Mới)
    # Một đứt gãy thực sự là 2 điểm cụt hướng vào nhau, cách nhau 3 -> 25 pixel
    breakages_pts = []
    dut_mach_count = 0
    used_idx = set()

    for i in range(len(valid_endpoints)):
        if i in used_idx: continue
        pt1 = valid_endpoints[i]

        # Tìm điểm cụt gần nó nhất
        min_d = float('inf')
        best_j = -1
        for j in range(i + 1, len(valid_endpoints)):
            if j in used_idx: continue
            pt2 = valid_endpoints[j]
            d = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
            if d < min_d:
                min_d = d
                best_j = j

        # Nếu điểm gần nhất nằm trong khoảng 3 đến 25 pixel -> Chính xác là 1 đoạn đứt
        if 3 <= min_d <= 25 and best_j != -1:
            dut_mach_count += 1
            breakages_pts.append(pt1)  # Lưu 1 điểm để vẽ cảnh báo trên giao diện
            used_idx.add(i)
            used_idx.add(best_j)

    return dut_mach_count, breakages_pts


def get_abnormal_branching(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 10, 1]], dtype=np.float32)
    skel_norm = (skeleton > 0).astype(np.float32)
    filtered = cv2.filter2D(skel_norm, -1, kernel)
    junctions = np.argwhere(filtered >= 13)

    raw_abnormal_junctions, raw_all_junctions, valid_angles = [], [], []
    h, w = skeleton.shape
    r = 8

    for j in junctions:
        y, x = j[0], j[1]
        if y < r or y >= h - r or x < r or x >= w - r: continue
        roi = skel_norm[y - r:y + r + 1, x - r:x + r + 1]
        y_coords, x_coords = np.where(roi > 0)
        dists = np.sqrt((x_coords - r) ** 2 + (y_coords - r) ** 2)
        branch_ends = [(x_coords[idx] - r, y_coords[idx] - r) for idx in range(len(dists)) if dists[idx] >= r - 1.5]

        if len(branch_ends) < 2: continue
        angles_rad = sorted([math.atan2(by, bx) for bx, by in branch_ends])

        clusters = []
        if angles_rad:
            curr = [angles_rad[0]]
            for a in angles_rad[1:]:
                if a - curr[-1] < 0.8:
                    curr.append(a)
                else:
                    clusters.append(np.mean(curr)); curr = [a]
            clusters.append(np.mean(curr))
            if len(clusters) > 1 and (2 * math.pi + clusters[0] - clusters[-1]) < 0.8:
                clusters[0] = np.mean([clusters[0], clusters[-1] - 2 * math.pi])
                clusters.pop()

        if len(clusters) >= 2:
            raw_all_junctions.append((x, y))
            is_ab = False
            for i in range(len(clusters)):
                for k in range(i + 1, len(clusters)):
                    ang = abs(clusters[i] - clusters[k]) * 180.0 / math.pi
                    ang = 360 - ang if ang > 180 else ang
                    if 20 < ang < 160:
                        valid_angles.append(ang)
                        if ang < 45 or ang > 105: is_ab = True
            if is_ab: raw_abnormal_junctions.append((x, y))

    def spatial_filter(pts, min_dist):
        res = []
        for p in pts:
            if not res or min([math.hypot(p[0] - e[0], p[1] - e[1]) for e in res]) >= min_dist:
                res.append(p)
        return res

    true_count = len(spatial_filter(raw_all_junctions, 15))
    ab_juncs = spatial_filter(raw_abnormal_junctions, 35)
    return np.mean(valid_angles) if valid_angles else 75.0, ab_juncs, true_count


def is_valid_fundus_image(img, min_bg_ratio=0.05):
    """
    Kiểm tra xem ảnh có phải là ảnh nhãn cầu tiêu chuẩn không.
    - img: ảnh đầu vào (BGR hoặc Grayscale đều được)
    - min_bg_ratio: Tỷ lệ nền đen tối thiểu (mặc định 5%)
    """
    # Chuyển về ảnh xám nếu đang là ảnh màu
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Lọc lấy phần nền đen (background)
    # Các pixel có độ sáng < 15 được coi là nền đen
    _, bg_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)

    # Tính tỷ lệ nền đen trên tổng diện tích ảnh
    total_pixels = gray.shape[0] * gray.shape[1]
    bg_pixels = cv2.countNonZero(bg_mask)
    bg_ratio = bg_pixels / total_pixels

    # Nếu nền đen chiếm ít hơn 5%, chứng tỏ ảnh bị zoom cận cảnh hoặc mất viền
    if bg_ratio < min_bg_ratio:
        return False  # Bỏ qua ảnh này

    # Kiểm tra thêm: Nhãn cầu không được chạm hoàn toàn vào cả 4 cạnh
    # (Tùy chọn, tỷ lệ nền đen ở trên thường là đủ mạnh rồi)

    return True  # Ảnh hợp lệ