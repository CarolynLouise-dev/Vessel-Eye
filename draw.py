import cv2
import numpy as np
from constant import THRESHOLD_TORT_LOCAL, THRESHOLD_NARROW_LOCAL

try:
    from skan import Skeleton, summarize
    _SKAN_AVAILABLE = True
except ImportError:
    _SKAN_AVAILABLE = False

# ── Colormap helpers ───────────────────────────────────────────────────────────

def _apply_colormap_jet(value_norm):
    """Chuyển giá trị 0..1 sang màu BGR theo jet colormap (xanh→vàng→đỏ)."""
    v = float(np.clip(value_norm, 0.0, 1.0))
    if v < 0.25:
        r, g, b = 0, int(v * 4 * 255), 255
    elif v < 0.5:
        r, g, b = 0, 255, int((1 - (v - 0.25) * 4) * 255)
    elif v < 0.75:
        r, g, b = int((v - 0.5) * 4 * 255), 255, 0
    else:
        r, g, b = 255, int((1 - (v - 0.75) * 4) * 255), 0
    return (b, g, r)  # BGR


def _make_colorbar(height, width=28, min_val=0.0, max_val=1.0,
                   label_min="Nhỏ", label_max="Lớn"):
    """Tạo thanh màu dọc với nhãn min/max."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        norm = 1.0 - float(y) / max(1, height - 1)
        c = _apply_colormap_jet(norm)
        bar[y, :] = c
    # Labels are rendered by caller since we can't use Vietnamese in putText reliably
    return bar


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_discontinuity_map(skeleton_bin, vessel_mask):
    """Gap map = closing(skeleton) - skeleton, lọc noise."""
    if skeleton_bin is None:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    sk = (skeleton_bin > 0).astype(np.uint8)
    if sk.sum() == 0:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sk_u8 = (sk * 255).astype(np.uint8)

    closed = cv2.morphologyEx(sk_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    near_vessel = cv2.dilate((vessel_mask > 0).astype(np.uint8) * 255,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                             iterations=1)
    gap_raw = ((closed > 0) & (sk_u8 == 0) & (near_vessel > 0)).astype(np.uint8) * 255

    endpoint = _compute_endpoint_map(sk_u8, np.ones_like(sk_u8, dtype=np.uint8) * 255)
    endpoint_near = cv2.dilate(endpoint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    gap = cv2.bitwise_and(gap_raw, endpoint_near)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
    filtered = np.zeros_like(gap)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            filtered[lbl == i] = 255
    gap = filtered

    score = float(np.count_nonzero(gap)) / max(1.0, float(np.count_nonzero(sk_u8 > 0)))
    return gap, score


def _compute_endpoint_map(skeleton_bin, fov_mask):
    if skeleton_bin is None:
        return np.zeros_like(fov_mask, dtype=np.uint8)
    sk = (skeleton_bin > 0).astype(np.uint8)
    k = np.ones((3, 3), dtype=np.uint8)
    neigh_count = cv2.filter2D(sk, ddepth=cv2.CV_16U, kernel=k, borderType=cv2.BORDER_CONSTANT)
    out = (((sk == 1) & (neigh_count == 2) & (fov_mask > 0)).astype(np.uint8) * 255)
    return out


def _cross_section_diameter(binary_mask, y, x, angle_rad, half_width=15):
    """Đo đường kính mạch tại (y,x) bằng mặt cắt perpendicular với skeleton."""
    h, w = binary_mask.shape
    perp = angle_rad + np.pi / 2.0
    dy, dx = np.sin(perp), np.cos(perp)
    count = 0
    for d in range(-half_width, half_width + 1):
        ny = int(round(y + dy * d))
        nx = int(round(x + dx * d))
        if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] > 0:
            count += 1
    return count


def _build_skeleton_segments(skeleton_bin, binary_mask, en_green,
                              brightness_threshold,
                              img_bgr=None, av_model=None):
    """
    Trích xuất từng đoạn skeleton, phân loại A/V, đo đường kính.
    Trả về list dict: { cy, cx, tort, diam, is_artery, path_coords }
    """
    segments = []
    if not _SKAN_AVAILABLE or skeleton_bin is None:
        return segments

    skel_bool = (skeleton_bin > 0)
    if skel_bool.sum() < 5:
        return segments

    _av_predict = None
    if av_model is not None and img_bgr is not None:
        try:
            from av_classifier import predict_av_segment as _fn
            _av_predict = _fn
        except ImportError:
            pass

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sk = Skeleton(skel_bool)
            stats = summarize(sk, separator="-")
        except Exception:
            return segments

    col_arc = "branch-distance"
    col_ec = "euclidean-distance" if "euclidean-distance" in stats.columns else None

    for _, row in stats.iterrows():
        arc_len = float(row[col_arc]) if col_arc in stats.columns else 0.0
        if arc_len < 30:
            continue

        if col_ec and not np.isnan(row[col_ec]):
            chord_len = float(row[col_ec])
        else:
            try:
                src_c = sk.coordinates[int(row["node-id-src"])]
                dst_c = sk.coordinates[int(row["node-id-dst"])]
                chord_len = float(np.linalg.norm(np.array(src_c) - np.array(dst_c)))
            except Exception:
                chord_len = arc_len

        tort = arc_len / chord_len if chord_len > 1 else 1.0

        try:
            path = sk.path_coordinates(row.name)
            if len(path) < 3:
                continue
            mid = path[len(path) // 2]
            my, mx = int(mid[0]), int(mid[1])

            p_b = path[max(0, len(path) // 2 - 2)]
            p_a = path[min(len(path) - 1, len(path) // 2 + 2)]
            angle = np.arctan2(float(p_a[0] - p_b[0]), float(p_a[1] - p_b[1]))

            diam = _cross_section_diameter(binary_mask, my, mx, angle)
            if diam < 1:
                continue

            if _av_predict is not None:
                is_artery = _av_predict(av_model, img_bgr, en_green, binary_mask, path)
            else:
                h, w = en_green.shape
                avg_int = float(en_green[my, mx]) if 0 <= my < h and 0 <= mx < w else brightness_threshold
                is_artery = avg_int > brightness_threshold

            segments.append({
                "cy": my, "cx": mx,
                "tort": tort,
                "diam": diam,
                "is_artery": is_artery,
                "path": path,
            })
        except Exception:
            continue

    return segments


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 1 — Optic Disc Detection
# ══════════════════════════════════════════════════════════════════════════════

def draw_optic_disc_vis(img_bgr, od_center, od_radius):
    """Vẽ vòng tròn optic disc và zone B lên ảnh fundus."""
    vis = img_bgr.copy()
    cx, cy = int(od_center[0]), int(od_center[1])
    r = int(od_radius)

    # Zone B annular ring (đây là vùng phân tích chính)
    cv2.circle(vis, (cx, cy), r * 2, (0, 255, 180), 2)
    # Optic disc boundary
    cv2.circle(vis, (cx, cy), r, (0, 165, 255), 2)
    # Center dot
    cv2.circle(vis, (cx, cy), 4, (0, 165, 255), -1)

    # Labels
    cv2.putText(vis, "OD", (cx - 14, cy - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "Zone B", (cx - 32, cy - r * 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA)

    # Thêm crosshair nhỏ tại tâm
    cv2.line(vis, (cx - 8, cy), (cx + 8, cy), (0, 165, 255), 1)
    cv2.line(vis, (cx, cy - 8), (cx, cy + 8), (0, 165, 255), 1)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 2 — A/V Calibre Heat-map
# ══════════════════════════════════════════════════════════════════════════════

def draw_av_calibre_map(skeleton_bin, vessel_mask, en_green,
                        fov_mask=None, img_bgr=None, av_model=None):
    """
    Panel 2: Bản đồ đường kính (calibre) A/V trên nền đen.
    Màu từ xanh → vàng → đỏ biểu thị đường kính từ nhỏ → lớn.
    Động mạch: nhánh đỏ-cam; Tĩnh mạch: nhánh xanh-lục.
    """
    if skeleton_bin is None or vessel_mask is None:
        h, w = vessel_mask.shape if vessel_mask is not None else (512, 512)
        return np.zeros((h, w, 3), dtype=np.uint8)

    h, w = vessel_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Background mờ (outline vessel)
    vessel_bg = np.zeros((h, w, 3), dtype=np.uint8)
    vessel_bg[vessel_mask > 0] = (15, 20, 15)
    vis = cv2.add(vis, vessel_bg)

    if en_green is None or not _SKAN_AVAILABLE:
        # Fallback: chỉ vẽ skeleton trắng
        sk_u8 = (skeleton_bin > 0).astype(np.uint8) * 255
        vis[sk_u8 > 0] = (200, 200, 200)
        return vis

    vessel_pixels = en_green[vessel_mask > 0]
    brightness_threshold = np.median(vessel_pixels) if len(vessel_pixels) > 0 else 128.0

    segments = _build_skeleton_segments(
        skeleton_bin, vessel_mask, en_green, brightness_threshold,
        img_bgr=img_bgr, av_model=av_model
    )

    if not segments:
        sk_u8 = (skeleton_bin > 0).astype(np.uint8) * 255
        vis[sk_u8 > 0] = (150, 200, 150)
        return vis

    # Chuẩn hóa đường kính để map màu
    all_diams = [s["diam"] for s in segments]
    d_min = max(1.0, float(np.percentile(all_diams, 5)))
    d_max = max(d_min + 1.0, float(np.percentile(all_diams, 95)))

    artery_segs = [s for s in segments if s["is_artery"]]
    vein_segs = [s for s in segments if not s["is_artery"]]

    def draw_segment_calibre(seg, palette_offset=0.0):
        """Vẽ từng điểm trên path với màu theo đường kính."""
        path = seg["path"]
        diam_norm = np.clip((seg["diam"] - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)

        # Artery: dải màu warm (đỏ-cam-vàng); Vein: dải màu cool (xanh-tím-lam)
        if seg["is_artery"]:
            # Jet shifted to warm: 0.5→1.0
            norm_shifted = 0.5 + diam_norm * 0.5
        else:
            # Jet shifted to cool: 0.0→0.5
            norm_shifted = diam_norm * 0.5

        color = _apply_colormap_jet(norm_shifted)

        # Độ dày nét vẽ theo đường kính
        thickness = max(1, min(4, int(seg["diam"] / 3)))
        pts = [(int(pt[1]), int(pt[0])) for pt in path]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    for seg in artery_segs:
        draw_segment_calibre(seg)
    for seg in vein_segs:
        draw_segment_calibre(seg)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    # Thêm nhãn nhỏ A/V
    _draw_legend_av(vis, h, w)

    return vis


def _draw_legend_av(vis, h, w):
    """Vẽ legend nhỏ tại góc ảnh."""
    x0, y0 = 8, h - 48
    cv2.rectangle(vis, (x0 - 2, y0 - 2), (x0 + 110, y0 + 40), (30, 30, 30), -1)
    # Artery sample
    cv2.line(vis, (x0, y0 + 8), (x0 + 20, y0 + 8), (0, 80, 255), 3)
    cv2.putText(vis, "Artery (A)", (x0 + 24, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 180, 180), 1, cv2.LINE_AA)
    # Vein sample
    cv2.line(vis, (x0, y0 + 28), (x0 + 20, y0 + 28), (255, 120, 30), 3)
    cv2.putText(vis, "Vein (V)", (x0 + 24, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 180), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 3 — Đứt đoạn mạch máu
# ══════════════════════════════════════════════════════════════════════════════

def draw_discontinuity_map(skeleton, vessel_mask, fov_mask):
    """
    Panel 3: Skeleton xanh lá + Gap đứt đoạn nổi bật màu đỏ-cam.
    Vòng tròn khoanh vùng từng điểm đứt đoạn rõ ràng.
    """
    h, w = vessel_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    if skeleton is None:
        return vis

    sk = (skeleton > 0).astype(np.uint8)
    if sk.sum() == 0:
        return vis

    sk_u8 = (sk * 255).astype(np.uint8)

    # 1. Background: vessel mask mờ (dark gray context)
    vis[vessel_mask > 0] = (20, 30, 20)

    # 2. Skeleton: xanh lá sáng, độ dày 1px → dilate nhẹ cho dễ nhìn
    sk_disp = cv2.dilate(sk_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    vis[sk_disp > 0] = (0, 230, 60)

    # 3. Morphological closing để tìm gap
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(sk_u8, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    near_vessel = cv2.dilate(
        (vessel_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )
    gap_raw = ((closed > 0) & (sk_u8 == 0) & (near_vessel > 0)).astype(np.uint8) * 255

    # 4. Chỉ giữ gap gần endpoint
    endpoint = _compute_endpoint_map(sk_u8, np.ones_like(sk_u8) * 255)
    endpoint_near = cv2.dilate(endpoint,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=1)
    gap = cv2.bitwise_and(gap_raw, endpoint_near)

    # 5. Lọc blob nhỏ (noise) — siết chặt hơn
    n_lbl, lbl_img, stats_cc, centroids = cv2.connectedComponentsWithStats(gap, connectivity=8)
    gap_filtered = np.zeros_like(gap)
    gap_centers = []
    for i in range(1, n_lbl):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if area >= 18:  # Tăng từ 4 → 18 để lọc noise tốt hơn
            gap_filtered[lbl_img == i] = 255
            cx_g = int(centroids[i][0])
            cy_g = int(centroids[i][1])
            gap_centers.append((cx_g, cy_g, area))

    # 6. Tô màu gap
    vis[gap_filtered > 0] = (50, 80, 255)

    # 7. Chỉ khoanh vòng tròn TOP-20 gap lớn nhất (tránh nhiễu)
    gap_centers.sort(key=lambda x: -x[2])
    for (gx, gy, area) in gap_centers[:20]:
        r = max(10, min(22, int(np.sqrt(area) * 1.8)))
        cv2.circle(vis, (gx, gy), r + 4, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (gx, gy), r, (0, 60, 220), 1, cv2.LINE_AA)
        cv2.circle(vis, (gx, gy), 3, (0, 200, 255), -1)

    # 8. Đánh dấu endpoint (đầu mút skeleton) — giới hạn số lượng
    ep_coords = np.where(endpoint > 0)
    ep_list = list(zip(ep_coords[0], ep_coords[1]))
    # Chỉ hiển thị endpoint nằm gần gap (tránh nhiều chấm thừa)
    gap_dilated = cv2.dilate(gap_filtered,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    for ey, ex in ep_list[:200]:
        if fov_mask is not None and fov_mask[ey, ex] == 0:
            continue
        if gap_dilated[ey, ex] > 0:  # Chỉ vẽ endpoint gần gap
            cv2.circle(vis, (ex, ey), 4, (0, 255, 200), 1, cv2.LINE_AA)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 3 — Phân đoạn mạch máu B&W cực rõ
# ══════════════════════════════════════════════════════════════════════════════

def draw_vessel_segmentation(vessel_mask, en_green=None, fov_mask=None):
    """
    Panel 3 (B&W): Ảnh phân đoạn mạch máu cực rõ ràng.
    - Xóa nhiễu nhỏ (CC filter)
    - Đóng kín các khe hở nhỏ (morphological close)
    - Làm dày nhẹ để dễ quan sát
    - White vessels trên black background
    """
    if vessel_mask is None:
        return np.zeros((512, 512), dtype=np.uint8)

    h, w = vessel_mask.shape

    # ── 1. Binary base ──────────────────────────────────────────────────────
    binary = (vessel_mask > 0).astype(np.uint8) * 255

    # ── 2. Close nhỏ để nối các đoạn đứt mảnh ──────────────────────────────
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=2)

    # ── 3. CC filter: bỏ ~noise nhỏ ────────────────────────────────────────
    fov_px = float(np.count_nonzero(fov_mask)) if fov_mask is not None else float(h * w)
    min_area = max(60, int(fov_px * 0.00007))
    n_lbl, lbl_img, stats_cc, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    filtered = np.zeros_like(closed)
    for i in range(1, n_lbl):
        if stats_cc[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[lbl_img == i] = 255

    # ── 4. Dày nhẹ để dễ nhìn trên màn hình nhỏ ─────────────────────────────
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thickened = cv2.dilate(filtered, k2, iterations=1)

    # ── 5. Nếu có en_green: blend thêm đường gradient để tăng độ rõ ─────────
    if en_green is not None:
        # Làm nổi bật mạch từ green channel bằng cách overlay lên mask
        g_norm = cv2.normalize(en_green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # CLAHE local contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        g_eq = clahe.apply(g_norm)
        # Chỉ giữ phần nằm trong vessel mask
        vessel_intensity = cv2.bitwise_and(g_eq, g_eq,
                                           mask=(thickened > 0).astype(np.uint8))
        # Blend: 60% intensity + 40% pure binary
        blend = cv2.addWeighted(vessel_intensity, 0.5, thickened, 0.8, 0)
        _, final = cv2.threshold(blend, 40, 255, cv2.THRESH_BINARY)
    else:
        final = thickened

    if fov_mask is not None:
        final = cv2.bitwise_and(final, fov_mask)

    return final  # uint8 2D grayscale — caller dùng is_gray=True


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 4 — Heat-map Hẹp/Phồng mạch
# ══════════════════════════════════════════════════════════════════════════════

def draw_diameter_heatmap(skeleton_bin, vessel_mask, en_green,
                          fov_mask=None, img_bgr=None, av_model=None):
    """
    Panel 4: Mỗi điểm trên skeleton được tô màu theo đường kính thực đo.
    Xanh = mạch nhỏ/hẹp bất thường, Vàng = bình thường, Đỏ = phồng/giãn bất thường.
    Đánh dấu THÊM:
      - Vòng tròn đỏ kép: điểm HẸP bất thường (arterial narrowing)
      - Mũi tên vàng nhạt: điểm rộng bất thường (venous dilation)
    """
    if skeleton_bin is None or vessel_mask is None:
        h_fb = vessel_mask.shape[0] if vessel_mask is not None else 512
        w_fb = vessel_mask.shape[1] if vessel_mask is not None else 512
        return np.zeros((h_fb, w_fb, 3), dtype=np.uint8)

    h, w = vessel_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[vessel_mask > 0] = (10, 15, 10)  # background cực mờ

    if not _SKAN_AVAILABLE or skeleton_bin is None:
        sk_u8 = (skeleton_bin > 0).astype(np.uint8) * 255
        vis[sk_u8 > 0] = (150, 200, 150)
        return vis

    vessel_pixels = en_green[vessel_mask > 0] if en_green is not None else np.array([128.0])
    brightness_threshold = float(np.median(vessel_pixels)) if len(vessel_pixels) > 0 else 128.0

    segments = _build_skeleton_segments(
        skeleton_bin, vessel_mask, en_green, brightness_threshold,
        img_bgr=img_bgr, av_model=av_model
    )

    if not segments:
        sk_u8 = (skeleton_bin > 0).astype(np.uint8) * 255
        vis[sk_u8 > 0] = (100, 180, 100)
        return vis

    a_diams = [s["diam"] for s in segments if s["is_artery"]]
    v_diams = [s["diam"] for s in segments if not s["is_artery"]]
    avg_a = float(np.mean(a_diams)) if a_diams else 4.0
    avg_v = float(np.mean(v_diams)) if v_diams else 6.0
    std_a = float(np.std(a_diams)) if len(a_diams) > 1 else 1.0
    std_v = float(np.std(v_diams)) if len(v_diams) > 1 else 1.5

    narrow_markers = []   # (cx, cy) điểm hẹp bất thường
    wide_markers = []     # (cx, cy) điểm phồng bất thường

    for seg in segments:
        path = seg["path"]
        diam = seg["diam"]
        is_artery = seg["is_artery"]

        if is_artery:
            ref_mean, ref_std = avg_a, std_a
            # HẸP: < mean - 1.5*std; PHỒNG: > mean + 1.5*std
            is_narrow = diam < (ref_mean - 1.5 * ref_std) and diam < avg_v * THRESHOLD_NARROW_LOCAL
            is_wide = diam > (ref_mean + 2.0 * ref_std)
        else:
            ref_mean, ref_std = avg_v, std_v
            is_narrow = diam < (ref_mean - 1.8 * ref_std)
            is_wide = diam > (ref_mean + 2.0 * ref_std)

        # Normalize diameter cho màu: 0=hẹp nhất (xanh), 1=rộng nhất (đỏ)
        all_ref = ref_mean
        spread = max(ref_std * 3, 2.0)
        norm_d = np.clip((diam - (all_ref - spread)) / (2 * spread), 0.0, 1.0)
        color = _apply_colormap_jet(float(norm_d))

        # Vẽ từng segment
        thickness = max(1, min(5, int(diam / 2.5)))
        pts = [(int(pt[1]), int(pt[0])) for pt in path]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

        # Ghi nhận vị trí bất thường
        cx_s, cy_s = seg["cx"], seg["cy"]
        if is_narrow:
            narrow_markers.append((cx_s, cy_s, diam))
        if is_wide:
            wide_markers.append((cx_s, cy_s, diam))

    # Vẽ marker HẸP: vòng kép đỏ sáng
    for (mx, my, diam) in narrow_markers:
        if not (0 <= my < h and 0 <= mx < w):
            continue
        if fov_mask is not None and fov_mask[my, mx] == 0:
            continue
        cv2.circle(vis, (mx, my), 16, (0, 50, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (mx, my), 22, (0, 100, 255), 1, cv2.LINE_AA)
        # Label ngắn
        cv2.putText(vis, "HEP", (mx + 14, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 150, 255), 1, cv2.LINE_AA)

    # Vẽ marker PHỒNG: vòng vàng
    for (mx, my, diam) in wide_markers:
        if not (0 <= my < h and 0 <= mx < w):
            continue
        if fov_mask is not None and fov_mask[my, mx] == 0:
            continue
        cv2.circle(vis, (mx, my), 18, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "PHONG", (mx + 14, my + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 220, 200), 1, cv2.LINE_AA)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    # Colorbar bên phải
    bar_h = min(h - 20, 180)
    bar = _make_colorbar(bar_h)
    bx = w - 36
    by = (h - bar_h) // 2
    if bx > 0 and by >= 0 and by + bar_h <= h:
        vis[by:by + bar_h, bx:bx + 28] = bar
        cv2.putText(vis, "RONG", (bx - 2, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 200, 255), 1)
        cv2.putText(vis, "HEP", (bx + 2, by + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 200, 255), 1)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Bản đồ lâm sàng tổng hợp (Clinical Feature Map)
# ══════════════════════════════════════════════════════════════════════════════

def draw_feature_map(img_disp, vessel_mask, en_green, regions,
                     img_no_bg=None, fov_mask=None, skeleton=None,
                     img_bgr=None, av_model=None, return_debug=False,
                     anatomy_details=None):
    """
    Bản đồ lâm sàng tổng hợp:
    - Nền: ảnh fundus làm mờ (để marker nổi bật hơn)
    - Mạch A/V tô màu đỏ/xanh
    - Marker HẸP (Narrowing): vòng kép đỏ lớn + nhãn
    - Marker XOẮN (Tortuosity): vòng xanh lá + nhãn
    - Marker ĐỨT (Discontinuity): vòng cam + nhãn
    - OD + Zone B
    - Vùng nguy hiểm cao: overlay đỏ bán trong suốt
    """
    if img_no_bg is None:
        img_no_bg = img_disp

    h, w = img_no_bg.shape[:2]

    if fov_mask is None:
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Làm mờ nền fundus để marker nổi bật hơn (×0.55 thay vì ×1.0)
    result_img = cv2.addWeighted(img_no_bg, 0.55, np.zeros_like(img_no_bg), 0.0, 0)
    sign_img = img_no_bg.copy()

    vessel_pixels = en_green[vessel_mask > 0]
    if len(vessel_pixels) == 0:
        if return_debug:
            empty = np.zeros((h, w, 3), dtype=np.uint8)
            return result_img, {"discontinuity_map": empty, "sign_map": result_img}
        return result_img

    brightness_threshold = np.median(vessel_pixels)
    gap_map, discontinuity_score = _compute_discontinuity_map(skeleton, vessel_mask)

    if _SKAN_AVAILABLE and skeleton is not None:
        segments = _build_skeleton_segments(
            skeleton, vessel_mask, en_green, brightness_threshold,
            img_bgr=img_bgr, av_model=av_model
        )

        if segments:
            v_diams = [s["diam"] for s in segments if not s["is_artery"]]
            a_diams = [s["diam"] for s in segments if s["is_artery"]]
            avg_v_diam = float(np.mean(v_diams)) if v_diams else 4.0
            avg_a_diam = float(np.mean(a_diams)) if a_diams else 4.0
            std_a = float(np.std(a_diams)) if len(a_diams) > 1 else 1.0

            # Map A/V màu lên ảnh
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for seg in segments:
                color = (0, 0, 220) if seg["is_artery"] else (220, 60, 0)
                for pt in seg["path"]:
                    py, px = int(pt[0]), int(pt[1])
                    if 0 <= py < h and 0 <= px < w:
                        color_mask[py, px] = color

            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=fov_mask)
            # Dày mạch lên để nhìn rõ hơn
            color_mask = cv2.dilate(color_mask,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            result_img = cv2.addWeighted(result_img, 1.0, color_mask, 0.6, 0)

            # ── Danger zone: cluster markers ──────────────────────────────────
            danger_map = np.zeros((h, w), dtype=np.uint8)

            for seg in segments:
                cy_s, cx_s = seg["cy"], seg["cx"]
                if not (0 <= cy_s < h and 0 <= cx_s < w):
                    continue
                if fov_mask[cy_s, cx_s] == 0:
                    continue

                # ── Marker XOẮN VẶN (Tortuosity) ─────────────────────────
                if seg["tort"] > THRESHOLD_TORT_LOCAL:
                    # Vòng xanh lá sáng
                    cv2.circle(result_img, (cx_s, cy_s), 18, (0, 255, 80), 2, cv2.LINE_AA)
                    cv2.circle(sign_img, (cx_s, cy_s), 18, (0, 255, 80), 2, cv2.LINE_AA)
                    cv2.putText(result_img, f"XOAN {seg['tort']:.1f}",
                                (cx_s + 20, cy_s - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80), 1, cv2.LINE_AA)
                    cv2.circle(danger_map, (cx_s, cy_s), 30, 60, -1)

                # ── Marker HẸP ĐỘNG MẠCH (Narrowing) ─────────────────────
                if seg["is_artery"] and avg_v_diam > 0:
                    narrow_thresh = THRESHOLD_NARROW_LOCAL
                    ratio = seg["diam"] / max(1.0, avg_v_diam)
                    if ratio < narrow_thresh:
                        # Vòng kép đỏ-cam
                        cv2.circle(result_img, (cx_s, cy_s), 20, (0, 60, 255), 2, cv2.LINE_AA)
                        cv2.circle(result_img, (cx_s, cy_s), 28, (0, 120, 255), 1, cv2.LINE_AA)
                        cv2.circle(sign_img, (cx_s, cy_s), 20, (0, 60, 255), 2, cv2.LINE_AA)
                        cv2.putText(result_img, f"HEP {seg['diam']}px",
                                    (cx_s + 22, cy_s + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 150, 255), 1, cv2.LINE_AA)
                        cv2.circle(danger_map, (cx_s, cy_s), 35, 80, -1)

            # ── Overlay gap đứt đoạn (cam) ────────────────────────────────
            gap_overlay = np.zeros_like(result_img, dtype=np.uint8)
            gap_overlay[:, :, 1] = (gap_map > 0).astype(np.uint8) * 140
            gap_overlay[:, :, 2] = (gap_map > 0).astype(np.uint8) * 255
            result_img = cv2.addWeighted(result_img, 1.0, gap_overlay, 0.55, 0)

            # ── Vòng tròn ĐỨT ĐOẠN ────────────────────────────────────────
            n_lbl_g, lbl_g, stats_g, cents_g = cv2.connectedComponentsWithStats(
                gap_map, connectivity=8)
            for i in range(1, n_lbl_g):
                area_g = stats_g[i, cv2.CC_STAT_AREA]
                if area_g < 20:
                    continue
                gx_c = int(cents_g[i][0])
                gy_c = int(cents_g[i][1])
                r_g = max(12, min(26, int(np.sqrt(area_g) * 2.0)))
                cv2.circle(result_img, (gx_c, gy_c), r_g, (0, 160, 255), 2, cv2.LINE_AA)
                cv2.circle(sign_img, (gx_c, gy_c), r_g, (0, 160, 255), 2, cv2.LINE_AA)
                cv2.putText(result_img, "DUT",
                            (gx_c + r_g + 2, gy_c + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.circle(danger_map, (gx_c, gy_c), 40, 90, -1)

            # ── Danger zone overlay (đỏ bán trong suốt) ───────────────────
            danger_blur = cv2.GaussianBlur(danger_map, (61, 61), 0)
            danger_colored = np.zeros((h, w, 3), dtype=np.uint8)
            danger_colored[:, :, 2] = danger_blur  # Red channel
            result_img = cv2.addWeighted(result_img, 1.0, danger_colored, 0.30, 0)

            # ── Zone B + OD marker ─────────────────────────────────────────
            if anatomy_details:
                cx_od, cy_od = anatomy_details.get("od_center", (None, None))
                od_r = anatomy_details.get("od_radius", None)
                if cx_od is not None and cy_od is not None and od_r is not None:
                    cv2.circle(result_img, (int(cx_od), int(cy_od)), int(od_r), (0, 165, 255), 2)
                    cv2.circle(result_img, (int(cx_od), int(cy_od)), int(2 * od_r), (0, 230, 150), 1)
                    cv2.putText(result_img, "OD", (int(cx_od) - 10, int(cy_od) - int(od_r) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.circle(sign_img, (int(cx_od), int(cy_od)), int(od_r), (0, 165, 255), 1)
                    cv2.circle(sign_img, (int(cx_od), int(cy_od)), int(2 * od_r), (0, 230, 150), 1)

            if return_debug:
                gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
                gap_vis[gap_map > 0] = (0, 140, 255)
                gap_vis = cv2.bitwise_and(gap_vis, gap_vis, mask=fov_mask)
                return result_img, {"discontinuity_map": gap_vis, "sign_map": sign_img}

            return result_img

    # Fallback: regionprops
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    v_diams_fb = []
    for r in regions:
        if r.area < 100:
            continue
        coords = r.coords
        intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])
        if intensity <= brightness_threshold:
            v_diams_fb.append(r.axis_minor_length)
    avg_v_diam_fb = np.mean(v_diams_fb) if v_diams_fb else 2.0

    for reg in regions:
        if reg.area < 100:
            continue
        coords = reg.coords
        center_y, center_x = map(int, reg.centroid)
        if fov_mask[center_y, center_x] == 0:
            continue
        avg_intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])
        is_artery = avg_intensity > brightness_threshold
        color = (0, 0, 220) if is_artery else (220, 60, 0)
        rr, cc = coords[:, 0], coords[:, 1]
        color_mask[rr, cc] = color
        t_score = (reg.perimeter ** 2) / (4 * np.pi * reg.area) if reg.area > 0 else 1.0
        if t_score > THRESHOLD_TORT_LOCAL:
            cv2.circle(result_img, (center_x, center_y), 18, (0, 255, 80), 2, cv2.LINE_AA)
            cv2.circle(sign_img, (center_x, center_y), 18, (0, 255, 80), 2, cv2.LINE_AA)
        if is_artery and avg_v_diam_fb > 0 and (reg.axis_minor_length / avg_v_diam_fb) < THRESHOLD_NARROW_LOCAL:
            cv2.circle(result_img, (center_x, center_y), 20, (0, 60, 255), 2, cv2.LINE_AA)
            cv2.circle(sign_img, (center_x, center_y), 20, (0, 60, 255), 2, cv2.LINE_AA)

    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=fov_mask)
    result_img = cv2.addWeighted(result_img, 1.0, color_mask, 0.4, 0)

    # Gap overlay
    gap_overlay = np.zeros_like(result_img, dtype=np.uint8)
    gap_overlay[:, :, 1] = (gap_map > 0).astype(np.uint8) * 140
    gap_overlay[:, :, 2] = (gap_map > 0).astype(np.uint8) * 255
    result_img = cv2.addWeighted(result_img, 1.0, gap_overlay, 0.55, 0)

    if anatomy_details:
        cx_od, cy_od = anatomy_details.get("od_center", (None, None))
        od_r = anatomy_details.get("od_radius", None)
        if cx_od is not None and cy_od is not None and od_r is not None:
            cv2.circle(result_img, (int(cx_od), int(cy_od)), int(od_r), (0, 165, 255), 2)
            cv2.circle(result_img, (int(cx_od), int(cy_od)), int(2 * od_r), (0, 230, 150), 1)

    if return_debug:
        gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gap_vis[gap_map > 0] = (0, 140, 255)
        gap_vis = cv2.bitwise_and(gap_vis, gap_vis, mask=fov_mask)
        return result_img, {"discontinuity_map": gap_vis, "sign_map": sign_img}

    return result_img


# ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED (kept for backward compat)
# ══════════════════════════════════════════════════════════════════════════════

def draw_closing_vis(skeleton, vessel_mask, fov_mask):
    """Backward compat — gọi draw_discontinuity_map."""
    return draw_discontinuity_map(skeleton, vessel_mask, fov_mask)