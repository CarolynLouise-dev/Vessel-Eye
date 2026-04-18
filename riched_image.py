import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
import warnings

try:
    from skimage.filters import apply_hysteresis_threshold
    _HYSTERESIS_AVAILABLE = True
except ImportError:
    _HYSTERESIS_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# PIPELINE TĂNG CƯỜNG + SEGMENTATION
# ==========================================


def _ensure_odd(value):
    value = int(round(value))
    return value if value % 2 == 1 else value + 1


def _build_fov_mask(img):
    """Tách Field of View robust: threshold + ellipse fitting để loại bỏ viền sáng ngoài."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Bước 1: threshold kép để tìm vùng võng mạc ──────────────────────────
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, mask_low = cv2.threshold(blur, 12, 255, cv2.THRESH_BINARY)
    _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fov_mask = cv2.bitwise_or(mask_low, mask_otsu)

    # ── Bước 2: giữ component lớn nhất ──────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fov_mask, connectivity=8)
    if num_labels > 1:
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fov_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    k_size = _ensure_odd(max(21, min(h, w) * 0.03))
    k_fov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, k_fov)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, k_fov)

    # ── Bước 3: Ellipse fitting để tạo mask tròn chính xác ──────────────────
    # Tìm contour lớn nhất và fit ellipse để loại bỏ viền sáng bên ngoài
    contours, _ = cv2.findContours(fov_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        if len(largest_cnt) >= 5:  # fitEllipse cần tối thiểu 5 điểm
            try:
                ellipse = cv2.fitEllipse(largest_cnt)
                ell_center, ell_axes, ell_angle = ellipse
                # Shrink ellipse nhẹ (~4%) để cắt đứt viền sáng ngoài võng mạc
                shrink = 0.96
                ell_axes_new = (ell_axes[0] * shrink, ell_axes[1] * shrink)

                fov_ellipse = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(fov_ellipse,
                            (int(ell_center[0]), int(ell_center[1])),
                            (int(ell_axes_new[0] / 2), int(ell_axes_new[1] / 2)),
                            ell_angle, 0, 360, 255, -1)

                # Kết hợp: chỉ giữ phần nằm trong cả FOV ban đầu lẫn ellipse
                fov_combined = cv2.bitwise_and(fov_mask, fov_ellipse)

                # Kiểm tra: nếu ellipse mask hợp lý (>60% diện tích FOV cũ) thì dùng
                ratio = float(np.count_nonzero(fov_combined)) / max(1.0, float(np.count_nonzero(fov_mask)))
                if ratio > 0.60:
                    fov_mask = fov_combined
            except cv2.error:
                pass  # Fallback về mask cũ nếu fitEllipse thất bại

    return fov_mask


def _normalize_in_mask(channel, fov_mask, lower=2.0, upper=98.0):
    """Chuẩn hóa intensity chỉ trong FOV để ảnh tối/sáng khác nhau về cùng miền."""
    vals = channel[fov_mask > 0].astype(np.float32)
    if vals.size < 32:
        return cv2.bitwise_and(channel, channel, mask=fov_mask)

    p_low, p_high = np.percentile(vals, [lower, upper])
    if p_high - p_low < 1e-6:
        out = channel.copy()
    else:
        out = ((channel.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)))
        out = np.clip(out, 0, 255).astype(np.uint8)

    out[fov_mask == 0] = 0
    return out


def _gamma_normalize(channel, fov_mask, target_mean=128.0):
    """Đưa mean intensity về mức ổn định hơn trước khi Frangi chạy."""
    vals = channel[fov_mask > 0].astype(np.float32)
    if vals.size < 32:
        return channel

    mean_val = float(np.mean(vals)) / 255.0
    target = float(target_mean) / 255.0
    if not (0.01 < mean_val < 0.99):
        return channel

    gamma = np.log(target) / np.log(mean_val + 1e-6)
    gamma = float(np.clip(gamma, 0.7, 1.6))

    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(channel, lut)
    out[fov_mask == 0] = 0
    return out


def _correct_illumination(green, fov_mask):
    """Ước lượng nền chiếu sáng và loại bỏ gradient sáng tối trong cùng ảnh."""
    h, w = green.shape[:2]
    bg_size = _ensure_odd(max(31, min(h, w) * 0.10))
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_size, bg_size))

    green_norm = _normalize_in_mask(green, fov_mask, lower=2.0, upper=99.0)
    background = cv2.morphologyEx(green_norm, cv2.MORPH_OPEN, bg_kernel)
    background = cv2.GaussianBlur(background, (bg_size, bg_size), 0)

    corrected = cv2.subtract(green_norm, background)
    bg_mean = float(np.mean(background[fov_mask > 0])) if np.any(fov_mask > 0) else 128.0
    corrected = np.clip(corrected.astype(np.float32) + 0.75 * bg_mean, 0, 255).astype(np.uint8)
    corrected = _normalize_in_mask(corrected, fov_mask, lower=1.0, upper=99.0)
    corrected = _gamma_normalize(corrected, fov_mask, target_mean=128.0)
    return corrected


def _hysteresis_vessel_threshold(resp_u8, fov_mask):
    """
    Ngưỡng kép kiểu Hysteresis (giống Canny):
    - High threshold: pixel mạch chắc chắn (seed)
    - Low threshold: pixel yếu nhưng nối liền với seed → giữ lại
    Kết quả: mạch liên tục hơn, ít đứt đoạn hơn so với ngưỡng đơn.
    """
    fov_vals = resp_u8[fov_mask > 0]
    positive = fov_vals[fov_vals > 0]
    if positive.size == 0:
        return np.zeros_like(resp_u8, dtype=np.uint8)

    # High threshold: chắc chắn là mạch (tương tự Otsu / percentile 88)
    high = float(np.percentile(positive, 88))
    # Low threshold: có thể là mạch nếu nối với pixel chắc chắn (percentile 72)
    low  = float(np.percentile(positive, 72))

    # Điều chỉnh density
    high_mask = (resp_u8.astype(np.float32) >= high) & (fov_mask > 0)
    density_high = float(np.count_nonzero(high_mask)) / max(1.0, float(np.count_nonzero(fov_mask)))
    if density_high < 0.03:   # Quá ít seed → hạ ngưỡng
        high = float(np.percentile(positive, 82))
        low  = float(np.percentile(positive, 65))
    elif density_high > 0.20:  # Quá nhiều → tăng ngưỡng
        high = float(np.percentile(positive, 92))
        low  = float(np.percentile(positive, 78))

    if _HYSTERESIS_AVAILABLE:
        resp_f = resp_u8.astype(np.float32)
        hyst = apply_hysteresis_threshold(resp_f, low, high)
        result = (hyst & (fov_mask > 0)).astype(np.uint8) * 255
    else:
        # Fallback: ngưỡng đơn với low threshold
        thresh = int(np.clip(low, 1, 255))
        result = ((resp_u8 >= thresh) & (fov_mask > 0)).astype(np.uint8) * 255

    return result


def _adaptive_vessel_threshold(resp_u8, fov_mask):
    """Fallback: chọn threshold ổn định giữa ảnh tối/sáng khác nhau."""
    fov_vals = resp_u8[fov_mask > 0]
    positive = fov_vals[fov_vals > 0]
    if positive.size == 0:
        return 1

    otsu_thresh, _ = cv2.threshold(positive.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    perc_thresh = np.percentile(positive, 84)
    thresh = int(np.clip(max(float(otsu_thresh), float(perc_thresh)), 1, 255))

    mask_try = (resp_u8 >= thresh).astype(np.uint8)
    density = float(np.count_nonzero(mask_try & (fov_mask > 0))) / max(1.0, float(np.count_nonzero(fov_mask)))

    if density < 0.05:
        thresh = int(max(1, np.percentile(positive, 80)))
    elif density > 0.18:
        thresh = int(min(255, np.percentile(positive, 90)))

    return thresh


def _filter_components(mask_u8, proc_mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    cleaned = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp_mask = labels == i
        overlap = np.count_nonzero(comp_mask & (proc_mask > 0)) / float(max(1, area))
        if overlap > 0.5:
            cleaned[comp_mask] = 255
    return cleaned


def get_enhanced_vessels(img, return_details=False):
    # ===== 1. Robust FOV mask =====
    fov_mask = _build_fov_mask(img)

    # Erode nhẹ vùng xử lý để tránh bắt nhầm viền sáng tròn của fundus.
    proc_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ensure_odd(max(9, min(img.shape[:2]) * 0.02)),) * 2)
    proc_mask = cv2.erode(fov_mask, proc_kernel, iterations=1)
    if np.count_nonzero(proc_mask) < 0.7 * np.count_nonzero(fov_mask):
        proc_mask = fov_mask.copy()

    # ===== 2. Xóa nền ngoài FOV =====
    img_no_bg = cv2.bitwise_and(img, img, mask=fov_mask)

    # ===== 3. Green channel + illumination correction + CLAHE =====
    green = img[:, :, 1]
    green_corrected = _correct_illumination(green, proc_mask)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g_eq = clahe.apply(green_corrected)
    en_green = cv2.bitwise_and(g_eq, g_eq, mask=proc_mask)

    # ===== 4. Multi-scale vessel enhancement =====
    g_norm = en_green.astype(np.float64) / 255.0
    frangi_ok = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            vessel_resp = frangi(
                g_norm,
                sigmas=tuple(np.arange(1, 9, 1)),
                alpha=0.5,
                beta=0.5,
                gamma=15,
                black_ridges=True,
            )
            frangi_ok = vessel_resp.max() > 0
        except Exception:
            frangi_ok = False

    g_smooth = cv2.GaussianBlur(en_green, (5, 5), 0)
    bh9 = cv2.morphologyEx(g_smooth, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bh15 = cv2.morphologyEx(g_smooth, cv2.MORPH_BLACKHAT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bh_resp = cv2.normalize(cv2.addWeighted(bh9, 0.6, bh15, 0.4, 0),
                            None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if frangi_ok:
        frangi_u8 = (vessel_resp / max(vessel_resp.max(), 1e-6) * 255).astype(np.uint8)
        resp_u8 = cv2.addWeighted(frangi_u8, 0.78, bh_resp, 0.22, 0)
    else:
        resp_u8 = bh_resp

    resp_u8 = _normalize_in_mask(resp_u8, proc_mask, lower=1.0, upper=99.5)
    resp_u8 = cv2.GaussianBlur(resp_u8, (3, 3), 0)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=proc_mask)

    # ===== 5. Hysteresis threshold (ngưỡng kép) =====
    # Giống kỹ thuật của chị trong team: high seed + low connect
    vessel_mask = _hysteresis_vessel_threshold(resp_u8, proc_mask)
    if np.count_nonzero(vessel_mask) < 100:  # Fallback nếu hysteresis thất bại
        thresh = _adaptive_vessel_threshold(resp_u8, proc_mask)
        vessel_mask = ((resp_u8 >= thresh) & (proc_mask > 0)).astype(np.uint8) * 255

    min_area = max(40, int(np.count_nonzero(proc_mask) * 0.00006))
    raw_vessel_mask = _filter_components(vessel_mask, proc_mask, min_area)

    # ===== 6. Morphology cleanup =====
    # Chỉ CLOSE để nối đoạn đứt nhỏ nhưng không làm mất mạch mảnh.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(raw_vessel_mask, cv2.MORPH_CLOSE, k3, iterations=1)

    # ===== 7. CC filter — area only + FOV overlap =====
    vessel_mask = _filter_components(vessel_mask, proc_mask, min_area)

    # ===== 8. Skeleton + prune =====
    skeleton = skeletonize(vessel_mask > 0).astype(np.uint8) * 255
    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=proc_mask)

    n_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8)
    sk_clean = np.zeros_like(skeleton)
    for i in range(1, n_s):
        if stats_s[i, cv2.CC_STAT_AREA] >= 12:
            sk_clean[lab_s == i] = 255
    skeleton = sk_clean

    if return_details:
        details = {
            "raw_vessel_mask": raw_vessel_mask,
        }
        return en_green, vessel_mask, skeleton, img_no_bg, proc_mask, details

    return en_green, vessel_mask, skeleton, img_no_bg, proc_mask