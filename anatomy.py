import cv2
import numpy as np


def detect_optic_disc(img_bgr, fov_mask=None, radius_ratio=0.08):
    """Detect optic disc from bright compact candidates inside FOV."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    v = hsv[:, :, 2]
    l = lab[:, :, 0]

    v_blur = cv2.GaussianBlur(v, (41, 41), 0)
    l_blur = cv2.GaussianBlur(l, (41, 41), 0)

    # Suppress vessel ridges so bright disc cup/rim dominates detection.
    green = img_bgr[:, :, 1]
    vessel_like = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    vessel_like = cv2.GaussianBlur(vessel_like, (11, 11), 0)
    score = cv2.addWeighted(v_blur, 0.65, l_blur, 0.45, 0)
    score = cv2.subtract(score, (0.55 * vessel_like).astype(np.uint8))

    if fov_mask is None:
        fov_mask = np.ones((h, w), dtype=np.uint8) * 255

    vals = score[fov_mask > 0]
    if vals.size == 0:
        center = (w // 2, h // 2)
        radius = max(8, int(min(h, w) * radius_ratio))
        return center, radius

    # Disc is normally nasal side and near mid-height; constrain ROI for robustness.
    left_peak = float(np.max(score[:, :max(1, w // 3)]))
    right_peak = float(np.max(score[:, max(1, 2 * w // 3):]))
    disc_on_right = right_peak >= left_peak

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = int(0.18 * h), int(0.82 * h)
    if disc_on_right:
        x0, x1 = int(0.45 * w), w
    else:
        x0, x1 = 0, int(0.55 * w)
    roi_mask[y0:y1, x0:x1] = 255
    roi_mask = cv2.bitwise_and(roi_mask, fov_mask)

    vals_roi = score[roi_mask > 0]
    if vals_roi.size == 0:
        vals_roi = vals

    thr = np.percentile(vals_roi, 98.5)
    cand = np.zeros((h, w), dtype=np.uint8)
    cand[(score >= thr) & (roi_mask > 0)] = 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    num, lbl, stats, cent = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if num <= 1:
        masked = score.copy()
        masked[fov_mask == 0] = 0
        _, _, _, max_loc = cv2.minMaxLoc(masked)
        radius = max(8, int(min(h, w) * radius_ratio))
        return (int(max_loc[0]), int(max_loc[1])), radius

    best_idx = 1
    best_score = -1.0
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        cx, cy = cent[i]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        local_score = float(score[cy_i, cx_i])

        # Prefer candidate slightly away from center (optic disc usually lệch tâm).
        off_center = abs(cx - (w / 2.0)) / max(1.0, (w / 2.0))
        rank = local_score * (1.0 + 0.25 * off_center) * (1.0 + 0.02 * np.sqrt(area))
        if rank > best_score:
            best_score = rank
            best_idx = i

    cx, cy = cent[best_idx]
    area = float(stats[best_idx, cv2.CC_STAT_AREA])
    radius = int(np.sqrt(max(20.0, area) / np.pi))
    radius = int(np.clip(radius, max(8, int(min(h, w) * 0.03)), int(min(h, w) * 0.12)))

    return (int(round(cx)), int(round(cy))), radius


def build_zone_b_mask(shape, od_center, od_radius, inner_scale=1.0, outer_scale=2.0):
    """Build annular zone B mask centered at optic disc."""
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx, cy = od_center

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner_r = float(od_radius) * float(inner_scale)
    outer_r = float(od_radius) * float(outer_scale)

    ring = (dist >= inner_r) & (dist <= outer_r)
    return (ring.astype(np.uint8) * 255)
