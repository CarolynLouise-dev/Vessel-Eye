import cv2
import numpy as np


def _normalize_float_map(arr, mask=None):
    data = arr.astype(np.float32)
    if mask is not None and np.count_nonzero(mask) > 0:
        vals = data[mask > 0]
    else:
        vals = data.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def _vessel_convergence_map(vessel_mask, fov_mask=None):
    if vessel_mask is None or np.count_nonzero(vessel_mask) == 0:
        return None

    vessel_u8 = (vessel_mask > 0).astype(np.uint8)
    h, w = vessel_u8.shape
    sigma_main = max(10.0, float(min(h, w)) * 0.045)
    sigma_aux = max(16.0, float(min(h, w)) * 0.075)

    dist = cv2.distanceTransform(vessel_u8, cv2.DIST_L2, 5)
    thick_map = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), sigmaX=sigma_main, sigmaY=sigma_main)
    density_map = cv2.GaussianBlur(vessel_u8.astype(np.float32), (0, 0), sigmaX=sigma_aux, sigmaY=sigma_aux)

    thick_norm = _normalize_float_map(thick_map, mask=fov_mask)
    density_norm = _normalize_float_map(density_map, mask=fov_mask)
    conv = 0.65 * thick_norm + 0.35 * density_norm
    return np.clip(conv, 0.0, 1.0)


def _disc_candidate_rank(score_norm, bright_norm, grad_mag, conv_map, comp_mask, area, center, radius, w, h):
    cx, cy = center
    radius = max(6, int(radius))

    if np.count_nonzero(comp_mask) == 0:
        return -1.0, {}

    comp_vals = score_norm[comp_mask > 0].astype(np.float32)
    mean_score = float(np.mean(comp_vals)) if comp_vals.size else 0.0

    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None and len(contour) >= 5 else 0.0
    circularity = (4.0 * np.pi * float(area) / max(perimeter * perimeter, 1.0)) if perimeter > 0 else 0.0
    circularity = float(np.clip(circularity, 0.0, 1.3))

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner = dist <= max(3.0, 0.85 * radius)
    annulus = (dist >= 0.95 * radius) & (dist <= 1.55 * radius)
    vessel_inner = dist <= max(6.0, 1.65 * radius)

    inner_mean = float(np.mean(bright_norm[inner])) if np.any(inner) else mean_score
    annulus_mean = float(np.mean(bright_norm[annulus])) if np.any(annulus) else mean_score
    rim_contrast = inner_mean - annulus_mean

    edge_support = float(np.mean(grad_mag[annulus])) if np.any(annulus) else 0.0
    vessel_support = float(np.mean(conv_map[vessel_inner])) if conv_map is not None and np.any(vessel_inner) else 0.0
    off_center = abs(float(cx) - (w / 2.0)) / max(1.0, (w / 2.0))
    centrality = 1.0 - float(np.clip(off_center, 0.0, 1.0))
    fill_ratio = float(area / max(np.pi * radius * radius, 1.0))
    fill_score = 1.0 - float(min(1.0, abs(fill_ratio - 1.0)))

    rank = (
        1.10 * mean_score
        + 0.70 * rim_contrast
        + 0.25 * edge_support
        + 1.45 * vessel_support
        + 0.55 * fill_score
        + 0.90 * circularity
        + 0.35 * centrality
    )

    return float(rank), {
        "mean_score": mean_score,
        "rim_contrast": float(rim_contrast),
        "edge_support": float(edge_support),
        "vessel_support": float(vessel_support),
        "circularity": circularity,
        "centrality": centrality,
        "fill_score": float(fill_score),
    }


def detect_optic_disc(img_bgr, fov_mask=None, vessel_mask=None, radius_ratio=0.08, return_details=False):
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
    grad_mag = cv2.GaussianBlur(cv2.Laplacian(l_blur, cv2.CV_32F, ksize=3), (7, 7), 0)

    if fov_mask is None:
        fov_mask = np.ones((h, w), dtype=np.uint8) * 255

    grad_mag = _normalize_float_map(cv2.convertScaleAbs(grad_mag), mask=fov_mask)
    bright_norm = _normalize_float_map(score, mask=fov_mask)
    conv_map = _vessel_convergence_map(vessel_mask, fov_mask=fov_mask)
    if conv_map is None:
        conv_map = np.zeros((h, w), dtype=np.float32)
    score_norm = np.clip(0.58 * bright_norm + 0.42 * conv_map, 0.0, 1.0)

    vals = score_norm[fov_mask > 0]
    if vals.size == 0:
        center = (w // 2, h // 2)
        radius = max(8, int(min(h, w) * radius_ratio))
        details = {"confidence": 0.0, "method": "fallback-center"}
        return (center, radius, details) if return_details else (center, radius)

    # Disc is normally nasal side and near mid-height; constrain ROI for robustness.
    left_peak = float(np.max(score_norm[:, :max(1, w // 3)]))
    right_peak = float(np.max(score_norm[:, max(1, 2 * w // 3):]))
    disc_on_right = right_peak >= left_peak

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = int(0.18 * h), int(0.82 * h)
    if disc_on_right:
        x0, x1 = int(0.45 * w), w
    else:
        x0, x1 = 0, int(0.55 * w)
    roi_mask[y0:y1, x0:x1] = 255
    roi_mask = cv2.bitwise_and(roi_mask, fov_mask)

    vals_roi = score_norm[roi_mask > 0]
    if vals_roi.size == 0:
        vals_roi = vals

    thr = np.percentile(vals_roi, 97.5)
    cand = np.zeros((h, w), dtype=np.uint8)
    cand[(score_norm >= thr) & (roi_mask > 0)] = 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    num, lbl, stats, cent = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if num <= 1:
        masked = score_norm.copy()
        masked[roi_mask == 0] = 0
        _, _, _, max_loc = cv2.minMaxLoc(masked)
        radius = max(8, int(min(h, w) * radius_ratio))
        details = {"confidence": 0.15, "method": "fallback-peak", "vessel_support": float(conv_map[int(max_loc[1]), int(max_loc[0])])}
        result = ((int(max_loc[0]), int(max_loc[1])), radius, details)
        return result if return_details else result[:2]

    best_idx = 1
    best_score = -1.0
    best_details = {"confidence": 0.0, "method": "candidate"}
    best_comp_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        cx, cy = cent[i]
        comp_mask = np.where(lbl == i, 255, 0).astype(np.uint8)
        radius = np.sqrt(max(20.0, area) / np.pi)
        rank, rank_details = _disc_candidate_rank(
            score_norm,
            bright_norm,
            grad_mag,
            conv_map,
            comp_mask,
            area,
            (int(round(cx)), int(round(cy))),
            radius,
            w,
            h,
        )
        rank *= (1.0 + 0.012 * np.sqrt(area))
        if rank > best_score:
            best_score = rank
            best_idx = i
            best_details = rank_details
            best_comp_mask = comp_mask

    cx, cy = cent[best_idx]
    area = float(stats[best_idx, cv2.CC_STAT_AREA])
    radius = int(np.sqrt(max(20.0, area) / np.pi))
    radius = int(np.clip(radius, max(8, int(min(h, w) * 0.03)), int(min(h, w) * 0.12)))

    confidence = 0.0
    if best_score > 0:
        confidence = float(np.clip(best_score / 2.8, 0.0, 1.0))
    details = {
        "confidence": confidence,
        "method": "candidate-ranking-vessel-guided",
        "disc_mask": best_comp_mask,
        **best_details,
    }
    result = ((int(round(cx)), int(round(cy))), radius, details)
    return result if return_details else result[:2]


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
