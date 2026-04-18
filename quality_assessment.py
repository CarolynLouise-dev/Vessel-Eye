import cv2
import numpy as np


def _masked_percentile(values, lower, upper):
    if values.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(values, [lower, upper])
    return float(lo), float(hi)


def _clip_score(value, low, high):
    if high <= low:
        return 0.0
    return float(np.clip((float(value) - float(low)) / (float(high) - float(low)), 0.0, 1.0))


def assess_image_quality(img_bgr, en_green=None, vessel_mask=None, fov_mask=None):
    """
    Assess input reliability for the classical pipeline.

    Important: low visible vessel density is not treated as a hard failure,
    because true pathology may also reduce vessel visibility.
    """
    h, w = img_bgr.shape[:2]
    total_area = float(max(1, h * w))

    if fov_mask is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    valid = fov_mask > 0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    green = img_bgr[:, :, 1]

    fov_coverage = float(np.count_nonzero(valid) / total_area)

    if np.count_nonzero(valid) < 64:
        return {
            "quality_score": 0.0,
            "quality_level": "ungradable",
            "quality_action": "reject",
            "reasons": ["Vung FOV qua nho hoac khong xac dinh duoc."],
            "low_vessel_visibility_may_be_pathology": False,
            "metrics": {
                "fov_coverage": fov_coverage,
                "focus_score": 0.0,
                "contrast_score": 0.0,
                "illumination_score": 0.0,
                "vessel_visibility_score": 0.0,
            },
        }

    gray_vals = gray[valid].astype(np.float32)
    green_vals = green[valid].astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    focus_raw = float(np.var(lap[valid]))
    focus_score = _clip_score(np.log1p(focus_raw), np.log1p(8.0), np.log1p(180.0))

    p05, p95 = _masked_percentile(green_vals, 5, 95)
    contrast_span = p95 - p05
    contrast_score = _clip_score(contrast_span, 28.0, 105.0)

    bg = cv2.GaussianBlur(green, (0, 0), sigmaX=max(8.0, min(h, w) * 0.04))
    bg_vals = bg[valid].astype(np.float32)
    bg_mean = float(np.mean(bg_vals)) if bg_vals.size else 0.0
    bg_std = float(np.std(bg_vals)) if bg_vals.size else 0.0
    illumination_cv = bg_std / max(bg_mean, 1.0)
    illumination_score = 1.0 - _clip_score(illumination_cv, 0.16, 0.42)

    vessel_density = None
    vessel_visibility_score = 0.5
    if vessel_mask is not None:
        vessel_valid = (vessel_mask > 0) & valid
        vessel_density = float(np.count_nonzero(vessel_valid) / max(1.0, float(np.count_nonzero(valid))))
        vessel_visibility_score = _clip_score(vessel_density, 0.015, 0.12)

    coverage_score = _clip_score(fov_coverage, 0.38, 0.60)

    quality_score = (
        0.30 * focus_score
        + 0.25 * contrast_score
        + 0.20 * illumination_score
        + 0.15 * coverage_score
        + 0.10 * vessel_visibility_score
    )

    low_vessel_visibility_may_be_pathology = bool(
        vessel_density is not None
        and vessel_density < 0.045
        and focus_score >= 0.45
        and contrast_score >= 0.45
    )

    reasons = []
    if fov_coverage < 0.38:
        reasons.append("Anh khong bao phu du truong nhin vong mac.")
    if focus_score < 0.30:
        reasons.append("Anh co dau hieu mo, nhieu kha nang mat net.")
    if contrast_score < 0.30:
        reasons.append("Do tuong phan mach mau thap, kho phan doan on dinh.")
    if illumination_score < 0.30:
        reasons.append("Anh bi lech sang/toi hoac chieu sang khong dong deu.")

    if low_vessel_visibility_may_be_pathology:
        reasons.append("Mach mau hien thi thua, nhung day co the la dau hieu benh ly chu khong chi la anh kem chat luong.")

    if fov_coverage < 0.25 or (focus_score < 0.18 and contrast_score < 0.18):
        quality_level = "ungradable"
        quality_action = "reject"
    elif quality_score < 0.42:
        quality_level = "low-confidence"
        quality_action = "review"
    elif quality_score < 0.68:
        quality_level = "usable-with-caution"
        quality_action = "review" if reasons else "proceed"
    else:
        quality_level = "usable"
        quality_action = "proceed"

    if not reasons:
        reasons.append("Chat luong anh du de tiep tuc pipeline tu dong.")

    return {
        "quality_score": float(np.clip(quality_score, 0.0, 1.0)),
        "quality_level": quality_level,
        "quality_action": quality_action,
        "reasons": reasons,
        "low_vessel_visibility_may_be_pathology": low_vessel_visibility_may_be_pathology,
        "metrics": {
            "fov_coverage": fov_coverage,
            "focus_score": focus_score,
            "contrast_score": contrast_score,
            "illumination_score": float(np.clip(illumination_score, 0.0, 1.0)),
            "vessel_visibility_score": float(np.clip(vessel_visibility_score, 0.0, 1.0)),
            "vessel_density": vessel_density,
        },
    }