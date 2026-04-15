import warnings

import cv2
import numpy as np
from skimage.measure import label, regionprops

import anatomy

try:
    from skan import Skeleton, summarize
    _SKAN_AVAILABLE = True
except ImportError:
    _SKAN_AVAILABLE = False


FEATURE_NAMES = [
    "AVR",
    "CRAE",
    "CRVE",
    "Tortuosity",
    "StdTortuosity",
    "VesselDensity",
    "FractalDim",
    "Discontinuity",
    "EndpointGapScore",
    "WhiteningScore",
]


def _measure_diameter_at_point(binary_mask, y, x, angle_rad, half_width=15):
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


def _fractal_dimension(mask_bin):
    bw = mask_bin > 0
    if np.count_nonzero(bw) < 16:
        return 1.0

    h, w = bw.shape
    min_side = min(h, w)
    sizes = []
    k = 2
    while k <= min_side // 2:
        sizes.append(k)
        k *= 2

    if len(sizes) < 2:
        return 1.0

    counts = []
    for size in sizes:
        count = 0
        for y in range(0, h, size):
            for x in range(0, w, size):
                block = bw[y:y + size, x:x + size]
                if np.any(block):
                    count += 1
        counts.append(max(1, count))

    xs = np.log(1.0 / np.array(sizes, dtype=np.float64))
    ys = np.log(np.array(counts, dtype=np.float64))
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def _compute_endpoint_map(skeleton_bin, fov_mask=None):
    if skeleton_bin is None:
        return np.zeros((512, 512), dtype=np.uint8) if fov_mask is None else np.zeros_like(fov_mask, dtype=np.uint8)

    sk = (skeleton_bin > 0).astype(np.uint8)
    endpoint_map = np.zeros_like(sk)

    # K1: N direction endpoint (Foreground central + top, background 7 surrounding)
    k1 = np.array([[-1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8)
    # K2: NE direction endpoint (Foreground central + top-right, background 7 surrounding)
    k2 = np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int8)

    kernels = [np.rot90(k1, i) for i in range(4)] + [np.rot90(k2, i) for i in range(4)]

    for k in kernels:
        hit = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, k)
        endpoint_map = cv2.bitwise_or(endpoint_map, hit)

    endpoint_map[[0, -1], :] = 0
    endpoint_map[:, [0, -1]] = 0

    if fov_mask is not None:
        endpoint_map &= (fov_mask > 0).astype(np.uint8)

    return endpoint_map * 255


def get_discontinuity_map(skeleton_bin, vessel_mask, fov_mask=None):
    """
    Chuẩn hóa gap map giữa tầng Trích xuất và Draw.
    Sử dụng MORPH_CLOSE diện rộng, lọc endpoint và noise.
    """
    if skeleton_bin is None or vessel_mask is None:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    sk = (skeleton_bin > 0).astype(np.uint8)
    if sk.sum() == 0:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    sk_u8 = sk * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(sk_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    near_vessel = cv2.dilate(
        (vessel_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1
    )

    gap_raw = ((closed > 0) & (sk_u8 == 0) & (near_vessel > 0)).astype(np.uint8) * 255

    endpoint = _compute_endpoint_map(sk_u8, fov_mask)
    endpoint_near = cv2.dilate(endpoint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    gap = cv2.bitwise_and(gap_raw, endpoint_near)

    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
    filtered = np.zeros_like(gap)
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 35 or max(width, height) < 6:
            continue

        comp_mask = (lbl == i).astype(np.uint8) * 255
        comp_near = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        nearby_endpoints = cv2.bitwise_and(comp_near, endpoint)
        if np.count_nonzero(nearby_endpoints) < 2:
            continue

        left, top, w_box, h_box, _ = stats[i]
        h_img, w_img = gap.shape
        if left <= 2 or top <= 2 or left + w_box >= w_img - 2 or top + h_box >= h_img - 2:
            continue

        filtered[lbl == i] = 255

    gap_map = filtered
    score = float(np.count_nonzero(gap_map)) / max(1.0, float(np.count_nonzero(sk_u8 > 0)))
    return gap_map, score


def _weighted_discontinuity_score(skeleton_bin, vessel_mask, en_green, zone_mask=None):
    gap_map, _ = get_discontinuity_map(skeleton_bin, vessel_mask, zone_mask)
    if np.count_nonzero(gap_map) == 0:
        return 0.0

    gap = gap_map > 0
    if zone_mask is not None:
        gap &= zone_mask > 0

    ys, xs = np.where(gap)
    if len(ys) == 0:
        return 0.0

    weighted = 0.0
    h, w = en_green.shape
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 5), min(h, y + 6)
        x0, x1 = max(0, x - 5), min(w, x + 6)
        patch = en_green[y0:y1, x0:x1].astype(np.float32)
        weighted += float(np.std(patch)) / 255.0

    sk = (skeleton_bin > 0).astype(np.uint8)
    if zone_mask is None:
        base = sk
    else:
        base = (sk > 0) & (zone_mask > 0)
    denom = max(1.0, float(np.count_nonzero(base)))
    return float(weighted / denom)


def _endpoint_gap_score(skeleton_bin, zone_mask=None, max_gap_px=40):
    endpoint_map = _compute_endpoint_map(skeleton_bin, zone_mask)
    ys, xs = np.where(endpoint_map > 0)
    if len(ys) < 2:
        return 0.0

    endpoints = []
    sk = (skeleton_bin > 0).astype(np.uint8)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(sk.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(sk.shape[1], x + 2)
        patch = sk[y0:y1, x0:x1].copy()
        
        c_y, c_x = y - y0, x - x0
        patch[c_y, c_x] = 0
        nyx = np.argwhere(patch > 0)
        if len(nyx) == 0:
            continue

        ny, nx = nyx[0]
        ty = float(ny - c_y)
        tx = float(nx - c_x)
        n = np.hypot(ty, tx)
        if n < 1e-6:
            continue
        endpoints.append((int(y), int(x), (ty / n, tx / n)))

    if len(endpoints) < 2:
        return 0.0

    pairs = 0
    for i in range(len(endpoints)):
        y1, x1, t1 = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            y2, x2, t2 = endpoints[j]
            d = float(np.hypot(y2 - y1, x2 - x1))
            if d > max_gap_px:
                continue
            sim = (t1[0] * -t2[0]) + (t1[1] * -t2[1])
            if sim > 0.85:
                pairs += 1

    base = sk if zone_mask is None else (sk > 0) & (zone_mask > 0)
    denom = max(1.0, float(np.count_nonzero(base)))
    return float(pairs / denom)


def _whitening_score(en_green, vessel_mask, fov_mask, od_mask=None):
    valid = (fov_mask > 0) & (vessel_mask == 0)
    if od_mask is not None:
        valid &= od_mask == 0
    if not np.any(valid):
        return 0.0

    vals = en_green[valid]
    if len(vals) == 0:
        return 0.0

    p98 = np.percentile(vals, 98)
    val_mean = np.mean(vals)
    val_std = np.std(vals)

    # Issue 3: Fix Whitening absolute threshold (bounded so normal eyes aren't scored high)
    thr = max(p98, val_mean + 2.5 * val_std, 180.0)

    blobs = np.zeros_like(en_green, dtype=np.uint8)
    blobs[(en_green >= thr) & valid] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, kernel)

    area = float(np.count_nonzero(fov_mask))
    if area <= 0:
        return 0.0
    return float(np.count_nonzero(blobs) / area)


def _skeleton_features(skeleton_bin, binary_mask, en_green, zone_mask=None, img_bgr=None, av_model=None):
    torts, a_diams, v_diams = [], [], []

    if not _SKAN_AVAILABLE or skeleton_bin is None:
        return torts, a_diams, v_diams

    skel_bool = skeleton_bin > 0
    if skel_bool.sum() < 5:
        return torts, a_diams, v_diams

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sk = Skeleton(skel_bool)
            stats = summarize(sk, separator="-")
        except Exception:
            return torts, a_diams, v_diams

    vessel_pixels = en_green[binary_mask > 0]
    brightness_threshold = np.median(vessel_pixels) if len(vessel_pixels) > 0 else 128

    _av_predict = None
    if av_model is not None and img_bgr is not None:
        try:
            from av_classifier import predict_av_segment as _av_predict_fn
            _av_predict = _av_predict_fn
        except ImportError:
            pass

    col_arc = "branch-distance"
    col_ec = "euclidean-distance" if "euclidean-distance" in stats.columns else None

    for _, row in stats.iterrows():
        arc_len = float(row[col_arc]) if col_arc in stats.columns else 0.0
        if arc_len < 10:
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

        try:
            path_coords = sk.path_coordinates(row.name)
            if len(path_coords) < 3:
                continue

            mid = path_coords[len(path_coords) // 2]
            my, mx = int(mid[0]), int(mid[1])
            if zone_mask is not None and zone_mask[my, mx] == 0:
                continue

            if chord_len > 1:
                torts.append(float(arc_len / chord_len))

            p_before = path_coords[max(0, len(path_coords) // 2 - 2)]
            p_after = path_coords[min(len(path_coords) - 1, len(path_coords) // 2 + 2)]
            dy_sk = float(p_after[0] - p_before[0])
            dx_sk = float(p_after[1] - p_before[1])
            angle = np.arctan2(dy_sk, dx_sk) if (dy_sk != 0 or dx_sk != 0) else 0.0

            diam = _measure_diameter_at_point(binary_mask, my, mx, angle)
            if diam < 1:
                continue

            if _av_predict is not None:
                is_artery = _av_predict(av_model, img_bgr, en_green, binary_mask, path_coords)
            else:
                avg_int = float(en_green[my, mx])
                is_artery = avg_int > brightness_threshold

            if is_artery:
                a_diams.append(float(diam))
            else:
                v_diams.append(float(diam))
        except Exception:
            continue

    return torts, a_diams, v_diams


def extract_features(binary_mask, en_green, skeleton=None, img_bgr=None, av_model=None, fov_mask=None, return_details=False):
    label_img = label(binary_mask)
    regions = regionprops(label_img)

    vessel_pixels = en_green[binary_mask > 0]
    if len(vessel_pixels) < 10:
        default = [1.0, 1.0, 1.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if return_details:
            return default, regions, {}
        return default, regions

    if fov_mask is None:
        fov_mask = np.where(en_green > 0, 255, 0).astype(np.uint8)

    if img_bgr is None:
        img_bgr = cv2.cvtColor(en_green, cv2.COLOR_GRAY2BGR)

    od_center, od_radius = anatomy.detect_optic_disc(img_bgr, fov_mask=fov_mask)
    zone_b_mask = anatomy.build_zone_b_mask(binary_mask.shape, od_center, od_radius, inner_scale=1.0, outer_scale=2.0)
    od_mask = anatomy.build_zone_b_mask(binary_mask.shape, od_center, od_radius, inner_scale=0.0, outer_scale=1.0)

    zone_valid = zone_b_mask > 0
    zone_area = float(np.count_nonzero(zone_valid))
    density = float(np.count_nonzero((binary_mask > 0) & zone_valid) / max(1.0, zone_area))

    if _SKAN_AVAILABLE and skeleton is not None:
        torts, a_diams, v_diams = _skeleton_features(
            skeleton,
            binary_mask,
            en_green,
            zone_mask=zone_b_mask,
            img_bgr=img_bgr,
            av_model=av_model,
        )
    else:
        torts, a_diams, v_diams = [], [], []

    avg_tort = float(np.mean(torts)) if torts else 1.0
    std_tort = float(np.std(torts)) if torts else 0.0

    crae = float(np.mean(a_diams)) if a_diams else 1.0
    crve = float(np.mean(v_diams)) if v_diams else 1.5
    av_ratio = float(crae / max(1e-6, crve))

    fractal = _fractal_dimension(((binary_mask > 0) & zone_valid).astype(np.uint8) * 255)
    discontinuity = _weighted_discontinuity_score(skeleton, binary_mask, en_green, zone_mask=zone_b_mask) if skeleton is not None else 0.0
    endpoint_gap = _endpoint_gap_score(skeleton, zone_mask=zone_b_mask) if skeleton is not None else 0.0
    whitening = _whitening_score(en_green, binary_mask, fov_mask, od_mask=od_mask)

    feats = [
        av_ratio,
        crae,
        crve,
        avg_tort,
        std_tort,
        density,
        fractal,
        discontinuity,
        endpoint_gap,
        whitening,
    ]

    details = {
        "od_center": od_center,
        "od_radius": od_radius,
        "zone_b_mask": zone_b_mask,
        "crae": crae,
        "crve": crve,
        "discontinuity_score": discontinuity,
        "endpoint_gap_score": endpoint_gap,
        "whitening_score": whitening,
    }

    if return_details:
        return feats, regions, details
    return feats, regions
