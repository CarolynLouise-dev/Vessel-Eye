import warnings
from typing import cast

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

import anatomy
import quality_assessment

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


def _endpoint_candidates(skeleton_bin, fov_mask=None):
    if skeleton_bin is None:
        return []

    sk = (skeleton_bin > 0).astype(np.uint8)
    if np.count_nonzero(sk) == 0:
        return []

    endpoint_map = _compute_endpoint_map(skeleton_bin, fov_mask)
    _, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    ys, xs = np.where(endpoint_map > 0)
    h, w = sk.shape
    candidates = []

    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        patch = sk[y0:y1, x0:x1].copy()
        c_y, c_x = y - y0, x - x0
        patch[c_y, c_x] = 0
        neighbors = np.argwhere(patch > 0)
        if len(neighbors) == 0:
            continue

        vec = np.zeros(2, dtype=np.float32)
        for ny, nx in neighbors:
            vec += np.array([ny - c_y, nx - c_x], dtype=np.float32)
        norm = float(np.hypot(vec[0], vec[1]))
        if norm < 1e-6:
            continue

        candidates.append({
            "y": int(y),
            "x": int(x),
            "dir": (float(vec[0] / norm), float(vec[1] / norm)),
            "component": int(comp_labels[y, x]),
            "component_size": int(comp_stats[comp_labels[y, x], cv2.CC_STAT_AREA]),
            "border_dist": int(min(y, x, h - 1 - y, w - 1 - x)),
        })

    return candidates


def _bezier_bridge_points(p0, p1, dir0, dir1, steps=28):
    p0a = np.array([float(p0[0]), float(p0[1])], dtype=np.float32)
    p1a = np.array([float(p1[0]), float(p1[1])], dtype=np.float32)
    cont0 = np.array([-float(dir0[0]), -float(dir0[1])], dtype=np.float32)
    cont1 = np.array([-float(dir1[0]), -float(dir1[1])], dtype=np.float32)
    gap = float(np.linalg.norm(p1a - p0a))
    alpha = float(np.clip(gap * 0.35, 4.0, 24.0))
    c0 = p0a + cont0 * alpha
    c1 = p1a + cont1 * alpha

    points = []
    for t in np.linspace(0.0, 1.0, steps):
        omt = 1.0 - t
        pt = (
            (omt ** 3) * p0a
            + 3.0 * (omt ** 2) * t * c0
            + 3.0 * omt * (t ** 2) * c1
            + (t ** 3) * p1a
        )
        points.append((int(round(pt[0])), int(round(pt[1]))))

    dedup = []
    for pt in points:
        if not dedup or pt != dedup[-1]:
            dedup.append(pt)
    return dedup


def _render_curve_mask(shape, points, thickness=1):
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) < 2:
        return mask
    for idx in range(len(points) - 1):
        y1, x1 = points[idx]
        y2, x2 = points[idx + 1]
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)
    return mask


def _mask_to_skeleton(mask_u8, fov_mask=None, min_area=6):
    if mask_u8 is None:
        return None
    sk = (skeletonize(mask_u8 > 0).astype(np.uint8)) * 255
    if fov_mask is not None:
        sk = cv2.bitwise_and(sk, sk, mask=fov_mask)

    n_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats((sk > 0).astype(np.uint8), connectivity=8)
    sk_clean = np.zeros_like(sk)
    for i in range(1, n_s):
        if stats_s[i, cv2.CC_STAT_AREA] >= min_area:
            sk_clean[lab_s == i] = 255
    return sk_clean


def _morphology_gap_map(skeleton_bin, vessel_mask, fov_mask=None):
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


def _hidden_gap_bridge_map(raw_vessel_mask, vessel_mask, fov_mask=None):
    if raw_vessel_mask is None or vessel_mask is None:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    if np.shape(raw_vessel_mask) != np.shape(vessel_mask):
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    raw_u8 = (raw_vessel_mask > 0).astype(np.uint8) * 255
    final_u8 = (vessel_mask > 0).astype(np.uint8) * 255
    bridge = cv2.subtract(final_u8, raw_u8)
    if fov_mask is not None:
        bridge = cv2.bitwise_and(bridge, bridge, mask=fov_mask)
    if np.count_nonzero(bridge) == 0:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    _, raw_labels = cv2.connectedComponents((raw_u8 > 0).astype(np.uint8), connectivity=8)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats((bridge > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(bridge)
    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0 or area > 96:
            continue

        comp = (lbl == i).astype(np.uint8) * 255
        neigh = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        touching = raw_labels[neigh > 0]
        touching = touching[touching > 0]
        if len(np.unique(touching)) < 2:
            continue

        comp_vis = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        out = cv2.bitwise_or(out, comp_vis)

    score = float(np.count_nonzero(out)) / max(1.0, float(np.count_nonzero(final_u8)))
    return out, score


def _graph_gap_pairs(skeleton_bin, vessel_mask, fov_mask=None, max_gap_px=44):
    if skeleton_bin is None or vessel_mask is None:
        return [], np.zeros_like(vessel_mask, dtype=np.uint8)

    endpoints = _endpoint_candidates(skeleton_bin, fov_mask)
    if len(endpoints) < 2:
        return [], np.zeros_like(vessel_mask, dtype=np.uint8)

    if fov_mask is None:
        fov_mask = np.ones_like(vessel_mask, dtype=np.uint8) * 255

    vessel_bool = vessel_mask > 0
    corridor = cv2.dilate(
        vessel_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    ) > 0

    candidate_edges = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            ep1 = endpoints[i]
            ep2 = endpoints[j]
            if ep1["component"] == ep2["component"]:
                continue
            if min(ep1["border_dist"], ep2["border_dist"]) < 4:
                continue
            if min(ep1["component_size"], ep2["component_size"]) < 6:
                continue
            if (ep1["component_size"] + ep2["component_size"]) < 18:
                continue

            dy = float(ep2["y"] - ep1["y"])
            dx = float(ep2["x"] - ep1["x"])
            dist = float(np.hypot(dy, dx))
            if dist < 6.0 or dist > float(max_gap_px):
                continue

            gap_dir = (dy / dist, dx / dist)
            cont1 = (-ep1["dir"][0], -ep1["dir"][1])
            cont2 = (-ep2["dir"][0], -ep2["dir"][1])
            align1 = float(cont1[0] * gap_dir[0] + cont1[1] * gap_dir[1])
            align2 = float(cont2[0] * -gap_dir[0] + cont2[1] * -gap_dir[1])
            if min(align1, align2) < 0.20 or (align1 + align2) < 0.65:
                continue

            curve_points = _bezier_bridge_points(
                (ep1["y"], ep1["x"]),
                (ep2["y"], ep2["x"]),
                ep1["dir"],
                ep2["dir"],
            )
            if len(curve_points) < 4:
                continue

            inside = 0
            overlap_existing = 0
            corridor_hits = 0
            interior_points = curve_points[1:-1] if len(curve_points) > 2 else curve_points
            for py, px in interior_points:
                if 0 <= py < fov_mask.shape[0] and 0 <= px < fov_mask.shape[1] and fov_mask[py, px] > 0:
                    inside += 1
                    if vessel_bool[py, px]:
                        overlap_existing += 1
                    if corridor[py, px]:
                        corridor_hits += 1

            if len(interior_points) == 0:
                continue

            inside_ratio = inside / float(len(interior_points))
            overlap_ratio = overlap_existing / float(len(interior_points))
            corridor_ratio = corridor_hits / float(len(interior_points))
            if inside_ratio < 0.98 or overlap_ratio > 0.35 or corridor_ratio < 0.40:
                continue

            score = (
                0.45 * ((align1 + align2) / 2.0)
                + 0.25 * (1.0 - dist / float(max_gap_px))
                + 0.20 * corridor_ratio
                + 0.10 * (1.0 - overlap_ratio)
            )
            if score < 0.72:
                continue

            candidate_edges.append({
                "i": i,
                "j": j,
                "score": float(score),
                "dist": float(dist),
                "points": curve_points,
                "radius": max(8, min(24, int(round(dist * 0.35)))),
                "center": (
                    int(round((ep1["x"] + ep2["x"]) / 2.0)),
                    int(round((ep1["y"] + ep2["y"]) / 2.0)),
                ),
            })

    best_for_endpoint = {}
    for idx, edge in enumerate(candidate_edges):
        for ep_idx in (edge["i"], edge["j"]):
            prev = best_for_endpoint.get(ep_idx)
            if prev is None:
                best_for_endpoint[ep_idx] = idx
                continue
            prev_edge = candidate_edges[prev]
            prev_key = (float(prev_edge["score"]), -float(prev_edge["dist"]))
            curr_key = (float(edge["score"]), -float(edge["dist"]))
            if curr_key > prev_key:
                best_for_endpoint[ep_idx] = idx

    reciprocal_edges = []
    for idx, edge in enumerate(candidate_edges):
        if best_for_endpoint.get(edge["i"]) == idx and best_for_endpoint.get(edge["j"]) == idx:
            reciprocal_edges.append(edge)

    selected = []
    used = set()
    gap_mask = np.zeros_like(vessel_mask, dtype=np.uint8)
    for edge in sorted(reciprocal_edges, key=lambda item: (-item["score"], item["dist"])):
        if edge["i"] in used or edge["j"] in used:
            continue
        used.add(edge["i"])
        used.add(edge["j"])
        selected.append(edge)
        thickness = 1 if edge["dist"] < 26 else 2
        gap_mask = cv2.bitwise_or(gap_mask, _render_curve_mask(vessel_mask.shape, edge["points"], thickness=thickness))
        if len(selected) >= 36:
            break

    return selected, gap_mask


def get_discontinuity_map(skeleton_bin, vessel_mask, fov_mask=None, raw_vessel_mask=None):
    """
    Phát hiện đứt đoạn ưu tiên theo ghép cặp endpoint dạng graph,
    có fallback về morphology khi graph không tìm được cặp hợp lệ.
    """
    if skeleton_bin is None or vessel_mask is None:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    candidate_views = []
    if raw_vessel_mask is not None and np.shape(raw_vessel_mask) == np.shape(vessel_mask) and np.count_nonzero(raw_vessel_mask) > 0:
        raw_skeleton = _mask_to_skeleton(raw_vessel_mask, fov_mask=fov_mask, min_area=5)
        if raw_skeleton is not None and np.count_nonzero(raw_skeleton) > 0:
            candidate_views.append((raw_skeleton, raw_vessel_mask))
    candidate_views.append((skeleton_bin, vessel_mask))

    hidden_gap, hidden_score = _hidden_gap_bridge_map(raw_vessel_mask, vessel_mask, fov_mask=fov_mask)

    combined_gap = np.zeros_like(vessel_mask, dtype=np.uint8)
    total_score = float(hidden_score)
    used_graph = False
    if np.count_nonzero(hidden_gap) > 0:
        combined_gap = cv2.bitwise_or(combined_gap, hidden_gap)
    for source_skeleton, source_mask in candidate_views:
        selected_pairs, graph_gap = _graph_gap_pairs(source_skeleton, source_mask, fov_mask)
        if np.count_nonzero(graph_gap) > 0:
            used_graph = True
            combined_gap = cv2.bitwise_or(combined_gap, graph_gap)
            sk = (source_skeleton > 0).astype(np.uint8)
            total_score += float(sum(pair["score"] * pair["dist"] for pair in selected_pairs)) / max(1.0, float(np.count_nonzero(sk)))

    if used_graph or np.count_nonzero(hidden_gap) > 0:
        return combined_gap, float(total_score)

    morph_gap = np.zeros_like(vessel_mask, dtype=np.uint8)
    morph_score = 0.0
    for source_skeleton, source_mask in candidate_views:
        gap_part, score_part = _morphology_gap_map(source_skeleton, source_mask, fov_mask)
        morph_gap = cv2.bitwise_or(morph_gap, gap_part)
        morph_score += float(score_part)
    return morph_gap, morph_score


def _weighted_discontinuity_score(skeleton_bin, vessel_mask, en_green, zone_mask=None, raw_vessel_mask=None):
    gap_map, _ = get_discontinuity_map(skeleton_bin, vessel_mask, zone_mask, raw_vessel_mask=raw_vessel_mask)
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


def _endpoint_gap_score(skeleton_bin, zone_mask=None, max_gap_px=40, raw_vessel_mask=None):
    if raw_vessel_mask is not None and zone_mask is not None and np.shape(raw_vessel_mask) == np.shape(zone_mask):
        raw_skeleton = _mask_to_skeleton(raw_vessel_mask, fov_mask=zone_mask, min_area=5)
        if raw_skeleton is not None and np.count_nonzero(raw_skeleton) > 0:
            skeleton_bin = raw_skeleton
    elif raw_vessel_mask is not None and skeleton_bin is not None and np.shape(raw_vessel_mask) == np.shape(skeleton_bin):
        raw_skeleton = _mask_to_skeleton(raw_vessel_mask, fov_mask=zone_mask, min_area=5)
        if raw_skeleton is not None and np.count_nonzero(raw_skeleton) > 0:
            skeleton_bin = raw_skeleton

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


def _skeleton_features(skeleton_bin, binary_mask, en_green, zone_mask=None, img_bgr=None, av_model=None,
                        artery_mask=None, vein_mask=None):
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

    # Deep A/V pixel masks take priority over SVM classifier
    _use_deep_av = (
        artery_mask is not None and vein_mask is not None
        and artery_mask.shape == binary_mask.shape
        and vein_mask.shape == binary_mask.shape
    )

    _av_predict = None
    if not _use_deep_av and av_model is not None and img_bgr is not None:
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
            path_index = cast(int, row.name)
            path_coords = sk.path_coordinates(path_index)
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

            if _use_deep_av:
                assert artery_mask is not None and vein_mask is not None
                art_votes = 0
                vein_votes = 0
                for py, px in path_coords:
                    iy, ix = int(py), int(px)
                    if 0 <= iy < artery_mask.shape[0] and 0 <= ix < artery_mask.shape[1]:
                        if artery_mask[iy, ix] > 0:
                            art_votes += 1
                        if vein_mask[iy, ix] > 0:
                            vein_votes += 1

                if art_votes > vein_votes:
                    is_artery = True
                elif vein_votes > art_votes:
                    is_artery = False
                else:
                    is_artery = float(en_green[my, mx]) > brightness_threshold
            elif _av_predict is not None:
                assert img_bgr is not None
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


def extract_features(binary_mask, en_green, skeleton=None, img_bgr=None, av_model=None, fov_mask=None, return_details=False,
                     od_center_override=None, od_radius_override=None, od_mask_override=None,
                     artery_mask=None, vein_mask=None, raw_vessel_mask=None):
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

    if od_center_override is not None and od_radius_override is not None:
        od_center = od_center_override
        od_radius = od_radius_override
        od_details = {"confidence": 1.0, "method": "deep_wnet", "source": "od_backend"}
    else:
        od_result = cast(tuple[tuple[int, int], int, dict], anatomy.detect_optic_disc(
            img_bgr,
            fov_mask=fov_mask,
            vessel_mask=binary_mask,
            return_details=True,
        ))
        od_center, od_radius, od_details = od_result
    zone_b_mask = anatomy.build_zone_b_mask(binary_mask.shape, od_center, od_radius, inner_scale=1.0, outer_scale=2.0)
    if od_mask_override is not None and od_mask_override.shape == binary_mask.shape:
        od_mask = (od_mask_override > 0).astype(np.uint8) * 255
    else:
        od_mask = od_details.get("disc_mask") if isinstance(od_details, dict) else None
        if od_mask is None or np.shape(od_mask) != np.shape(binary_mask):
            od_mask = anatomy.build_zone_b_mask(binary_mask.shape, od_center, od_radius, inner_scale=0.0, outer_scale=1.0)
        else:
            od_mask = (od_mask > 0).astype(np.uint8) * 255

    quality = quality_assessment.assess_image_quality(
        img_bgr,
        en_green=en_green,
        vessel_mask=binary_mask,
        fov_mask=fov_mask,
    )

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
            artery_mask=artery_mask,
            vein_mask=vein_mask,
        )
    else:
        torts, a_diams, v_diams = [], [], []

    avg_tort = float(np.mean(torts)) if torts else 1.0
    std_tort = float(np.std(torts)) if torts else 0.0

    crae = float(np.mean(a_diams)) if a_diams else 1.0
    crve = float(np.mean(v_diams)) if v_diams else 1.5
    av_ratio = float(crae / max(1e-6, crve))

    fractal = _fractal_dimension(((binary_mask > 0) & zone_valid).astype(np.uint8) * 255)
    discontinuity = _weighted_discontinuity_score(skeleton, binary_mask, en_green, zone_mask=zone_b_mask, raw_vessel_mask=raw_vessel_mask) if skeleton is not None else 0.0
    endpoint_gap = _endpoint_gap_score(skeleton, zone_mask=zone_b_mask, raw_vessel_mask=raw_vessel_mask) if skeleton is not None else 0.0
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
        "od_details": od_details,
        "od_mask": od_mask,
        "raw_vessel_mask": raw_vessel_mask,
        "zone_b_mask": zone_b_mask,
        "crae": crae,
        "crve": crve,
        "discontinuity_score": discontinuity,
        "endpoint_gap_score": endpoint_gap,
        "whitening_score": whitening,
        "quality": quality,
    }

    if return_details:
        return feats, regions, details
    return feats, regions
