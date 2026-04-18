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


import feature_extract


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
                              img_bgr=None, av_model=None,
                              artery_mask=None, vein_mask=None):
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

    # Deep A/V pixel masks take priority over SVM classifier
    _use_deep_av = (
        artery_mask is not None and vein_mask is not None
        and artery_mask.shape[:2] == binary_mask.shape
        and vein_mask.shape[:2] == binary_mask.shape
    )

    _av_predict = None
    if av_model is not None and img_bgr is not None and not _use_deep_av:
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
        if arc_len < 12:
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

            if _use_deep_av:
                art_votes = 0
                vein_votes = 0
                overlap_votes = 0
                for py, px in path:
                    iy, ix = int(py), int(px)
                    if 0 <= iy < artery_mask.shape[0] and 0 <= ix < artery_mask.shape[1]:
                        art = artery_mask[iy, ix] > 0
                        vein = vein_mask[iy, ix] > 0
                        if art:
                            art_votes += 1
                        if vein:
                            vein_votes += 1
                        if art and vein:
                            overlap_votes += 1

                if art_votes > vein_votes:
                    is_artery = True
                elif vein_votes > art_votes:
                    is_artery = False
                else:
                    h_m, w_m = en_green.shape
                    avg_int = float(en_green[my, mx]) if 0 <= my < h_m and 0 <= mx < w_m else brightness_threshold
                    is_artery = avg_int > brightness_threshold
            elif _av_predict is not None:
                is_artery = _av_predict(av_model, img_bgr, en_green, binary_mask, path)
            else:
                h_m, w_m = en_green.shape
                avg_int = float(en_green[my, mx]) if 0 <= my < h_m and 0 <= mx < w_m else brightness_threshold
                is_artery = avg_int > brightness_threshold

            segments.append({
                "cy": my,
                "cx": mx,
                "tort": float(tort),
                "diam": float(diam),
                "is_artery": bool(is_artery),
                "path": path,
                "overlap_votes": int(overlap_votes) if _use_deep_av else 0,
                "length": float(arc_len),
            })
        except Exception:
            continue

    return segments


def _pick_spaced_markers(markers, min_dist=32, max_items=None):
    selected = []
    for marker in sorted(markers, key=lambda item: -float(item.get("score", 0.0))):
        cx = int(marker["cx"])
        cy = int(marker["cy"])
        keep = True
        for prev in selected:
            if np.hypot(cx - prev["cx"], cy - prev["cy"]) < min_dist:
                keep = False
                break
        if keep:
            selected.append(marker)
            if max_items is not None and len(selected) >= max_items:
                break
    return selected


def _gap_markers(gap_mask, fov_mask=None):
    if gap_mask is None or np.count_nonzero(gap_mask) == 0:
        return []

    n_lbl, lbl_img, stats_cc, centroids = cv2.connectedComponentsWithStats(gap_mask, connectivity=8)
    markers = []
    for i in range(1, n_lbl):
        area = int(stats_cc[i, cv2.CC_STAT_AREA])
        if area < 18:
            continue
        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))
        if fov_mask is not None and not (0 <= cy < fov_mask.shape[0] and 0 <= cx < fov_mask.shape[1] and fov_mask[cy, cx] > 0):
            continue
        markers.append({
            "cx": cx,
            "cy": cy,
            "area": area,
            "radius": max(10, min(26, int(np.sqrt(area) * 1.9))),
            "score": float(area),
        })
    return _pick_spaced_markers(markers, min_dist=28)


def _marker_focus_masks(shape, anatomy_details=None):
    h, w = shape[:2]
    zone_mask = None
    exclusion_mask = np.zeros((h, w), dtype=np.uint8)
    if not anatomy_details:
        return zone_mask, exclusion_mask

    zone_candidate = anatomy_details.get("zone_b_mask")
    if zone_candidate is not None and np.shape(zone_candidate)[:2] == (h, w):
        zone_mask = (zone_candidate > 0).astype(np.uint8) * 255

    od_mask = anatomy_details.get("od_mask")
    if od_mask is not None and np.shape(od_mask)[:2] == (h, w):
        exclusion_mask = (od_mask > 0).astype(np.uint8) * 255
        exclusion_mask = cv2.dilate(exclusion_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
    return zone_mask, exclusion_mask


def _marker_allowed(cx, cy, fov_mask=None, zone_mask=None, exclusion_mask=None):
    if fov_mask is not None and (cy < 0 or cx < 0 or cy >= fov_mask.shape[0] or cx >= fov_mask.shape[1] or fov_mask[cy, cx] == 0):
        return False
    if exclusion_mask is not None and exclusion_mask.size > 0 and 0 <= cy < exclusion_mask.shape[0] and 0 <= cx < exclusion_mask.shape[1] and exclusion_mask[cy, cx] > 0:
        return False
    if zone_mask is not None and zone_mask.size > 0:
        return 0 <= cy < zone_mask.shape[0] and 0 <= cx < zone_mask.shape[1] and zone_mask[cy, cx] > 0
    return True


def analyze_pathology_findings(skeleton_bin, vessel_mask, en_green,
                               fov_mask=None, img_bgr=None, av_model=None,
                               artery_mask=None, vein_mask=None,
                               anatomy_details=None, raw_vessel_mask=None):
    findings = {
        "segments": [],
        "gap_map": np.zeros_like(vessel_mask, dtype=np.uint8) if vessel_mask is not None else None,
        "tortuosity": [],
        "narrowing": [],
        "dilation": [],
        "gaps": [],
        "summary": [],
    }

    if skeleton_bin is None or vessel_mask is None or en_green is None:
        return findings

    zone_mask, exclusion_mask = _marker_focus_masks(vessel_mask.shape, anatomy_details=anatomy_details)

    vessel_pixels = en_green[vessel_mask > 0]
    brightness_threshold = float(np.median(vessel_pixels)) if len(vessel_pixels) > 0 else 128.0

    segments = _build_skeleton_segments(
        skeleton_bin, vessel_mask, en_green, brightness_threshold,
        img_bgr=img_bgr, av_model=av_model,
        artery_mask=artery_mask, vein_mask=vein_mask,
    )
    findings["segments"] = segments

    gap_map, _ = feature_extract.get_discontinuity_map(skeleton_bin, vessel_mask, fov_mask, raw_vessel_mask=raw_vessel_mask)
    findings["gap_map"] = gap_map

    tort_markers = []
    for seg in segments:
        if seg["tort"] <= THRESHOLD_TORT_LOCAL:
            continue
        if not _marker_allowed(int(seg["cx"]), int(seg["cy"]), fov_mask=fov_mask, zone_mask=zone_mask, exclusion_mask=exclusion_mask):
            continue
        tort_markers.append({
            "type": "Tortuosity",
            "cx": int(seg["cx"]),
            "cy": int(seg["cy"]),
            "score": float(seg["tort"] * max(1.0, seg.get("length", 1.0))),
            "value": float(seg["tort"]),
            "diam": float(seg["diam"]),
        })

    a_diams = [s["diam"] for s in segments if s["is_artery"]]
    v_diams = [s["diam"] for s in segments if not s["is_artery"]]
    avg_a = float(np.mean(a_diams)) if a_diams else 4.0
    avg_v = float(np.mean(v_diams)) if v_diams else 6.0
    std_a = float(np.std(a_diams)) if len(a_diams) > 1 else 1.0
    std_v = float(np.std(v_diams)) if len(v_diams) > 1 else 1.5

    narrow_markers = []
    wide_markers = []
    for seg in segments:
        diam = float(seg["diam"])
        if not _marker_allowed(int(seg["cx"]), int(seg["cy"]), fov_mask=fov_mask, zone_mask=zone_mask, exclusion_mask=exclusion_mask):
            continue
        if seg["is_artery"]:
            narrow_cut = min(avg_a - 0.9 * std_a, avg_v * THRESHOLD_NARROW_LOCAL)
            if diam < max(1.5, narrow_cut):
                narrow_markers.append({
                    "type": "Arterial narrowing",
                    "cx": int(seg["cx"]),
                    "cy": int(seg["cy"]),
                    "score": float(max(0.0, avg_a - diam)),
                    "value": diam,
                    "reference": avg_a,
                })
            if diam > max(avg_a + 1.2 * std_a, avg_a * 1.45):
                wide_markers.append({
                    "type": "Arterial dilation",
                    "cx": int(seg["cx"]),
                    "cy": int(seg["cy"]),
                    "score": float(max(0.0, diam - avg_a)),
                    "value": diam,
                    "reference": avg_a,
                })
        else:
            if diam > max(avg_v + 1.2 * std_v, avg_v * 1.35):
                wide_markers.append({
                    "type": "Venous dilation",
                    "cx": int(seg["cx"]),
                    "cy": int(seg["cy"]),
                    "score": float(max(0.0, diam - avg_v)),
                    "value": diam,
                    "reference": avg_v,
                })

    gap_markers = []
    for marker in _gap_markers(gap_map, fov_mask=fov_mask):
        if not _marker_allowed(int(marker["cx"]), int(marker["cy"]), fov_mask=fov_mask, zone_mask=zone_mask, exclusion_mask=None):
            continue
        gap_markers.append({
            "type": "Discontinuity",
            "cx": int(marker["cx"]),
            "cy": int(marker["cy"]),
            "score": float(marker.get("score", marker.get("area", 0.0))),
            "value": float(marker.get("area", 0.0)),
            "radius": int(marker.get("radius", 10)),
        })

    findings["tortuosity"] = _pick_spaced_markers(tort_markers, min_dist=30, max_items=8)
    findings["narrowing"] = _pick_spaced_markers(narrow_markers, min_dist=34, max_items=8)
    findings["dilation"] = _pick_spaced_markers(wide_markers, min_dist=34, max_items=8)
    findings["gaps"] = _pick_spaced_markers(gap_markers, min_dist=28, max_items=8)

    summary = []
    for group_name in ("narrowing", "dilation", "tortuosity", "gaps"):
        for marker in findings[group_name]:
            summary.append(marker)
    findings["summary"] = sorted(summary, key=lambda item: -float(item.get("score", 0.0)))[:10]
    return findings


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 1 — Optic Disc Detection
# ══════════════════════════════════════════════════════════════════════════════

def draw_optic_disc_vis(img_bgr, od_center, od_radius, zone_b_mask=None, disc_mask=None, confidence=None):
    """Vẽ optic disc và Zone B lên ảnh fundus."""
    vis = img_bgr.copy()
    cx, cy = int(od_center[0]), int(od_center[1])
    r = int(od_radius)

    if disc_mask is not None and disc_mask.shape[:2] == vis.shape[:2] and np.count_nonzero(disc_mask) > 0:
        disc_mask_u8 = (disc_mask > 0).astype(np.uint8) * 255
        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = (disc_mask_u8 > 0).astype(np.uint8) * 70
        overlay[:, :, 2] = (disc_mask_u8 > 0).astype(np.uint8) * 180
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.28, 0)
        contours, _ = cv2.findContours(disc_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis, contours, -1, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        cv2.circle(vis, (cx, cy), r, (0, 165, 255), 2)

    if zone_b_mask is not None and zone_b_mask.shape[:2] == vis.shape[:2] and np.count_nonzero(zone_b_mask) > 0:
        zone_mask_u8 = (zone_b_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(zone_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(vis, contours, -1, (0, 255, 180), 2, cv2.LINE_AA)
    else:
        cv2.circle(vis, (cx, cy), r * 2, (0, 255, 180), 2)

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

    if confidence is not None:
        cv2.putText(vis, f"Conf {confidence * 100:.0f}%", (max(8, cx - 28), cy + r + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 235, 255), 1, cv2.LINE_AA)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 2 — A/V Calibre Heat-map
# ══════════════════════════════════════════════════════════════════════════════

def draw_av_calibre_map(skeleton_bin, vessel_mask, en_green,
                        fov_mask=None, img_bgr=None, av_model=None,
                        artery_mask=None, vein_mask=None):
    """
    Panel 2: Bản đồ phân loại động/tĩnh mạch.
    Động mạch: đỏ, Tĩnh mạch: xanh dương, Overlap: xanh lá.
    """
    if skeleton_bin is None or vessel_mask is None:
        h, w = vessel_mask.shape if vessel_mask is not None else (512, 512)
        return np.zeros((h, w, 3), dtype=np.uint8)

    h, w = vessel_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    vis[vessel_mask > 0] = (18, 26, 18)

    use_deep_masks = (
        artery_mask is not None and vein_mask is not None
        and artery_mask.shape[:2] == vessel_mask.shape
        and vein_mask.shape[:2] == vessel_mask.shape
    )

    if use_deep_masks:
        vessel_bool = vessel_mask > 0
        art_bool = (artery_mask > 0) & vessel_bool
        vein_bool = (vein_mask > 0) & vessel_bool
        overlap_bool = art_bool & vein_bool
        artery_only = art_bool & ~overlap_bool
        vein_only = vein_bool & ~overlap_bool

        vis[artery_only] = (40, 40, 235)
        vis[vein_only] = (235, 120, 40)
        vis[overlap_bool] = (40, 210, 80)

        vessel_outline = cv2.dilate((vessel_bool.astype(np.uint8) * 255), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        outline = cv2.Canny(vessel_outline, 40, 120)
        vis[outline > 0] = (235, 235, 235)

        if fov_mask is not None:
            vis = cv2.bitwise_and(vis, vis, mask=fov_mask)
        _draw_legend_av(vis, h, w, has_overlap=True)
        return vis

    if en_green is None or not _SKAN_AVAILABLE:
        # Fallback: chỉ vẽ skeleton trắng
        sk_u8 = (skeleton_bin > 0).astype(np.uint8) * 255
        vis[sk_u8 > 0] = (200, 200, 200)
        return vis

    vessel_pixels = en_green[vessel_mask > 0]
    brightness_threshold = np.median(vessel_pixels) if len(vessel_pixels) > 0 else 128.0

    segments = _build_skeleton_segments(
        skeleton_bin, vessel_mask, en_green, brightness_threshold,
        img_bgr=img_bgr, av_model=av_model,
        artery_mask=artery_mask, vein_mask=vein_mask
    )

    if not segments:
        sk_u8 = (skeleton_bin > 0).astype(np.uint8) * 255
        vis[sk_u8 > 0] = (150, 200, 150)
        return vis

    # Chuẩn hóa đường kính để map màu
    all_diams = [s["diam"] for s in segments]
    d_min = max(1.0, float(np.percentile(all_diams, 5)))
    d_max = max(d_min + 1.0, float(np.percentile(all_diams, 95)))

    for seg in segments:
        path = seg["path"]
        diam_norm = np.clip((seg["diam"] - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        base = np.array((30, 30, 230) if seg["is_artery"] else (230, 120, 30), dtype=np.float32)
        boost = np.array((0, 0, 25) if seg["is_artery"] else (25, 20, 0), dtype=np.float32) * diam_norm
        color = tuple(np.clip(base + boost, 0, 255).astype(np.uint8).tolist())
        thickness = max(1, min(5, int(round(seg["diam"] / 2.5))))
        pts = [(int(pt[1]), int(pt[0])) for pt in path]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    _draw_legend_av(vis, h, w, has_overlap=False)

    return vis


def _draw_legend_av(vis, h, w, has_overlap=False):
    """Vẽ legend nhỏ tại góc ảnh."""
    box_h = 58 if has_overlap else 40
    x0, y0 = 8, h - (box_h + 10)
    cv2.rectangle(vis, (x0 - 2, y0 - 2), (x0 + 116, y0 + box_h), (30, 30, 30), -1)
    cv2.line(vis, (x0, y0 + 8), (x0 + 20, y0 + 8), (30, 30, 230), 3)
    cv2.putText(vis, "Artery", (x0 + 24, y0 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 180, 180), 1, cv2.LINE_AA)
    cv2.line(vis, (x0, y0 + 28), (x0 + 20, y0 + 28), (230, 120, 30), 3)
    cv2.putText(vis, "Vein", (x0 + 24, y0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 200, 180), 1, cv2.LINE_AA)
    if has_overlap:
        cv2.line(vis, (x0, y0 + 48), (x0 + 20, y0 + 48), (40, 210, 80), 3)
        cv2.putText(vis, "Overlap", (x0 + 24, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 220, 180), 1, cv2.LINE_AA)


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

    # 3. Sử dụng Gap Map thống nhất từ tầng trích xuất
    gap_filtered, _ = feature_extract.get_discontinuity_map(skeleton, vessel_mask, fov_mask)
    if np.count_nonzero(gap_filtered) == 0:
        return vis

    # 4. Tô màu gap
    vis[gap_filtered > 0] = (50, 80, 255)

    n_lbl, lbl_img, stats_cc, centroids = cv2.connectedComponentsWithStats(gap_filtered, connectivity=8)
    gap_centers = []
    for i in range(1, n_lbl):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        cx_g = int(centroids[i][0])
        cy_g = int(centroids[i][1])
        gap_centers.append((cx_g, cy_g, area))

    gap_centers.sort(key=lambda x: -x[2])
    for (gx, gy, area) in gap_centers:
        if area < 18:
            continue
        r = max(8, min(22, int(np.sqrt(area) * 1.6)))
        cv2.circle(vis, (gx, gy), r + 4, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (gx, gy), r, (0, 60, 220), 1, cv2.LINE_AA)
        cv2.circle(vis, (gx, gy), 3, (0, 200, 255), -1)

    # 6. Đánh dấu endpoint (khớp theo hàm mới)
    endpoint = feature_extract._compute_endpoint_map(skeleton, fov_mask)
    ep_coords = np.where(endpoint > 0)
    ep_list = list(zip(ep_coords[0], ep_coords[1]))
    # Chỉ hiển thị endpoint nằm gần gap
    gap_dilated = cv2.dilate(gap_filtered,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    for ey, ex in ep_list[:200]:
        if fov_mask is not None and fov_mask[ey, ex] == 0:
            continue
        if gap_dilated[ey, ex] > 0:  # Chỉ vẽ endpoint gần gap
            cv2.circle(vis, (ex, ey), 4, (0, 255, 200), 1, cv2.LINE_AA)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    return vis


def draw_structural_map(skeleton, vessel_mask, en_green=None, fov_mask=None, img_bgr=None, av_model=None,
                        artery_mask=None, vein_mask=None, raw_vessel_mask=None):
    """
    Panel 3: Bản đồ Cấu trúc — vừa hiển thị xoắn vặn vừa hiển thị đứt đoạn.
    """
    h, w = vessel_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[vessel_mask > 0] = (14, 20, 14)

    if skeleton is None:
        return vis

    sk = (skeleton > 0).astype(np.uint8) * 255
    if np.count_nonzero(sk) == 0:
        return vis

    sk_disp = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
    vis[sk_disp > 0] = (0, 185, 60)

    brightness_threshold = 128.0
    if en_green is not None:
        vessel_pixels = en_green[vessel_mask > 0]
        if len(vessel_pixels) > 0:
            brightness_threshold = float(np.median(vessel_pixels))

    segments = _build_skeleton_segments(
        skeleton, vessel_mask, en_green if en_green is not None else np.zeros_like(vessel_mask),
        brightness_threshold,
        img_bgr=img_bgr, av_model=av_model,
        artery_mask=artery_mask, vein_mask=vein_mask,
    ) if en_green is not None else []

    tort_markers = []
    for seg in segments:
        if seg["tort"] <= THRESHOLD_TORT_LOCAL:
            continue
        pts = [(int(pt[1]), int(pt[0])) for pt in seg["path"]]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], (0, 255, 90), 2, cv2.LINE_AA)
        tort_markers.append({
            "cx": seg["cx"],
            "cy": seg["cy"],
            "score": seg["tort"] * seg.get("length", 1.0),
            "tort": seg["tort"],
        })

    for marker in _pick_spaced_markers(tort_markers, min_dist=30, max_items=10):
        cx = int(marker["cx"])
        cy = int(marker["cy"])
        cv2.circle(vis, (cx, cy), 15, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(vis, "XOAN", (cx + 16, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 255, 120), 1, cv2.LINE_AA)

    gap_mask, _ = feature_extract.get_discontinuity_map(skeleton, vessel_mask, fov_mask, raw_vessel_mask=raw_vessel_mask)
    vis[gap_mask > 0] = (30, 145, 255)
    for marker in _gap_markers(gap_mask, fov_mask=fov_mask):
        cx = int(marker["cx"])
        cy = int(marker["cy"])
        radius = int(marker["radius"])
        cv2.circle(vis, (cx, cy), radius, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), max(4, radius - 6), (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "GAP", (cx + radius + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 220, 255), 1, cv2.LINE_AA)

    endpoint = feature_extract._compute_endpoint_map(skeleton, fov_mask)
    endpoint_near_gap = cv2.bitwise_and(
        endpoint,
        cv2.dilate(gap_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1),
    )
    ys, xs = np.where(endpoint_near_gap > 0)
    for ey, ex in zip(ys[:300], xs[:300]):
        cv2.circle(vis, (int(ex), int(ey)), 3, (0, 255, 220), 1, cv2.LINE_AA)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 3 — Phân đoạn mạch máu B&W cực rõ
# ══════════════════════════════════════════════════════════════════════════════

def draw_vessel_segmentation(vessel_mask, en_green=None, fov_mask=None):
    """
    Panel 3 (B&W): Ảnh phân đoạn mạch máu với độ rộng CHÍNH XÁC.

    LƯU Ý: vessel_mask từ riched_image đã qua morphological close rồi.
    Hàm này KHÔNG close/dilate thêm để tránh làm phồng mạch.
    Chỉ: CC filter (bỏ noise) + erosion nhẹ (trả về độ mảnh thực tế).
    """
    if vessel_mask is None:
        return np.zeros((512, 512), dtype=np.uint8)

    h, w = vessel_mask.shape

    # ── 1. Binary base (giữ nguyên, không close thêm) ───────────────────────
    binary = (vessel_mask > 0).astype(np.uint8) * 255

    # ── 2. CC filter: bỏ noise nhỏ, giữ mạch thực ──────────────────────────
    fov_px = float(np.count_nonzero(fov_mask)) if fov_mask is not None else float(h * w)
    min_area = max(50, int(fov_px * 0.00006))
    n_lbl, lbl_img, stats_cc, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, n_lbl):
        if stats_cc[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[lbl_img == i] = 255

    # ── 3. Erosion nhẹ 1px để bù lại phần đã dày từ upstream close ──────────
    # riched_image dùng MORPH_CLOSE k3 iter=2 → mạch có thể rộng hơn ~1-2px
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thinned = cv2.erode(filtered, k_erode, iterations=1)

    # ── 4. Đảm bảo không mất mạch mảnh do erode quá ─────────────────────────
    # Nếu erode làm mất >40% pixel → bỏ erode, dùng lại filtered
    if np.count_nonzero(thinned) < 0.60 * np.count_nonzero(filtered):
        thinned = filtered

    # ── 5. Tăng tương phản cạnh mạch qua CLAHE (KHÔNG thay đổi độ dày) ──────
    if en_green is not None:
        g_norm = cv2.normalize(en_green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        g_eq = clahe.apply(g_norm)
        # Chỉ lấy cường độ trong vùng mask — KHÔNG dùng để phình to
        vessel_intensity = cv2.bitwise_and(g_eq, g_eq,
                                           mask=(thinned > 0).astype(np.uint8))
        # Blend nhẹ: 70% binary (giữ hình dạng) + 30% intensity (tăng cạnh)
        blend = cv2.addWeighted(thinned.astype(np.float32), 0.70,
                                vessel_intensity.astype(np.float32), 0.30, 0)
        _, final = cv2.threshold(blend.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
    else:
        final = thinned

    if fov_mask is not None:
        final = cv2.bitwise_and(final, fov_mask)

    return final  # uint8 2D grayscale — caller dùng is_gray=True


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: Panel 4 — Heat-map Hẹp/Phồng mạch
# ══════════════════════════════════════════════════════════════════════════════

def draw_diameter_heatmap(skeleton_bin, vessel_mask, en_green,
                          fov_mask=None, img_bgr=None, av_model=None,
                          artery_mask=None, vein_mask=None,
                          anatomy_details=None, raw_vessel_mask=None):
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
        img_bgr=img_bgr, av_model=av_model,
        artery_mask=artery_mask, vein_mask=vein_mask
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

    narrow_markers = []
    wide_markers = []
    zone_mask, exclusion_mask = _marker_focus_masks(vessel_mask.shape, anatomy_details=anatomy_details)

    for seg in segments:
        path = seg["path"]
        diam = seg["diam"]
        is_artery = seg["is_artery"]
        marker_ok = _marker_allowed(int(seg["cx"]), int(seg["cy"]), fov_mask=fov_mask, zone_mask=zone_mask, exclusion_mask=exclusion_mask)

        # Issue 5: Narrow/Swollen absolute threshold vs V_mean
        if is_artery:
            narrow_cut = min(avg_a - 0.9 * std_a, avg_v * THRESHOLD_NARROW_LOCAL)
            is_narrow = diam < max(1.5, narrow_cut)
            is_wide = diam > max(avg_a + 1.2 * std_a, avg_a * 1.45)
        else:
            is_narrow = False
            is_wide = diam > max(avg_v + 1.2 * std_v, avg_v * 1.35)

        # Normalize diameter cho màu: 0=hẹp nhất (xanh), 1=rộng nhất (đỏ)
        ref_mean = avg_a if is_artery else avg_v
        spread = max(ref_mean * 0.5, 2.0)
        norm_d = np.clip((diam - (ref_mean - spread)) / (2 * spread), 0.0, 1.0)
        color = _apply_colormap_jet(float(norm_d))

        # Vẽ từng segment
        thickness = max(1, min(5, int(diam / 2.5)))
        pts = [(int(pt[1]), int(pt[0])) for pt in path]
        for i in range(len(pts) - 1):
            cv2.line(vis, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

        # Ghi nhận vị trí bất thường
        cx_s, cy_s = seg["cx"], seg["cy"]
        if is_narrow and marker_ok:
            narrow_markers.append({"cx": cx_s, "cy": cy_s, "diam": diam, "score": max(0.0, avg_a - diam)})
        if is_wide and marker_ok:
            wide_markers.append({"cx": cx_s, "cy": cy_s, "diam": diam, "score": max(0.0, diam - (avg_a if is_artery else avg_v))})

    for marker in _pick_spaced_markers(narrow_markers, min_dist=34, max_items=10):
        mx, my, diam = int(marker["cx"]), int(marker["cy"]), float(marker["diam"])
        if not (0 <= my < h and 0 <= mx < w):
            continue
        if fov_mask is not None and fov_mask[my, mx] == 0:
            continue
        cv2.circle(vis, (mx, my), 16, (0, 50, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (mx, my), 22, (0, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"HEP {diam:.1f}px", (mx + 14, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 150, 255), 1, cv2.LINE_AA)

    for marker in _pick_spaced_markers(wide_markers, min_dist=34, max_items=10):
        mx, my, diam = int(marker["cx"]), int(marker["cy"]), float(marker["diam"])
        if not (0 <= my < h and 0 <= mx < w):
            continue
        if fov_mask is not None and fov_mask[my, mx] == 0:
            continue
        cv2.circle(vis, (mx, my), 18, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"PHONG {diam:.1f}px", (mx + 14, my + 8),
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
                     anatomy_details=None, artery_mask=None, vein_mask=None,
                     raw_vessel_mask=None):
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
    findings = analyze_pathology_findings(
        skeleton, vessel_mask, en_green,
        fov_mask=fov_mask, img_bgr=img_bgr, av_model=av_model,
        artery_mask=artery_mask, vein_mask=vein_mask,
        anatomy_details=anatomy_details,
        raw_vessel_mask=raw_vessel_mask,
    )
    gap_map = findings.get("gap_map", np.zeros((h, w), dtype=np.uint8))

    if _SKAN_AVAILABLE and skeleton is not None:
        segments = _build_skeleton_segments(
            skeleton, vessel_mask, en_green, brightness_threshold,
            img_bgr=img_bgr, av_model=av_model,
            artery_mask=artery_mask, vein_mask=vein_mask
        )

        if segments:
            v_diams = [s["diam"] for s in segments if not s["is_artery"]]
            a_diams = [s["diam"] for s in segments if s["is_artery"]]
            avg_v_diam = float(np.mean(v_diams)) if v_diams else 4.0
            avg_a_diam = float(np.mean(a_diams)) if a_diams else 4.0
            std_a = float(np.std(a_diams)) if len(a_diams) > 1 else 1.0

            # Map A/V màu lên ảnh
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            use_deep_masks = (
                artery_mask is not None and vein_mask is not None
                and artery_mask.shape[:2] == vessel_mask.shape
                and vein_mask.shape[:2] == vessel_mask.shape
            )
            if use_deep_masks:
                art = (artery_mask > 0) & (vessel_mask > 0)
                vein = (vein_mask > 0) & (vessel_mask > 0)
                overlap = art & vein
                color_mask[art & ~overlap] = (0, 0, 220)
                color_mask[vein & ~overlap] = (220, 60, 0)
                color_mask[overlap] = (0, 210, 90)
            else:
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

            for marker in findings.get("tortuosity", [])[:6]:
                cy_s, cx_s = int(marker["cy"]), int(marker["cx"])
                cv2.circle(result_img, (cx_s, cy_s), 18, (0, 255, 80), 2, cv2.LINE_AA)
                cv2.circle(sign_img, (cx_s, cy_s), 18, (0, 255, 80), 2, cv2.LINE_AA)
                cv2.putText(result_img, f"XOAN {float(marker['value']):.1f}",
                            (cx_s + 20, cy_s - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80), 1, cv2.LINE_AA)
                cv2.circle(danger_map, (cx_s, cy_s), 28, 55, -1)

            for marker in findings.get("narrowing", [])[:6]:
                cy_s, cx_s = int(marker["cy"]), int(marker["cx"])
                cv2.circle(result_img, (cx_s, cy_s), 20, (0, 60, 255), 2, cv2.LINE_AA)
                cv2.circle(result_img, (cx_s, cy_s), 28, (0, 120, 255), 1, cv2.LINE_AA)
                cv2.circle(sign_img, (cx_s, cy_s), 20, (0, 60, 255), 2, cv2.LINE_AA)
                cv2.putText(result_img, f"HEP {float(marker['value']):.1f}px",
                            (cx_s + 22, cy_s + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 150, 255), 1, cv2.LINE_AA)
                cv2.circle(danger_map, (cx_s, cy_s), 34, 80, -1)

            for marker in findings.get("dilation", [])[:5]:
                cy_s, cx_s = int(marker["cy"]), int(marker["cx"])
                cv2.circle(result_img, (cx_s, cy_s), 18, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(sign_img, (cx_s, cy_s), 18, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(result_img, f"PHONG {float(marker['value']):.1f}px",
                            (cx_s + 20, cy_s + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 200), 1, cv2.LINE_AA)
                cv2.circle(danger_map, (cx_s, cy_s), 30, 65, -1)

            # ── Overlay gap đứt đoạn (cam) ────────────────────────────────
            gap_overlay = np.zeros_like(result_img, dtype=np.uint8)
            gap_overlay[:, :, 1] = (gap_map > 0).astype(np.uint8) * 140
            gap_overlay[:, :, 2] = (gap_map > 0).astype(np.uint8) * 255
            result_img = cv2.addWeighted(result_img, 1.0, gap_overlay, 0.55, 0)

            # ── Vòng tròn ĐỨT ĐOẠN ────────────────────────────────────────
            for marker in findings.get("gaps", [])[:6]:
                gx_c = int(marker["cx"])
                gy_c = int(marker["cy"])
                r_g = int(marker.get("radius", 14))
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