import cv2
import numpy as np
from constant import THRESHOLD_TORT_LOCAL, THRESHOLD_NARROW_LOCAL

try:
    from skan import Skeleton, summarize
    _SKAN_AVAILABLE = True
except ImportError:
    _SKAN_AVAILABLE = False


def _compute_discontinuity_map(skeleton_bin, vessel_mask):
    """Gap map = closing(skeleton) - skeleton, chuẩn hóa theo diện tích skeleton."""
    if skeleton_bin is None:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    sk = (skeleton_bin > 0).astype(np.uint8)
    if sk.sum() == 0:
        return np.zeros_like(vessel_mask, dtype=np.uint8), 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sk_u8 = (sk * 255).astype(np.uint8)

    # Gap candidates from morphological closure of skeleton.
    closed = cv2.morphologyEx(sk_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    near_vessel = cv2.dilate((vessel_mask > 0).astype(np.uint8) * 255,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                             iterations=1)
    gap_raw = ((closed > 0) & (sk_u8 == 0) & (near_vessel > 0)).astype(np.uint8) * 255

    # Keep gaps that are close to endpoints to represent true discontinuity candidates.
    endpoint = _compute_endpoint_map(sk_u8, np.ones_like(sk_u8, dtype=np.uint8) * 255)
    endpoint_near = cv2.dilate(endpoint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    gap = cv2.bitwise_and(gap_raw, endpoint_near)

    # Keep only meaningful gap blobs so map is clinically readable.
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

# ── Phase 2.3: Cross-section diameter helper ──────────────────────────────────

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
    Phase 3: Dùng skan để trích xuất từng đoạn mạch.
    Phân loại A/V qua SVM (nếu có img_bgr + av_model) hoặc intensity fallback.
    Trả về list dict: { cy, cx, tort, diam, is_artery, path_coords }
    """
    segments = []
    if not _SKAN_AVAILABLE or skeleton_bin is None:
        return segments

    skel_bool = (skeleton_bin > 0)
    if skel_bool.sum() < 5:
        return segments

    # Lazy-load av predict
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
    col_ec  = "euclidean-distance" if "euclidean-distance" in stats.columns else None

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

            p_b = path[max(0, len(path)//2 - 2)]
            p_a = path[min(len(path)-1, len(path)//2 + 2)]
            angle = np.arctan2(float(p_a[0]-p_b[0]), float(p_a[1]-p_b[1]))

            diam = _cross_section_diameter(binary_mask, my, mx, angle)
            if diam < 1:
                continue

            # ── Phase 3: A/V SVM classification ─────────────────────────────
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


def draw_feature_map(img_disp, vessel_mask, en_green, regions,
                     img_no_bg=None, fov_mask=None, skeleton=None,
                     img_bgr=None, av_model=None, return_debug=False,
                     anatomy_details=None):
    if img_no_bg is None:
        img_no_bg = img_disp

    h, w = img_no_bg.shape[:2]

    if fov_mask is None:
        gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    result_img = img_no_bg.copy()
    sign_img = img_no_bg.copy()
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    vessel_pixels = en_green[vessel_mask > 0]
    if len(vessel_pixels) == 0:
        if return_debug:
            empty = np.zeros((h, w, 3), dtype=np.uint8)
            return result_img, {"discontinuity_map": empty, "sign_map": result_img}
        return result_img

    brightness_threshold = np.median(vessel_pixels)

    gap_map, discontinuity_score = _compute_discontinuity_map(skeleton, vessel_mask)

    # Skeleton-based drawing
    if _SKAN_AVAILABLE and skeleton is not None:
        segments = _build_skeleton_segments(
            skeleton, vessel_mask, en_green, brightness_threshold,
            img_bgr=img_bgr, av_model=av_model
        )

        if segments:
            v_diams = [s["diam"] for s in segments if not s["is_artery"]]
            avg_v_diam = np.mean(v_diams) if v_diams else 4.0

            for seg in segments:
                color = (0, 0, 255) if seg["is_artery"] else (255, 0, 0)
                for pt in seg["path"]:
                    py, px = int(pt[0]), int(pt[1])
                    if 0 <= py < h and 0 <= px < w:
                        color_mask[py, px] = color

                cy, cx = seg["cy"], seg["cx"]
                if not (0 <= cy < h and 0 <= cx < w):
                    continue
                if fov_mask[cy, cx] == 0:
                    continue

                if seg["tort"] > THRESHOLD_TORT_LOCAL:
                    cv2.circle(result_img, (cx, cy), 12, (0, 255, 0), 2)
                    cv2.circle(sign_img, (cx, cy), 12, (0, 255, 0), 2)
                    cv2.putText(result_img, f"T{seg['tort']:.2f}",
                                (cx + 14, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (0, 255, 0), 1)

                if seg["is_artery"] and avg_v_diam > 0:
                    if (seg["diam"] / avg_v_diam) < THRESHOLD_NARROW_LOCAL:
                        cv2.circle(result_img, (cx, cy), 16, (0, 200, 255), 2)
                        cv2.circle(sign_img, (cx, cy), 16, (0, 200, 255), 2)
                        cv2.putText(result_img, f"N{seg['diam']}px",
                                    (cx + 18, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35, (0, 200, 255), 1)

            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=fov_mask)
            result_img = cv2.addWeighted(result_img, 1.0, color_mask, 0.35, 0)

            # Overlay discontinuity (orange)
            gap_overlay = np.zeros_like(result_img, dtype=np.uint8)
            gap_overlay[:, :, 1] = (gap_map > 0).astype(np.uint8) * 160
            gap_overlay[:, :, 2] = (gap_map > 0).astype(np.uint8) * 255
            result_img = cv2.addWeighted(result_img, 1.0, gap_overlay, 0.40, 0)

            # Discontinuity markers on sign map
            gap_contours, _ = cv2.findContours(gap_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gap_contours = sorted(gap_contours, key=cv2.contourArea, reverse=True)[:80]
            for c in gap_contours:
                if cv2.contourArea(c) < 35:
                    continue
                x, y, ww, hh = cv2.boundingRect(c)
                cv2.circle(sign_img, (x + ww // 2, y + hh // 2), 4, (0, 140, 255), 1)

            if anatomy_details:
                cx, cy = anatomy_details.get("od_center", (None, None))
                od_r = anatomy_details.get("od_radius", None)
                if cx is not None and cy is not None and od_r is not None:
                    cv2.circle(result_img, (int(cx), int(cy)), int(od_r), (0, 165, 255), 1)
                    cv2.circle(result_img, (int(cx), int(cy)), int(2 * od_r), (0, 255, 0), 1)
                    cv2.circle(sign_img, (int(cx), int(cy)), int(od_r), (0, 165, 255), 1)
                    cv2.circle(sign_img, (int(cx), int(cy)), int(2 * od_r), (0, 255, 0), 1)

            if return_debug:
                gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
                gap_vis[gap_map > 0] = (0, 140, 255)
                gap_vis = cv2.bitwise_and(gap_vis, gap_vis, mask=fov_mask)
                return result_img, {"discontinuity_map": gap_vis, "sign_map": sign_img}

            return result_img

    # Fallback: regionprops
    v_diams = []
    for r in regions:
        if r.area < 100:
            continue
        coords = r.coords
        intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])
        if intensity <= brightness_threshold:
            v_diams.append(r.axis_minor_length)
    avg_v_diam = np.mean(v_diams) if v_diams else 2.0

    for reg in regions:
        if reg.area < 100:
            continue
        coords = reg.coords
        center_y, center_x = map(int, reg.centroid)
        if fov_mask[center_y, center_x] == 0:
            continue
        avg_intensity = np.mean(en_green[coords[:, 0], coords[:, 1]])
        if avg_intensity > brightness_threshold:
            color = (0, 0, 255)
            is_artery = True
        else:
            color = (255, 0, 0)
            is_artery = False
        rr, cc = coords[:, 0], coords[:, 1]
        color_mask[rr, cc] = color
        t_score = (reg.perimeter ** 2) / (4 * np.pi * reg.area) if reg.area > 0 else 1.0
        if t_score > THRESHOLD_TORT_LOCAL:
            cv2.circle(result_img, (center_x, center_y), 12, (0, 255, 0), 2)
            cv2.circle(sign_img, (center_x, center_y), 12, (0, 255, 0), 2)
        if is_artery and (reg.axis_minor_length / avg_v_diam) < THRESHOLD_NARROW_LOCAL:
            cv2.circle(result_img, (center_x, center_y), 10, (255, 255, 0), 2)
            cv2.circle(sign_img, (center_x, center_y), 10, (255, 255, 0), 2)

    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=fov_mask)
    result_img = cv2.addWeighted(result_img, 1.0, color_mask, 0.3, 0)
    if return_debug:
        gap_vis = np.zeros((h, w, 3), dtype=np.uint8)
        gap_vis[gap_map > 0] = (0, 140, 255)
        gap_vis = cv2.bitwise_and(gap_vis, gap_vis, mask=fov_mask)
        return result_img, {"discontinuity_map": gap_vis, "sign_map": sign_img}

    return result_img


def draw_optic_disc_vis(img_bgr, od_center, od_radius):
    vis = img_bgr.copy()
    cx, cy = int(od_center[0]), int(od_center[1])
    r = int(od_radius)
    cv2.circle(vis, (cx, cy), r, (0, 165, 255), 2)
    cv2.circle(vis, (cx, cy), r * 2, (0, 255, 0), 1)
    cv2.putText(vis, "OD", (cx - 12, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(vis, "Zone B", (cx - 25, cy - r * 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return vis


def draw_closing_vis(skeleton, vessel_mask, fov_mask):
    h, w = vessel_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    if skeleton is None:
        return vis

    sk = (skeleton > 0).astype(np.uint8)
    if sk.sum() == 0:
        return vis

    sk_u8 = (sk * 255).astype(np.uint8)

    # Dim vessel mask as context background (dark gray so vessels visible)
    vis[vessel_mask > 0] = (25, 25, 25)

    # Skeleton in bright green
    vis[sk > 0] = (0, 220, 0)

    # Morphological closing with larger kernel to bridge vessel gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(sk_u8, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Gaps: regions filled by closing but not present in original skeleton,
    # constrained to be near the vessel mask (not free-floating noise).
    near_vessel = cv2.dilate(
        (vessel_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    gap = ((closed > 0) & (sk_u8 == 0) & (near_vessel > 0)).astype(np.uint8) * 255

    # Mark gaps orange (BGR: 0, 140, 255)
    vis[gap > 0] = (0, 140, 255)

    if fov_mask is not None:
        vis = cv2.bitwise_and(vis, vis, mask=fov_mask)

    return vis