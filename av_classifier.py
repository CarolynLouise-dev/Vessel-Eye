# av_classifier.py
# Phase 3: SVM-based Artery/Vein Classifier
#
# Thay thế heuristic "intensity > median" bằng SVM được huấn luyện trên
# multi-channel colour features.  Vì K-means trên colour features thất bại
# (phân phối R/B không bimodal sau CLAHE), ta dùng percentile threshold:
# top 35% R/B ratio → artery pseudo-label, rồi huấn luyện SVM.
#
# Public API:
#   train_av_classifier(dataset_path)   → Pipeline | None
#   load_av_classifier()                → Pipeline | None
#   predict_av_segment(model, img_bgr, en_green, binary_mask, path_coords)
#                                       → bool  (True = artery)

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from constant import IMG_SIZE, MODEL_DIR
import input_data
import riched_image

try:
    from skan import Skeleton, summarize
    _SKAN_AVAILABLE = True
except ImportError:
    _SKAN_AVAILABLE = False

AV_MODEL_PATH = os.path.join(MODEL_DIR, "av_classifier.pkl")


# ── Internal helper (self-contained, avoids circular import) ──────────────────

def _perp_diameter(binary_mask: np.ndarray, y: int, x: int,
                   angle_rad: float, half_width: int = 15) -> int:
    """Cross-section pixel count perpendicular à la feature_extract."""
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


# ── Feature vector per segment ────────────────────────────────────────────────

def _segment_fv(img_bgr: np.ndarray, en_green: np.ndarray,
                binary_mask: np.ndarray, path_coords) -> list | None:
    """
    Tạo vector đặc trưng 8 chiều cho một đoạn mạch (path_coords từ skan).

    [0] R_mean          — kênh R trung bình trên path (ảnh BGR gốc chưa tăng cường)
    [1] G_mean          — kênh G
    [2] B_mean          — kênh B
    [3] R_B_ratio       — R / (B + ε)  (động mạch thường cao hơn tĩnh mạch)
    [4] en_green_mean   — enhanced-green trung bình (CLAHE)
    [5] diameter        — cross-section perpendicular (pixels)
    [6] rel_y           — vị trí Y tương đối [-1, 1]
    [7] rel_x           — vị trí X tương đối [-1, 1]
    """
    h, w = binary_mask.shape
    r_vals, g_vals, b_vals, eg_vals = [], [], [], []

    for pt in path_coords:
        py, px = int(pt[0]), int(pt[1])
        if 0 <= py < h and 0 <= px < w and binary_mask[py, px] > 0:
            b, g, r = img_bgr[py, px].astype(float)
            r_vals.append(r)
            g_vals.append(g)
            b_vals.append(b)
            eg_vals.append(float(en_green[py, px]))

    if not r_vals:
        return None

    r_m  = float(np.mean(r_vals))
    g_m  = float(np.mean(g_vals))
    b_m  = float(np.mean(b_vals))
    eg_m = float(np.mean(eg_vals))
    r_b  = r_m / (b_m + 1e-6)

    # Midpoint geometry
    mid_idx  = len(path_coords) // 2
    mid      = path_coords[mid_idx]
    rel_y    = (float(mid[0]) - h / 2.0) / (h / 2.0)
    rel_x    = (float(mid[1]) - w / 2.0) / (w / 2.0)

    # Skeleton direction at midpoint → perpendicular diameter
    p_b = path_coords[max(0, mid_idx - 2)]
    p_a = path_coords[min(len(path_coords) - 1, mid_idx + 2)]
    dy_sk = float(p_a[0] - p_b[0])
    dx_sk = float(p_a[1] - p_b[1])
    angle = np.arctan2(dy_sk, dx_sk) if (dy_sk != 0 or dx_sk != 0) else 0.0
    diam  = float(_perp_diameter(binary_mask, int(mid[0]), int(mid[1]), angle))

    return [r_m, g_m, b_m, r_b, eg_m, diam, rel_y, rel_x]


def _collect_from_image(img_bgr: np.ndarray, en_green: np.ndarray,
                        binary_mask: np.ndarray, skeleton) -> list[list]:
    """Thu thập feature vectors của tất cả đoạn skan hợp lệ từ một ảnh."""
    if not _SKAN_AVAILABLE or skeleton is None:
        return []
    skel_bool = skeleton > 0
    if skel_bool.sum() < 5:
        return []

    feats = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sk    = Skeleton(skel_bool)
            stats = summarize(sk, separator="-")
        except Exception:
            return feats

    for _, row in stats.iterrows():
        arc_len = float(row.get("branch-distance", 0))
        if arc_len < 10:
            continue
        try:
            path = sk.path_coordinates(row.name)
            if len(path) < 3:
                continue
            fv = _segment_fv(img_bgr, en_green, binary_mask, path)
            if fv is not None:
                feats.append(fv)
        except Exception:
            continue

    return feats


# ── Public: train ─────────────────────────────────────────────────────────────

def train_av_classifier(dataset_path: str) -> "Pipeline | None":
    """
    Phase 3.2: Huấn luyện SVM phân loại Động mạch / Tĩnh mạch.

    Quy trình:
    1. Thu thập features (8D) của mọi đoạn skan từ toàn dataset
    2. K-means(2) trên [R/B ratio, en_green_mean] → pseudo-labels
       (nhãn 1 = artery = cluster có R/B trung bình cao hơn)
    3. StandardScaler + SVC(rbf) — huấn luyện trên toàn bộ pseudo-labels
    4. Lưu pipeline → models/av_classifier.pkl
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[A/V] Training A/V SVM Classifier ...")

    all_feats = []
    n_imgs, n_err = 0, 0

    for label_dir in ["0", "1"]:
        folder = os.path.join(dataset_path, label_dir)
        if not os.path.exists(folder):
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"  Label {label_dir}: {len(files)} images")

        for fname in files:
            fpath = os.path.join(folder, fname)
            img   = cv2.imread(fpath)
            if img is None:
                continue
            img_r = input_data.standardize_fundus_image(img, IMG_SIZE)
            try:
                en, mask, skeleton, *_ = riched_image.get_enhanced_vessels(img_r)
                seg_feats = _collect_from_image(img_r, en, mask, skeleton)
                all_feats.extend(seg_feats)
                n_imgs += 1
                if n_imgs % 50 == 0:
                    print(f"    Processed {n_imgs} images, {len(all_feats)} segments")
            except Exception:
                n_err += 1
            finally:
                del img, img_r

    print(f"  Collected {len(all_feats)} segments from {n_imgs} images ({n_err} errors)")

    if len(all_feats) < 20:
        print("[A/V] ERROR: Too few segments to train")
        return None

    X = np.array(all_feats, dtype=np.float32)  # (N, 8)

    # ── Percentile pseudo-labeling trên R/B ratio (idx 3) ────────────────────
    # K-means trên colour features thất bại: mọi ảnh trong dataset đã qua CLAHE
    # nên phân phối R/B không bimodal — K-means hội tụ thành 1 outlier cluster nhỏ
    # (137/194552 = 0.07%) thay vì chia artery/vein thực sự.
    # Giải pháp: percentile threshold trên R/B ratio — top 35% R/B → artery
    # (sinh lý học: ~35% mạch máu võng mạc là động mạch)
    print("  Pseudo-labeling: top 35% R/B ratio -> artery")
    rb_vals = X[:, 3].astype(np.float64)
    threshold = np.percentile(rb_vals, 65)
    pseudo = (rb_vals >= threshold).astype(int)

    n_art = int((pseudo == 1).sum())
    n_vei = int((pseudo == 0).sum())
    print(f"  Artery: {n_art}, Vein: {n_vei}")

    # ── Subsample để SVM có thể train trong thời gian hợp lý ───────────────
    # SVC(rbf) trên 194k samples quá chậm (O(n^2—n^3)).
    # Lấy 3000 mẫu mỗi lớp (balanced) → SVM vẫn generalise tốt.
    MAX_PER_CLASS = 3000
    rng = np.random.default_rng(42)
    art_idx = np.where(pseudo == 1)[0]
    vei_idx = np.where(pseudo == 0)[0]
    art_sample = rng.choice(art_idx, min(MAX_PER_CLASS, len(art_idx)), replace=False)
    vei_sample = rng.choice(vei_idx, min(MAX_PER_CLASS, len(vei_idx)), replace=False)
    sample_idx = np.concatenate([art_sample, vei_sample])
    X_train = X[sample_idx]
    y_train = pseudo[sample_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, AV_MODEL_PATH)
    print(f"[A/V] Model saved -> {AV_MODEL_PATH}")

    return pipeline


# ── Public: load ──────────────────────────────────────────────────────────────

def load_av_classifier() -> "Pipeline | None":
    if os.path.exists(AV_MODEL_PATH):
        try:
            return joblib.load(AV_MODEL_PATH)
        except Exception:
            pass
    return None


# ── Public: predict ───────────────────────────────────────────────────────────

def predict_av_segment(model, img_bgr: np.ndarray, en_green: np.ndarray,
                       binary_mask: np.ndarray, path_coords) -> bool:
    """
    Dự đoán một đoạn mạch là ĐỘNG MẠCH (True) hay TĨNH MẠCH (False).

    - Nếu model=None hoặc inference thất bại → fallback về intensity median.
    - path_coords: numpy array (N, 2) từ skan.path_coordinates().
    """
    if model is None or len(path_coords) < 3:
        vessel_pixels = en_green[binary_mask > 0]
        threshold = float(np.median(vessel_pixels)) if len(vessel_pixels) > 0 else 128.0
        mid = path_coords[len(path_coords) // 2]
        my, mx = int(mid[0]), int(mid[1])
        h, w = binary_mask.shape
        val = float(en_green[my, mx]) if 0 <= my < h and 0 <= mx < w else threshold
        return val > threshold

    try:
        fv = _segment_fv(img_bgr, en_green, binary_mask, path_coords)
        if fv is None:
            raise ValueError("empty path")
        X = np.array([fv], dtype=np.float32)
        return bool(model.predict(X)[0] == 1)
    except Exception:
        # Fallback intensity
        vessel_pixels = en_green[binary_mask > 0]
        threshold = float(np.median(vessel_pixels)) if len(vessel_pixels) > 0 else 128.0
        mid = path_coords[len(path_coords) // 2]
        my, mx = int(mid[0]), int(mid[1])
        h, w = binary_mask.shape
        val = float(en_green[my, mx]) if 0 <= my < h and 0 <= mx < w else threshold
        return val > threshold
