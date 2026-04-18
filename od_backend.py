# od_backend.py — Deep optic disc detection using AutoMorph M2_lwnet_disc_cup (wnet ensemble).
#
# Classical image processing pipeline xử lý hình ảnh ban đầu (CLAHE, FOV mask).
# Wnet deep learning được dùng để phát hiện đĩa thị chính xác hơn heuristic sáng.
#
# Public API
# ----------
#   load_od_ensemble(n_models=3, device='cpu') -> list[nn.Module] | None
#   detect_optic_disc_deep(img_bgr, models, device='cpu')
#       -> (od_center, od_radius, confidence, disc_mask_u8)
#          od_center  : (cx, cy) int tuple in original image pixel coords
#          od_radius  : int, radius in original image pixel coords
#          confidence : float in [0, 1]
#          disc_mask  : H×W uint8 binary mask at original image size

import os
import sys
import warnings

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_LWNET_DIR = os.path.join(_BASE, "Lka-suggest", "Fundus", "M2_lwnet_disc_cup")
_EXPERIMENTS_DIR = os.path.join(
    _LWNET_DIR, "experiments", "wnet_All_three_1024_disc_cup"
)

_SEEDS = [28, 30, 32, 34, 36, 38, 40, 42]
_SEED_DIRS = [
    os.path.join(_EXPERIMENTS_DIR, str(s))
    for s in _SEEDS
    if os.path.isdir(os.path.join(_EXPERIMENTS_DIR, str(s)))
]

_IM_SIZE = 512      # wnet input size
_PTHRESHOLD = 40    # red-channel FOV threshold for z-score normalisation

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _inject_lwnet_path():
    """Add M2_lwnet_disc_cup to sys.path so 'models' is importable as a package."""
    if _LWNET_DIR not in sys.path:
        sys.path.insert(0, _LWNET_DIR)


def _load_one_od_model(seed_dir: str, device: str):
    """Load a single wnet model from a seed experiment directory."""
    _inject_lwnet_path()
    from models.get_model import get_arch   # relative import inside models/ resolved by __init__.py

    model = get_arch("wnet", in_c=3, n_classes=3)
    # Set mode to non-'train' so forward() returns a single tensor
    model.mode = "eval"

    ckpt_path = os.path.join(seed_dir, "model_checkpoint.pth")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _preprocess_od(img_bgr: np.ndarray) -> "torch.Tensor":
    """
    Tiền xử lý ảnh cổ điển trước khi đưa vào wnet:
    BGR → RGB → resize 512×512 → z-score normalisation theo FOV pixels → tensor (1,3,512,512).
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_rs = cv2.resize(rgb, (_IM_SIZE, _IM_SIZE), interpolation=cv2.INTER_LANCZOS4)
    arr = rgb_rs.astype(np.float32)

    fov = arr[..., 0] > _PTHRESHOLD
    if fov.sum() > 64:
        mean_v = np.mean(arr[fov], axis=0)
        std_v = np.std(arr[fov], axis=0) + 1e-6
    else:
        mean_v = arr.mean(axis=(0, 1))
        std_v = arr.std(axis=(0, 1)) + 1e-6

    arr = (arr - mean_v) / std_v
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


# ── Public: load ensemble ──────────────────────────────────────────────────────

def load_od_ensemble(n_models: int = 3, device: str = "cpu"):
    """
    Load up to n_models wnet models for optic disc detection.

    Parameters
    ----------
    n_models : int
        Maximum number of seed models to load (1–8).
    device : str
        PyTorch device string ('cpu', 'cuda', …).

    Returns
    -------
    list[nn.Module] | None
    """
    if not _TORCH_AVAILABLE:
        warnings.warn("[od_backend] torch not installed. OD deep backend unavailable.")
        return None
    if not _SEED_DIRS:
        warnings.warn(
            f"[od_backend] No M2_lwnet_disc_cup weights found at:\n  {_EXPERIMENTS_DIR}"
        )
        return None

    dirs = _SEED_DIRS[:n_models]
    models = []
    for d in dirs:
        try:
            m = _load_one_od_model(d, device)
            models.append(m)
        except Exception as e:
            warnings.warn(f"[od_backend] Skipping seed {os.path.basename(d)}: {e}")

    if not models:
        warnings.warn("[od_backend] All OD model loads failed. Falling back to heuristic.")
        return None

    return models


# ── Public: inference ──────────────────────────────────────────────────────────

def detect_optic_disc_deep(
    img_bgr: np.ndarray,
    models: list,
    device: str = "cpu",
):
    """
    Run the wnet ensemble and extract optic disc centre + radius.

    Bước 1 (cổ điển): tiền xử lý ảnh (resize, z-score normalisation).
    Bước 2 (deep):    wnet inference → probability map.
    Bước 3 (cổ điển): threshold → largest connected component → centroid + radius.

    Parameters
    ----------
    img_bgr : H×W×3 uint8 BGR image (any resolution; resized internally to 512×512).
    models  : list of loaded wnet models from load_od_ensemble().
    device  : PyTorch device string.

    Returns
    -------
    od_center  : (cx, cy) int tuple in original image pixel coords
    od_radius  : int, approximate disc radius in original image pixel coords
    confidence : float in [0, 1] based on circularity + area score
    disc_mask  : H×W uint8 binary mask at original image size
    """
    h, w = img_bgr.shape[:2]
    tensor = _preprocess_od(img_bgr).to(device)

    avg_prob = None
    with torch.no_grad():
        for model in models:
            out = model(tensor)          # (1, 3, 512, 512) raw logits (mode='eval' → single output)
            prob = torch.softmax(out, dim=1)
            if avg_prob is None:
                avg_prob = prob.clone()
            else:
                avg_prob = avg_prob + prob

    avg_prob = avg_prob / len(models)
    pred = torch.argmax(avg_prob, dim=1)[0].cpu().numpy()
    prob_np = avg_prob[0].cpu().numpy()

    # AutoMorph encoding: class 1 = disc rim, class 2 = cup. For optic disc area,
    # take the union of disc and cup.
    disc_mask_512 = ((pred == 1) | (pred == 2)).astype(np.uint8) * 255

    # ── Bước cổ điển: phân tích hình thái học ─────────────────────────────────
    # Resize disc mask back to original image resolution
    disc_mask_full = cv2.resize(disc_mask_512, (w, h), interpolation=cv2.INTER_NEAREST)

    # Clean up noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disc_mask_full = cv2.morphologyEx(disc_mask_full, cv2.MORPH_OPEN, kernel)

    # Find largest connected component (= optic disc)
    n_lbl, lbl, stats, centroids = cv2.connectedComponentsWithStats(
        disc_mask_full, connectivity=8
    )

    if n_lbl < 2:
        # No disc found — return image centre with zero confidence
        return (w // 2, h // 2), int(min(h, w) * 0.08), 0.0, disc_mask_full

    best_i = max(range(1, n_lbl), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    best_area = int(stats[best_i, cv2.CC_STAT_AREA])
    cx = int(centroids[best_i][0])
    cy = int(centroids[best_i][1])
    radius = max(5, int(np.sqrt(best_area / np.pi)))

    # ── Confidence score ───────────────────────────────────────────────────────
    comp_mask = (lbl == best_i).astype(np.uint8) * 255
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = max(1.0, cv2.arcLength(contours[0], True))
        circularity = float(min(1.0, (4 * np.pi * best_area) / (perimeter ** 2)))
    else:
        circularity = 0.0

    # Expected OD radius: ~6% of image diagonal (empirical for standard fundus photos)
    diag = float(np.hypot(h, w))
    expected_area = np.pi * (diag * 0.06) ** 2
    area_ratio = float(best_area) / max(1.0, expected_area)
    area_score = float(1.0 - min(1.0, abs(area_ratio - 1.0)))

    prob_disc = prob_np[1] + prob_np[2]
    mean_prob = float(np.mean(prob_disc[disc_mask_512 > 0])) if np.count_nonzero(disc_mask_512) > 0 else 0.0
    confidence = float(np.clip(0.40 * circularity + 0.25 * area_score + 0.35 * mean_prob, 0.0, 1.0))

    return (cx, cy), radius, confidence, disc_mask_full
