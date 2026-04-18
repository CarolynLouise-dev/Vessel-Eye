# deep_backend.py
# Optional deep-learning vessel segmentation backend using AutoMorph M2_Vessel_seg.
#
# Falls back to the classical riched_image.get_enhanced_vessels() if torch or
# model weights are unavailable.
#
# Public API
# ----------
#   load_vessel_seg_ensemble(n_models=3, device='cpu') -> list[nn.Module] | None
#   segment_vessels_deep(img_bgr, models, device='cpu')
#       -> (vessel_mask_u8: ndarray, prob_map_f32: ndarray)
#   get_enhanced_vessels_deep(img_bgr, models=None, device='cpu')
#       -> (en_green, vessel_mask, skeleton, img_no_bg, proc_mask)
#          same tuple as riched_image.get_enhanced_vessels()

import os
import sys
import warnings

import cv2
import numpy as np
from skimage.morphology import skeletonize

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_M2_DIR = os.path.join(_BASE, "Lka-suggest", "Fundus", "M2_Vessel_seg")
_WEIGHTS_ROOT = os.path.join(_M2_DIR, "Saved_model", "train_on_ALL-SIX")

# Seeds used in the original AutoMorph training (ascending order)
_SEEDS = [24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
_WEIGHT_PATHS = [
    os.path.join(
        _WEIGHTS_ROOT,
        f"20210630_uniform_thres40_ALL-SIX_savebest_randomseed_{s}",
        "G_best_F1_epoch.pth",
    )
    for s in _SEEDS
    if os.path.exists(
        os.path.join(
            _WEIGHTS_ROOT,
            f"20210630_uniform_thres40_ALL-SIX_savebest_randomseed_{s}",
            "G_best_F1_epoch.pth",
        )
    )
]

# AutoMorph parameters (confirmed from test_outside_integrated.py)
_TARGET_SIZE = 912   # uniform image size expected by the model
_PTHRESHOLD  = 40    # minimum red-channel value defining the FOV for z-score

# Attempt a one-time torch import
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _inject_m2_path():
    """Add M2_Vessel_seg to sys.path if needed (isolated helper)."""
    if _M2_DIR not in sys.path:
        sys.path.insert(0, _M2_DIR)


def _load_one_model(weights_path: str, device: str):
    """Load a single Segmenter (UNet) from AutoMorph M2_Vessel_seg."""
    _inject_m2_path()
    from model import Segmenter  # AutoMorph's model.py

    net = Segmenter(input_channels=3, n_filters=32, n_classes=1, bilinear=False)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch without weights_only
        state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def _preprocess(img_bgr: np.ndarray) -> "torch.Tensor":
    """
    Convert BGR fundus image to the tensor expected by AutoMorph M2_Vessel_seg.
    Steps:
      1. BGR → RGB
      2. Resize to 912 × 912 (AutoMorph uniform size)
      3. Per-image z-score normalisation using FOV pixels (R > 40)
      4. HWC → NCHW float32 tensor
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_rs = cv2.resize(rgb, (_TARGET_SIZE, _TARGET_SIZE),
                        interpolation=cv2.INTER_LANCZOS4)
    arr = rgb_rs.astype(np.float32)

    fov = arr[..., 0] > _PTHRESHOLD
    if fov.sum() > 64:
        mean_v = np.mean(arr[fov], axis=0)
        std_v  = np.std(arr[fov],  axis=0) + 1e-6
    else:
        mean_v = arr.mean(axis=(0, 1))
        std_v  = arr.std(axis=(0, 1)) + 1e-6

    arr = (arr - mean_v) / std_v
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def _prune_small(mask_u8: np.ndarray, min_area: int = 40) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 255
    return out


# ── Public: load ensemble ──────────────────────────────────────────────────────

def load_vessel_seg_ensemble(n_models: int = 3, device: str = "cpu"):
    """
    Load at most n_models from the AutoMorph vessel segmentation ensemble.

    Parameters
    ----------
    n_models : int
        Maximum number of models to load (1–10). Fewer models = faster,
        but lower ensemble accuracy. 3 is a good balance for CPU inference.
    device : str
        PyTorch device string ('cpu', 'cuda', 'cuda:0', …).

    Returns
    -------
    list[nn.Module] | None
        Loaded and eval-mode models, or None if torch / weights unavailable.
    """
    if not _TORCH_AVAILABLE:
        warnings.warn("[deep_backend] torch not installed. Deep backend unavailable.")
        return None
    if not _WEIGHT_PATHS:
        warnings.warn(
            "[deep_backend] No M2_Vessel_seg weights found at:\n  " + _WEIGHTS_ROOT
        )
        return None

    paths = _WEIGHT_PATHS[:n_models]
    models = []
    for p in paths:
        try:
            m = _load_one_model(p, device)
            models.append(m)
        except Exception as e:
            warnings.warn(f"[deep_backend] Skipping {os.path.basename(os.path.dirname(p))}: {e}")

    if not models:
        warnings.warn("[deep_backend] All model loads failed. Deep backend unavailable.")
        return None

    return models


# ── Public: inference ──────────────────────────────────────────────────────────

def segment_vessels_deep(
    img_bgr: np.ndarray,
    models: list,
    device: str = "cpu",
):
    """
    Run the AutoMorph ensemble on img_bgr and return vessel segmentation.

    Parameters
    ----------
    img_bgr  : H×W×3 uint8 BGR image (any resolution; resized internally).
    models   : list of loaded Segmenter models from load_vessel_seg_ensemble().
    device   : PyTorch device string.

    Returns
    -------
    vessel_mask_u8 : H×W uint8 binary mask (0 / 255), original resolution.
    prob_map_f32   : H×W float32 ensemble probability (0–1), original resolution.
    """
    if not _TORCH_AVAILABLE or not models:
        H, W = img_bgr.shape[:2]
        return np.zeros((H, W), dtype=np.uint8), np.zeros((H, W), dtype=np.float32)

    H, W = img_bgr.shape[:2]
    tensor = _preprocess(img_bgr).to(device)

    prob_sum = None
    with torch.no_grad():
        for m in models:
            out = torch.sigmoid(m(tensor))  # (1, 1, 912, 912)
            p = out[0, 0].cpu().numpy().astype(np.float32)
            prob_sum = p if prob_sum is None else prob_sum + p

    prob_avg = prob_sum / len(models)

    # Resize probability map back to the original image size
    prob_full = cv2.resize(prob_avg, (W, H), interpolation=cv2.INTER_LINEAR)
    vessel_mask = ((prob_full >= 0.5).astype(np.uint8) * 255)
    vessel_mask = _prune_small(vessel_mask, min_area=40)

    return vessel_mask, prob_full


# ── Public: drop-in replacement ────────────────────────────────────────────────

def get_enhanced_vessels_deep(
    img_bgr: np.ndarray,
    models=None,
    device: str = "cpu",
    return_details: bool = False,
):
    """
    Drop-in replacement for riched_image.get_enhanced_vessels().

    Uses AutoMorph's deep vessel segmentation as the mask source, but keeps
    the classical FOV mask and green-channel enhancement for feature extraction
    compatibility.

    Falls back to riched_image.get_enhanced_vessels() if torch is unavailable
    or models cannot be loaded.

    Returns
    -------
    (en_green, vessel_mask, skeleton, img_no_bg, proc_mask)
    """
    import riched_image  # local import to avoid circular dependency at module level

    # Automatic model loading
    if models is None:
        if _TORCH_AVAILABLE:
            models = load_vessel_seg_ensemble(n_models=3, device=device)
        else:
            models = None

    # Graceful fallback
    if not models:
        return riched_image.get_enhanced_vessels(img_bgr, return_details=return_details)

    # Classical pipeline for FOV mask and green enhancement (kept for feature compat.)
    en_green_cls, _, _, img_no_bg, proc_mask = riched_image.get_enhanced_vessels(img_bgr)

    # Deep segmentation
    vessel_mask, _ = segment_vessels_deep(img_bgr, models, device=device)

    # Restrict to classical FOV
    vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=proc_mask)

    # Skeleton + prune (same thresholds as classical pipeline)
    skeleton = (skeletonize(vessel_mask > 0).astype(np.uint8)) * 255
    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=proc_mask)

    n_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8
    )
    sk_clean = np.zeros_like(skeleton)
    for i in range(1, n_s):
        if stats_s[i, cv2.CC_STAT_AREA] >= 12:
            sk_clean[lab_s == i] = 255
    skeleton = sk_clean

    # Use classical enhanced green (CLAHE-processed) for feature colour features
    en_green = en_green_cls

    if return_details:
        details = {
            "raw_vessel_mask": vessel_mask.copy(),
        }
        return en_green, vessel_mask, skeleton, img_no_bg, proc_mask, details

    return en_green, vessel_mask, skeleton, img_no_bg, proc_mask
