# av_backend.py — Deep artery/vein classification using AutoMorph M2_Artery_vein.
#
# Pipeline:
#   Cổ điển: tiền xử lý ảnh (resize, z-score normalisation với FOV mask)
#   Deep:    Generator_branch (artery/vein) + Generator_main ensemble inference
#   Cổ điển: decode class map → artery/vein masks + remove_small_objects + resize
#
# Public API
# ----------
#   load_av_ensemble(n_models=2, device='cpu')
#       -> list[dict{'G', 'GA', 'GV'}] | None
#   segment_av_deep(img_bgr, ensemble, device='cpu')
#       -> (artery_mask_u8, vein_mask_u8)
#          Both H×W uint8 binary masks (255=vessel) at original image size.

import os
import sys
import warnings

import cv2
import numpy as np
from skimage.morphology import remove_small_objects

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_AV_DIR = os.path.join(_BASE, "Lka-suggest", "Fundus", "M2_Artery_vein")
_AV_SCRIPTS_DIR = os.path.join(_AV_DIR, "scripts")
_AV_WEIGHTS_ROOT = os.path.join(_AV_DIR, "ALL-AV")

_SEEDS = [28, 30, 32, 34, 36, 38, 40, 42]
_SEED_CKPT_DIRS = [
    os.path.join(
        _AV_WEIGHTS_ROOT,
        f"20210724_ALL-AV_randomseed_{s}",
        "Discriminator_unet",
    )
    for s in _SEEDS
    if os.path.isdir(
        os.path.join(
            _AV_WEIGHTS_ROOT,
            f"20210724_ALL-AV_randomseed_{s}",
            "Discriminator_unet",
        )
    )
]

_TARGET_SIZE = 912   # uniform image size expected by AutoMorph A/V models
_PTHRESHOLD = 40     # red-channel FOV threshold for z-score normalisation

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _inject_av_scripts_path():
    """Add M2_Artery_vein/scripts/ to sys.path so model.py is importable directly."""
    if _AV_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _AV_SCRIPTS_DIR)


def _load_state(path: str, device: str) -> dict:
    """Load a checkpoint, handling both nested and plain state_dict formats."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state:
                return state[key]
    return state


def _load_one_av_triplet(seed_ckpt_dir: str, device: str) -> dict:
    """
    Load (Generator_main, Generator_branch_A, Generator_branch_V) for one seed.
    Generator_main  : n_classes=4 (bg=0, artery=1, vein=2, overlap=3)
    Generator_branch: n_classes=1 (binary), produces feature maps passed to G_main
    """
    _inject_av_scripts_path()
    from model import Generator_main, Generator_branch  # noqa: E402

    net_G = Generator_main(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
    net_GA = Generator_branch(input_channels=3, n_filters=32, n_classes=1, bilinear=False)
    net_GV = Generator_branch(input_channels=3, n_filters=32, n_classes=1, bilinear=False)

    net_G.load_state_dict(
        _load_state(os.path.join(seed_ckpt_dir, "CP_best_F1_all.pth"), device)
    )
    net_GA.load_state_dict(
        _load_state(os.path.join(seed_ckpt_dir, "CP_best_F1_A.pth"), device)
    )
    net_GV.load_state_dict(
        _load_state(os.path.join(seed_ckpt_dir, "CP_best_F1_V.pth"), device)
    )

    for net in (net_G, net_GA, net_GV):
        net.to(device)
        net.eval()

    return {"G": net_G, "GA": net_GA, "GV": net_GV}


def _preprocess_av(img_bgr: np.ndarray) -> "torch.Tensor":
    """
    Tiền xử lý ảnh cổ điển trước khi đưa vào Generator:
    BGR → RGB → resize 912×912 → z-score normalisation theo FOV pixels → tensor (1,3,912,912).
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_rs = cv2.resize(rgb, (_TARGET_SIZE, _TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
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

def load_av_ensemble(n_models: int = 2, device: str = "cpu"):
    """
    Load up to n_models (G, GA, GV) triplets for deep A/V segmentation.

    Parameters
    ----------
    n_models : int
        Maximum number of seed triplets to load (1–8).
        Lower values are faster; 2 gives a good ensemble on CPU.
    device : str
        PyTorch device string ('cpu', 'cuda', …).

    Returns
    -------
    list[dict] | None
        Each dict has keys 'G', 'GA', 'GV' → loaded eval-mode models.
        Returns None if torch / weights unavailable.
    """
    if not _TORCH_AVAILABLE:
        warnings.warn("[av_backend] torch not installed. A/V deep backend unavailable.")
        return None
    if not _SEED_CKPT_DIRS:
        warnings.warn(
            f"[av_backend] No M2_Artery_vein weights found at:\n  {_AV_WEIGHTS_ROOT}"
        )
        return None

    dirs = _SEED_CKPT_DIRS[:n_models]
    ensemble = []
    for d in dirs:
        try:
            triplet = _load_one_av_triplet(d, device)
            ensemble.append(triplet)
        except Exception as e:
            seed_name = os.path.basename(os.path.dirname(d))
            warnings.warn(f"[av_backend] Skipping {seed_name}: {e}")

    if not ensemble:
        warnings.warn("[av_backend] All A/V model loads failed.")
        return None

    return ensemble


# ── Public: inference ──────────────────────────────────────────────────────────

def segment_av_deep(
    img_bgr: np.ndarray,
    ensemble: list,
    device: str = "cpu",
):
    """
    Run the Generator ensemble to produce pixel-level artery/vein masks.

    Inference flow (per seed):
      1. G_A(img) → logit_a, x_a_final   (Generator_branch artery feature map)
      2. G_V(img) → logit_v, x_v_final   (Generator_branch vein feature map)
      3. G(img, x_a_final, x_v_final) → logits_main  (4-class: bg/artery/vein/overlap)
    Average softmax across seeds → argmax → decode → morphological post-processing.

    Parameters
    ----------
    img_bgr  : H×W×3 uint8 BGR image (any resolution; resized internally to 912×912).
    ensemble : list of dicts from load_av_ensemble().
    device   : PyTorch device string.

    Returns
    -------
    artery_mask : H×W uint8 (255=artery pixel)  at original image size
    vein_mask   : H×W uint8 (255=vein pixel)    at original image size
    """
    h, w = img_bgr.shape[:2]
    tensor = _preprocess_av(img_bgr).to(device)

    avg_softmax = None   # accumulated (1, 4, 912, 912)

    with torch.no_grad():
        for triplet in ensemble:
            # Branch networks: returns (logit, x_final_feature_map)
            _, x_a = triplet["GA"](tensor)
            _, x_v = triplet["GV"](tensor)

            # Main network: returns (logit, s1, s2, s3)
            logits, _, _, _ = triplet["G"](tensor, x_a, x_v)

            sm = F.softmax(logits, dim=1).cpu()
            if avg_softmax is None:
                avg_softmax = sm
            else:
                avg_softmax = avg_softmax + sm

    avg_softmax = avg_softmax / len(ensemble)
    pred = torch.argmax(avg_softmax, dim=1)[0].numpy()   # (912, 912) int

    # ── Decode class map ───────────────────────────────────────────────────────
    # 0 = background, 1 = artery, 2 = vein, 3 = overlap (both artery+vein)
    artery_bool = (pred == 1) | (pred == 3)
    vein_bool = (pred == 2) | (pred == 3)

    # ── Classical morphological post-processing (AutoMorph standard) ──────────
    artery_bool = remove_small_objects(artery_bool, min_size=30, connectivity=2)
    vein_bool = remove_small_objects(vein_bool, min_size=30, connectivity=2)

    artery_912 = artery_bool.astype(np.uint8) * 255
    vein_912 = vein_bool.astype(np.uint8) * 255

    # Resize to original image size
    artery_full = cv2.resize(artery_912, (w, h), interpolation=cv2.INTER_NEAREST)
    vein_full = cv2.resize(vein_912, (w, h), interpolation=cv2.INTER_NEAREST)

    return artery_full, vein_full
