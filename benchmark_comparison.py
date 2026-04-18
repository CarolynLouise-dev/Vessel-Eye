# benchmark_comparison.py
# Compare classical Frangi pipeline vs AutoMorph DL vessel segmentation
# on the test set. Outputs a classification metrics table, per-image feature
# comparison, and a visual side-by-side grid.
#
# Usage
# -----
#   python benchmark_comparison.py               # defaults: 5 images/class, 3 models
#   python benchmark_comparison.py --n_images 10 --n_models 5
#   python benchmark_comparison.py --n_images 10 --device cuda

import argparse
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

import av_classifier
import feature_extract
import input_data
import riched_image
from constant import IMG_SIZE, RISK_DECISION_THRESHOLD


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_test_images(n_per_class: int):
    """Return up to n_per_class (filename, label, path, bgr_img) tuples per class."""
    results = []
    for label in [0, 1]:
        folder = os.path.join(_BASE, "test", str(label))
        if not os.path.isdir(folder):
            continue
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:n_per_class]
        for fn in files:
            p = os.path.join(folder, fn)
            img = cv2.imread(p)
            if img is not None:
                results.append((fn, label, p, img))
    return results


# ── Pipeline runners ───────────────────────────────────────────────────────────

def _run_classical(img_bgr, av_model):
    """Run full classical pipeline; return (feats, mask, en, skel, fov, img_r, time_s)."""
    img_r = input_data.standardize_fundus_image(img_bgr, IMG_SIZE)
    t0 = time.perf_counter()
    en, mask, skel, _, fov, pipe_details = riched_image.get_enhanced_vessels(img_r, return_details=True)
    t_proc = time.perf_counter() - t0
    feats, _, det = feature_extract.extract_features(
        mask, en,
        skeleton=skel, img_bgr=img_r,
        av_model=av_model, fov_mask=fov,
        return_details=True,
        raw_vessel_mask=pipe_details.get("raw_vessel_mask"),
    )
    return feats, mask, en, skel, fov, img_r, float(t_proc)


def _run_deep(img_bgr, av_model, deep_models, device, img_r=None):
    """Run deep vessel segmentation then classical feature extraction."""
    import deep_backend
    if img_r is None:
        img_r = input_data.standardize_fundus_image(img_bgr, IMG_SIZE)

    # Classical pass for FOV mask + enhanced green
    en, _, _, img_no_bg, fov, _pipe_details = riched_image.get_enhanced_vessels(img_r, return_details=True)

    t0 = time.perf_counter()
    vessel_mask, prob_map = deep_backend.segment_vessels_deep(img_r, deep_models, device=device)
    t_proc = time.perf_counter() - t0

    vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=fov)

    from skimage.morphology import skeletonize as _sk
    skel = (_sk(vessel_mask > 0).astype(np.uint8)) * 255
    skel = cv2.bitwise_and(skel, skel, mask=fov)
    n_s, lab_s, stats_s, _ = cv2.connectedComponentsWithStats(
        (skel > 0).astype(np.uint8), connectivity=8
    )
    sk_clean = np.zeros_like(skel)
    for i in range(1, n_s):
        if stats_s[i, cv2.CC_STAT_AREA] >= 12:
            sk_clean[lab_s == i] = 255
    skel = sk_clean

    feats, _, det = feature_extract.extract_features(
        vessel_mask, en,
        skeleton=skel, img_bgr=img_r,
        av_model=av_model, fov_mask=fov,
        return_details=True,
        raw_vessel_mask=vessel_mask,
    )
    return feats, vessel_mask, en, skel, fov, img_r, float(t_proc)


# ── Classification ─────────────────────────────────────────────────────────────

def _classify(feats, rf_model):
    proba = float(rf_model.predict_proba([feats])[0][1])
    return proba


# ── Classification metrics ─────────────────────────────────────────────────────

def _compute_metrics(rows: list) -> dict:
    if not rows:
        return {}
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    y     = np.array([r["label"] for r in rows], dtype=int)
    proba = np.array([r["prob"]  for r in rows], dtype=float)
    pred  = (proba >= RISK_DECISION_THRESHOLD).astype(int)
    try:
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else None
    except Exception:
        auc = None

    feature_means = {}
    for fn in feature_extract.FEATURE_NAMES:
        vals = [r[fn] for r in rows if fn in r]
        feature_means[fn] = float(np.mean(vals)) if vals else None

    return {
        "n_images":          len(rows),
        "accuracy":          float(accuracy_score(y, pred)),
        "recall_sensitivity":float(recall_score(y, pred, zero_division=0)),
        "precision":         float(precision_score(y, pred, zero_division=0)),
        "f1":                float(f1_score(y, pred, zero_division=0)),
        "auc":               auc,
        "mean_time_s":       float(np.mean([r["time_s"] for r in rows])),
        "feature_means":     feature_means,
    }


# ── Visualisation ──────────────────────────────────────────────────────────────

def _make_visual_row(orig_bgr, mask_cls, mask_deep, thumb_h: int = 240):
    """
    Create a horizontal strip for one image:
      original | classical mask (cyan tint) | deep mask (orange tint)
    """
    def _thumb(img):
        return cv2.resize(img, (thumb_h, thumb_h))

    orig = _thumb(orig_bgr)

    def _colorise(mask, bgr_tint):
        m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        tint = np.full_like(m3, bgr_tint)
        colored = np.where(m3 > 0, cv2.addWeighted(m3, 0.55, tint, 0.45, 0), m3)
        return _thumb(colored)

    col_cls  = _colorise(mask_cls,  (255, 200, 0))    # cyan-ish
    col_deep = _colorise(mask_deep, (0,   140, 255))  # orange
    return np.hstack([orig, col_cls, col_deep])


def _draw_row_label(row_img, fname, label, prob_cls, prob_deep):
    h, w = row_img.shape[:2]
    out = row_img.copy()
    lbl_str  = "RISK" if label == 1 else "NORMAL"
    deep_str = f"DL:{prob_deep:.2f}" if prob_deep is not None else ""
    text = f"{fname}  GT={lbl_str}  Classical:{prob_cls:.2f}  {deep_str}"
    cv2.rectangle(out, (0, 0), (w, 22), (20, 20, 20), -1)
    cv2.putText(out, text, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA)
    return out


def _draw_grid_header(width: int, has_deep: bool) -> np.ndarray:
    hdr = np.full((32, width, 3), 14, dtype=np.uint8)
    labels = ["Original", "Classical (Frangi+CLAHE)",
              "AutoMorph DL (M2_Vessel_seg)" if has_deep else "AutoMorph DL (unavailable)"]
    col_w = width // 3
    for i, lbl in enumerate(labels):
        x = i * col_w + 8
        cv2.putText(hdr, lbl, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 210, 255), 1, cv2.LINE_AA)
    return hdr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark classical Frangi pipeline vs AutoMorph DL vessel segmentation"
    )
    parser.add_argument("--n_images", type=int, default=5,
                        help="Images per class to benchmark (default 5)")
    parser.add_argument("--n_models",  type=int, default=3,
                        help="AutoMorph ensemble size 1-10 (default 3, ~2x slower per step up)")
    parser.add_argument("--device",   type=str, default="cpu",
                        help="PyTorch device (cpu | cuda | cuda:0, …)")
    args = parser.parse_args()

    # ── Load classifier + AV model ──
    import joblib
    rf_path = os.path.join(_BASE, "models", "stroke_risk_model.pkl")
    if not os.path.exists(rf_path):
        print("[ERROR] stroke_risk_model.pkl not found. Run: python training_model.py")
        sys.exit(1)
    rf_model = joblib.load(rf_path)
    av_model = av_classifier.load_av_classifier()

    # ── Load AutoMorph deep models ──
    import deep_backend
    print(f"[INFO] Loading AutoMorph ensemble ({args.n_models} model(s)) on {args.device} …")
    deep_models = deep_backend.load_vessel_seg_ensemble(
        n_models=args.n_models, device=args.device
    )
    has_deep = bool(deep_models)
    if has_deep:
        print(f"[OK]   {len(deep_models)} AutoMorph model(s) loaded.")
    else:
        print("[WARN] AutoMorph DL unavailable — visual comparison will duplicate classical mask.")

    # ── Load test images ──
    images = _load_test_images(args.n_images)
    if not images:
        print("[ERROR] No test images found in test/0 or test/1.")
        sys.exit(1)
    n_class0 = sum(1 for _, l, *_ in images if l == 0)
    n_class1 = sum(1 for _, l, *_ in images if l == 1)
    print(f"[INFO] Images: {len(images)} total  (class 0: {n_class0}, class 1: {n_class1})\n")

    # ── Benchmark loop ──
    rows_cls   = []
    rows_deep  = []
    visual_rows = []
    feat_names  = feature_extract.FEATURE_NAMES

    for idx, (fname, label, fpath, img_bgr) in enumerate(images):
        print(f"  [{idx+1:02d}/{len(images)}] {fname:<30} label={label}", end="", flush=True)

        # Classical ──────────────────────────────────────────────────────────
        feats_c, mask_c, en_c, skel_c, fov_c, img_r, t_c = _run_classical(img_bgr, av_model)
        prob_c = _classify(feats_c, rf_model)
        row_c  = {"fname": fname, "label": label, "prob": prob_c, "time_s": t_c}
        for i, fn in enumerate(feat_names):
            row_c[fn] = feats_c[i]
        rows_cls.append(row_c)
        print(f"  cls: {t_c:5.1f}s p={prob_c:.3f}", end="", flush=True)

        # Deep ───────────────────────────────────────────────────────────────
        mask_d = mask_c  # fallback: same as classical
        prob_d = None
        if has_deep:
            try:
                feats_d, mask_d, _en_d, _sk_d, _fov_d, _, t_d = _run_deep(
                    img_bgr, av_model, deep_models, args.device, img_r=img_r
                )
                prob_d = _classify(feats_d, rf_model)
                row_d  = {"fname": fname, "label": label, "prob": prob_d, "time_s": t_d}
                for i, fn in enumerate(feat_names):
                    row_d[fn] = feats_d[i]
                rows_deep.append(row_d)
                print(f"  dl: {t_d:5.1f}s p={prob_d:.3f}", end="", flush=True)
            except Exception as e:
                print(f"  dl: ERROR({e})", end="", flush=True)

        print()

        # Visual row ─────────────────────────────────────────────────────────
        vrow = _make_visual_row(img_r, mask_c, mask_d)
        vrow = _draw_row_label(vrow, fname, label, prob_c, prob_d)
        visual_rows.append(vrow)

    # ── Compute metrics ──
    m_cls  = _compute_metrics(rows_cls)
    m_deep = _compute_metrics(rows_deep) if rows_deep else {"note": "AutoMorph DL not available"}

    # ── Print comparison table ──
    print("\n" + "=" * 70)
    print("  BENCHMARK: Classical Frangi  vs  AutoMorph DL (M2_Vessel_seg)")
    print("=" * 70)
    col_a = "Classical"
    col_b = "AutoMorph DL" if has_deep else "AutoMorph DL (N/A)"
    print(f"  {'Metric':<30} {col_a:>14} {col_b:>14}")
    print("-" * 70)

    cmp_rows = [
        ("n_images",           "Images evaluated"),
        ("accuracy",           "Accuracy"),
        ("recall_sensitivity", "Recall (Sensitivity)"),
        ("precision",          "Precision"),
        ("f1",                 "F1 Score"),
        ("auc",                "AUC-ROC"),
        ("mean_time_s",        "Time / image (s)"),
    ]
    for key, label_str in cmp_rows:
        v_c = m_cls.get(key)
        v_d = m_deep.get(key)
        s_c = f"{v_c:.4f}" if isinstance(v_c, float) else str(v_c)
        s_d = f"{v_d:.4f}" if isinstance(v_d, float) else (str(v_d) if v_d is not None else "N/A")
        print(f"  {label_str:<30} {s_c:>14} {s_d:>14}")

    print("-" * 70)
    print(f"  {'Feature means':^58}")
    print("-" * 70)
    for fn in feat_names:
        vc = m_cls.get("feature_means", {}).get(fn)
        vd = m_deep.get("feature_means", {}).get(fn) if has_deep else None
        sc = f"{vc:.4f}" if isinstance(vc, float) else "N/A"
        sd = f"{vd:.4f}" if isinstance(vd, float) else "N/A"
        print(f"  {fn:<30} {sc:>14} {sd:>14}")

    # Classification comparison context
    print("=" * 70)
    prev_json = os.path.join(_BASE, "reports", "test_metrics.json")
    if os.path.exists(prev_json):
        with open(prev_json, encoding="utf-8") as f:
            prev = json.load(f)
        print(f"\n  [Context] Full test-set metrics (evaluate_testset.py, {prev.get('n_samples','?')} imgs):")
        print(f"    Accuracy={prev.get('accuracy',0):.4f}  Recall={prev.get('recall_sensitivity',0):.4f}"
              f"  F1={prev.get('f1',0):.4f}  AUC={prev.get('auc') or 'N/A'}")

    # ── Save JSON report ──
    report = {
        "threshold": float(RISK_DECISION_THRESHOLD),
        "n_models_deep": len(deep_models) if deep_models else 0,
        "classical": m_cls,
        "deep": m_deep,
    }
    os.makedirs(os.path.join(_BASE, "reports"), exist_ok=True)
    report_path = os.path.join(_BASE, "reports", "benchmark_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  [Saved] {report_path}")

    # ── Save visual grid ──
    if visual_rows:
        max_w = max(r.shape[1] for r in visual_rows)
        padded = []
        for r in visual_rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        hdr  = _draw_grid_header(max_w, has_deep)
        grid = np.vstack([hdr] + padded)
        grid_path = os.path.join(_BASE, "reports", "benchmark_grid.png")
        cv2.imwrite(grid_path, grid)
        print(f"  [Saved] {grid_path}")


if __name__ == "__main__":
    main()
