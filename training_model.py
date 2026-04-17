import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import os
from datetime import datetime

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from constant import IMG_SIZE, MODEL_DIR, MODEL_PATH
import feature_extract
import input_data
import riched_image
import av_classifier

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def _get_vessel_bundle(img, backend="classical", deep_models=None, device="cpu"):
    if backend == "automorph":
        import deep_backend
        if not deep_models:
            raise RuntimeError("AutoMorph backend requested but deep models are unavailable.")
        return deep_backend.get_enhanced_vessels_deep(img, models=deep_models, device=device)
    return riched_image.get_enhanced_vessels(img)


def _collect_features(dataset_path, av_model=None, backend="classical", deep_models=None, device="cpu",
                       od_models=None, av_ensemble=None):
    X_features, y_labels = [], []

    if not os.path.exists(dataset_path):
        print(f"ERROR: '{dataset_path}' not found!")
        return X_features, y_labels

    n_total = n_ok = n_err = 0

    for label in ["0", "1"]:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"  Label {label}: {len(files)} images")

        for filename in files:
            full_path = os.path.join(folder, filename)
            img_raw = cv2.imread(full_path)
            if img_raw is None:
                continue
            img = input_data.standardize_fundus_image(img_raw, IMG_SIZE)
            n_total += 1

            try:
                en, mask, skeleton, *_ = _get_vessel_bundle(
                    img,
                    backend=backend,
                    deep_models=deep_models,
                    device=device,
                )

                od_center_override = od_radius_override = None
                if od_models:
                    try:
                        import od_backend
                        od_center_override, od_radius_override, _, _ = \
                            od_backend.detect_optic_disc_deep(img, od_models, device)
                    except Exception:
                        pass

                artery_mask = vein_mask = None
                if av_ensemble:
                    try:
                        import av_backend
                        artery_mask, vein_mask = av_backend.segment_av_deep(
                            img, av_ensemble, device
                        )
                    except Exception:
                        pass

                feats, _ = feature_extract.extract_features(
                    mask, en, skeleton=skeleton,
                    img_bgr=img, av_model=av_model,
                    od_center_override=od_center_override,
                    od_radius_override=od_radius_override,
                    artery_mask=artery_mask,
                    vein_mask=vein_mask,
                )
                X_features.append(feats)
                y_labels.append(int(label))
                n_ok += 1

                if n_ok % 50 == 0:
                    print(f"    [{n_ok}/{n_total}] features extracted")

                del img_raw, img, en, mask
            except Exception:
                n_err += 1

    print(f"  Dataset: {n_ok} OK, {n_err} errors (total {n_total})")
    return X_features, y_labels


def train_optimized(dataset_path, backend="classical", deep_n_models=3, device="cpu",
                     od_n_models=3, av_n_models=2):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 50)
    print("  VESSEL-EYE Training Pipeline")
    print("=" * 50)

    # Step 0: Pre-train A/V Classifier
    print("[0/4] Pre-training A/V SVM Classifier ...")
    av_mdl = av_classifier.train_av_classifier(dataset_path)

    deep_models = None
    od_models = None
    av_ensemble = None
    if backend == "automorph":
        import deep_backend
        print(f"[0b/4] Loading AutoMorph vessel ensemble ({deep_n_models} model(s)) on {device} ...")
        deep_models = deep_backend.load_vessel_seg_ensemble(n_models=deep_n_models, device=device)
        if not deep_models:
            raise RuntimeError("AutoMorph backend requested but no vessel models could be loaded.")

        import od_backend as _od_backend
        print(f"[0c/4] Loading OD wnet ensemble ({od_n_models} model(s)) ...")
        od_models = _od_backend.load_od_ensemble(n_models=od_n_models, device=device)
        if not od_models:
            print("  WARNING: OD models unavailable; Zone B will use heuristic.")

        import av_backend as _av_backend
        print(f"[0d/4] Loading A/V Generator ensemble ({av_n_models} triplet(s)) ...")
        av_ensemble = _av_backend.load_av_ensemble(n_models=av_n_models, device=device)
        if not av_ensemble:
            print("  WARNING: A/V models unavailable; SVM classifier will be used.")

    # Step 1: Feature extraction
    print("[1/4] Extracting features ...")
    X_features, y_labels = _collect_features(
        dataset_path,
        av_model=av_mdl,
        backend=backend,
        deep_models=deep_models,
        device=device,
        od_models=od_models,
        av_ensemble=av_ensemble,
    )

    if not X_features:
        print("ERROR: No features extracted!")
        return None

    X = np.array(X_features)
    y = np.array(y_labels)
    n_samples = len(y)
    n_pos = int(y.sum())
    n_neg = n_samples - n_pos

    print(f"  Total: {n_samples} (label 0: {n_neg}, label 1: {n_pos})")

    # Step 2: 5-fold Stratified Cross-Validation + hyperparameter search
    print("[2/4] 5-fold Stratified Cross-Validation + hyperparameter search ...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_rf = RandomForestClassifier(
        n_estimators=150, class_weight="balanced", random_state=42
    )
    param_grid = {
        "max_depth": [4, 5, 6],
        "min_samples_leaf": [2],
    }
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=skf,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X, y)
    best_params = grid.best_params_
    print(f"  Best hyperparameters: {best_params}")

    cv_model = grid.best_estimator_
    cv_acc = cross_val_score(cv_model, X, y, cv=skf, scoring="accuracy")
    cv_f1  = cross_val_score(cv_model, X, y, cv=skf, scoring="f1")
    cv_auc = cross_val_score(cv_model, X, y, cv=skf, scoring="roc_auc")

    print(f"  CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"  CV F1:       {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  CV AUC-ROC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    # Step 3: 80/20 Hold-out
    print("[3/4] Hold-out evaluation (80/20) ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    final_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=best_params.get("max_depth", 6),
        min_samples_leaf=best_params.get("min_samples_leaf", 2),
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    holdout_acc = accuracy_score(y_test, y_pred)
    holdout_f1  = f1_score(y_test, y_pred, zero_division=0)
    holdout_auc = (
        roc_auc_score(y_test, y_prob)
        if len(np.unique(y_test)) > 1 else float("nan")
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    tn, fp, fn, tp = (
        confusion_matrix(y_test, y_pred).ravel()
        if len(np.unique(y_test)) == 2 else (0, 0, 0, 0)
    )
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"  Accuracy:    {holdout_acc:.4f}")
    print(f"  F1:          {holdout_f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")

    feat_names = feature_extract.FEATURE_NAMES
    importances = dict(zip(feat_names, final_model.feature_importances_.round(4).tolist()))

    # Step 4: Final model
    print("[4/4] Training final model on full dataset ...")
    full_model = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=42
    )
    full_model.fit(X, y)
    joblib.dump(full_model, MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")

    metrics = {
        "generated_at": datetime.now().isoformat(),
        "dataset": {
            "path": dataset_path,
            "backend": backend,
            "total_samples": n_samples,
            "label_0_normal": n_neg,
            "label_1_high_risk": n_pos,
        },
        "cross_validation_5fold": {
            "accuracy_mean": round(float(cv_acc.mean()), 4),
            "accuracy_std":  round(float(cv_acc.std()),  4),
            "f1_mean":       round(float(cv_f1.mean()),  4),
            "f1_std":        round(float(cv_f1.std()),   4),
            "auc_roc_mean":  round(float(cv_auc.mean()), 4),
            "auc_roc_std":   round(float(cv_auc.std()),  4),
        },
        "holdout_80_20": {
            "accuracy":           round(holdout_acc, 4),
            "f1_score":           round(holdout_f1,  4),
            "auc_roc":            round(holdout_auc, 4) if not np.isnan(holdout_auc) else None,
            "sensitivity_recall": round(sensitivity, 4) if not np.isnan(sensitivity) else None,
            "specificity":        round(specificity, 4) if not np.isnan(specificity) else None,
            "confusion_matrix":   cm,
        },
        "feature_importances": importances,
    }

    report_path = os.path.join(REPORTS_DIR, "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("  Training complete. See reports/metrics_report.json")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vessel-Eye random forest model")
    parser.add_argument("--dataset", default="dataset", help="Dataset root containing 0/ and 1/")
    parser.add_argument("--backend", choices=["classical", "automorph"], default="classical",
                        help="Vessel segmentation backend used before feature extraction")
    parser.add_argument("--n_models", type=int, default=3,
                        help="Number of AutoMorph vessel ensemble models when backend=automorph")
    parser.add_argument("--od_n_models", type=int, default=3,
                        help="Number of OD wnet ensemble models (AutoMorph mode only)")
    parser.add_argument("--av_n_models", type=int, default=2,
                        help="Number of A/V Generator ensemble triplets (AutoMorph mode only)")
    parser.add_argument("--device", default="cpu", help="Torch device for AutoMorph backend")
    args = parser.parse_args()
    train_optimized(
        args.dataset,
        backend=args.backend,
        deep_n_models=args.n_models,
        device=args.device,
        od_n_models=args.od_n_models,
        av_n_models=args.av_n_models,
    )