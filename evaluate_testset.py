import json
import os

import cv2
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import av_classifier
import feature_extract
import input_data
import riched_image
from constant import IMG_SIZE, RISK_DECISION_THRESHOLD


def _metrics_at_threshold(y, proba, threshold):
    pred = (proba >= float(threshold)).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    specificity = float(cm[0, 0]) / max(1, float(cm[0, 0] + cm[0, 1]))
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def _build_threshold_scan(y, proba):
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    rows = [_metrics_at_threshold(y, proba, thr) for thr in thresholds]
    best_row = max(rows, key=lambda row: (row["recall_sensitivity"] + row["specificity"], row["f1"], row["accuracy"]))
    return rows, best_row


def main():
    risk = joblib.load("models/stroke_risk_model.pkl")
    av = av_classifier.load_av_classifier()

    X, y = [], []
    for label in [0, 1]:
        folder = f"test/{label}"
        if not os.path.isdir(folder):
            continue

        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            img = cv2.imread(p)
            if img is None:
                continue

            try:
                img = input_data.standardize_fundus_image(img, IMG_SIZE)
                en, mask, skel, _img_no_bg, fov, pipe_details = riched_image.get_enhanced_vessels(img, return_details=True)
                feats, _, _details = feature_extract.extract_features(
                    mask,
                    en,
                    skeleton=skel,
                    img_bgr=img,
                    av_model=av,
                    fov_mask=fov,
                    return_details=True,
                    raw_vessel_mask=pipe_details.get("raw_vessel_mask"),
                )
                X.append(feats)
                y.append(label)
            except Exception:
                continue

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    proba = risk.predict_proba(X)[:, 1]
    metrics_at_deploy = _metrics_at_threshold(y, proba, RISK_DECISION_THRESHOLD)
    threshold_scan, best_threshold = _build_threshold_scan(y, proba)

    metrics = {
        "threshold": float(RISK_DECISION_THRESHOLD),
        "n_samples": int(len(y)),
        "accuracy": metrics_at_deploy["accuracy"],
        "precision": metrics_at_deploy["precision"],
        "recall_sensitivity": metrics_at_deploy["recall_sensitivity"],
        "specificity": metrics_at_deploy["specificity"],
        "f1": metrics_at_deploy["f1"],
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else None,
        "confusion_matrix": metrics_at_deploy["confusion_matrix"],
        "threshold_scan": threshold_scan,
        "best_threshold_by_balanced_sum": best_threshold,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved reports/test_metrics.json")


if __name__ == "__main__":
    main()
