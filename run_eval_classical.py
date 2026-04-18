import json
import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import av_classifier
import feature_extract
import input_data
import riched_image
from constant import IMG_SIZE, RISK_DECISION_THRESHOLD

def _metrics_at_threshold(y, proba, threshold):
    pred = (proba >= float(threshold)).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn) / max(1, float(tn + fp))
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }

def main():
    risk = joblib.load("models/stroke_risk_model.pkl")
    av = av_classifier.load_av_classifier()
    
    X, y = [], []
    for label in [0, 1]:
        folder = f"test/{label}"
        if not os.path.isdir(folder): continue
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            img = cv2.imread(p)
            if img is None: continue
            try:
                img = input_data.standardize_fundus_image(img, IMG_SIZE)
                en, mask, skel, _bg, fov, pipe = riched_image.get_enhanced_vessels(img, return_details=True)
                feats, _, _ = feature_extract.extract_features(
                    mask, en, skeleton=skel, img_bgr=img, av_model=av, fov_mask=fov,
                    return_details=True, raw_vessel_mask=pipe.get("raw_vessel_mask")
                )
                X.append(feats)
                y.append(label)
            except: continue
            
    X = np.array(X)
    y = np.array(y)
    probas = risk.predict_proba(X)[:, 1]
    
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = [_metrics_at_threshold(y, probas, t) for t in thresholds]
    best = max(results, key=lambda r: (r["recall"] + r["specificity"], r["f1"]))
    
    output = {
        "backend": "classical",
        "threshold_scan": results,
        "best_threshold_row": best
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
