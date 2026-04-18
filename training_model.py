import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from constant import MODEL_PATH, MODEL_DIR, IMG_SIZE
import riched_image, feature_extract
from riched_image import resize_with_pad

def _metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= float(threshold)).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    specificity = float(cm[0, 0]) / max(1.0, float(cm[0, 0] + cm[0, 1]))
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall_sensitivity": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "f2_score": round(float(f2), 4),
        "confusion_matrix": cm.tolist(),
    }


def _select_operating_threshold(y_true, y_prob):
    candidates = np.arange(0.15, 0.56, 0.02)
    best = None
    best_metrics = None
    for thr in candidates:
        metrics = _metrics_at_threshold(y_true, y_prob, float(thr))
        ranking_key = (
            float(metrics["recall_sensitivity"]),
            float(metrics["f2_score"]),
            float(metrics["f1_score"]),
            float(metrics["specificity"]),
        )
        if best is None or ranking_key > best:
            best = ranking_key
            best_metrics = metrics
    return best_metrics

    print("--- BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG CHUẨN Y KHOA ---")

    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Thư mục '{dataset_path}' không tồn tại!")
        return

    n_total = n_ok = n_err = 0

    for label in ["0", "1"]:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Đang xử lý nhóm nhãn {label}: {len(files)} ảnh")

        for filename in files:
            full_path = os.path.join(folder, filename)
            img = cv2.imread(full_path)
            if img is None: continue

            # 2. KIỂM TRA CHẤT LƯỢNG ẢNH TRƯỚC KHI XỬ LÝ
            if not feature_extract.is_valid_fundus_image(img):
                print(f"⏩ Bỏ qua ảnh {filename}: Không đủ tiêu chuẩn (Ảnh zoom/mất viền).")
                continue  # Skip, nhảy sang ảnh tiếp theo luôn

            # 3. Nếu hợp lệ mới đưa vào pipeline trích xuất đặc trưng
            print(f"✅ Đang xử lý {filename}...")
            # features = extract_features(img, ...)

            # GIỮ NGUYÊN TỶ LỆ KHI ĐẨY VÀO HUẤN LUYỆN
            img = resize_with_pad(img, IMG_SIZE)

            try:
                en, mask, skel, img_no_bg, fov_mask = riched_image.get_enhanced_vessels(img)

                # TRUYỀN ẢNH GỐC ĐỂ LẤY KHÔNG GIAN MÀU
                ai_feats, _, _, _, _ = feature_extract.extract_features(img, mask, en, skel, fov_mask)

                X_features.append(ai_feats)
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

            except Exception as e:
                print(f"❌ Lỗi tại file {filename}: {e}")

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
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [150, 250],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", {0: 1.0, 1: 2.0}, {0: 1.0, 1: 3.0}],
    }
    recall_focused_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=skf,
        scoring=recall_focused_scorer,
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
    cv_f2  = cross_val_score(cv_model, X, y, cv=skf, scoring=recall_focused_scorer)
    cv_rec = cross_val_score(cv_model, X, y, cv=skf, scoring="recall")
    cv_auc = cross_val_score(cv_model, X, y, cv=skf, scoring="roc_auc")

    print(f"  CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"  CV F1:       {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  CV F2:       {cv_f2.mean():.4f} +/- {cv_f2.std():.4f}")
    print(f"  CV Recall:   {cv_rec.mean():.4f} +/- {cv_rec.std():.4f}")
    print(f"  CV AUC-ROC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    # Step 3: 80/20 Hold-out
    print("[3/4] Hold-out evaluation (80/20) ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f"\n--- ĐANG TIẾN HÀNH TỐI ƯU HÓA MÔ HÌNH ---")

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )

    grid_search.fit(X_features, y_labels)

    best_model = grid_search.best_estimator_

    print(f"\n✅ TÌM KIẾM HOÀN TẤT!")
    print(f"🏆 Bộ tham số chiến thắng: {grid_search.best_params_}")
    print(f"🎯 Độ chính xác (Accuracy): {grid_search.best_score_ * 100:.2f}%")

    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 XONG: Model xuất sắc nhất đã được lưu tại {MODEL_PATH}")


if __name__ == "__main__":
    train_optimized("dataset")
