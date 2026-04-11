import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from constant import MODEL_PATH, MODEL_DIR, IMG_SIZE
import riched_image, feature_extract


def train_optimized(dataset_path):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    X_features = []
    y_labels = []

    print("--- BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG (BẢN ỔN ĐỊNH 4 CHỈ SỐ) ---")

    if not os.path.exists(dataset_path):
        print(f"Lỗi: Thư mục '{dataset_path}' không tồn tại!")
        return

    for label in ['0', '1']:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            print(f"Cảnh báo: Thư mục nhãn {label} không tìm thấy tại {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Đang xử lý nhóm nhãn {label}: {len(files)} ảnh...")

        for filename in files:
            full_path = os.path.join(folder, filename)
            img = cv2.imread(full_path)
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)

            try:
                en, mask, skel, *_ = riched_image.get_enhanced_vessels(img)

                # Truyền skel vào và lấy list đặc trưng cực gọn gàng
                feats, _, _ = feature_extract.extract_features(mask, en, skel)

                X_features.append(feats)
                y_labels.append(int(label))

                del img, en, mask, skel
            except Exception as e:
                print(f"❌ Lỗi tại file {filename}: {e}")

    if not X_features:
        print("❌ Không có dữ liệu đặc trưng nào được trích xuất!")
        return

    print(f"--- Đã trích xuất xong {len(X_features)} mẫu. Bắt đầu Tinh Chỉnh (Fine-tuning) ---")

    # 1. Chia tập dữ liệu: 80% để train, 20% để test (đánh giá khách quan)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # 2. Thiết lập lưới siêu tham số (Hyperparameter Grid)
    param_grid = {
        'n_estimators': [50, 100, 200],  # Số lượng cây quyết định
        'max_depth': [None, 10, 15, 20],  # Độ sâu tối đa của cây (tránh học vẹt/overfitting)
        'min_samples_split': [2, 5, 10],  # Số mẫu tối thiểu để chia nhánh
        'min_samples_leaf': [1, 2, 4]  # Số mẫu tối thiểu ở lá cuối cùng
    }

    print("Đang tìm kiếm bộ não AI tối ưu nhất (GridSearchCV)... Vui lòng đợi khoảng vài mươi giây.")

    # Sử dụng GridSearchCV với Cross-Validation = 5
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Lấy mô hình xuất sắc nhất
    best_model = grid_search.best_estimator_
    print(f"\n✅ Đã tìm thấy cấu hình tham số tốt nhất:\n{grid_search.best_params_}")

    # 3. Làm "Bài thi" đánh giá trên tập Test
    y_pred = best_model.predict(X_test)
    print("\n" + "=" * 50)
    print(" BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH (REPORT CARD)")
    print("=" * 50)
    print(f"Độ chính xác tổng thể (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Bình thường (0)', 'Nguy cơ Đột quỵ (1)']))

    # 4. Huấn luyện lại model xuất sắc nhất với 100% dữ liệu để mang ra thực chiến
    print("\nĐang đóng gói mô hình với toàn bộ dữ liệu...")
    best_model.fit(X_features, y_labels)

    # Lưu model
    joblib.dump(best_model, MODEL_PATH)
    print(f"🎉 HOÀN TẤT! Đã lưu model tinh chỉnh tại: {MODEL_PATH}")


if __name__ == '__main__':
    # Tên thư mục dataset của bạn (hãy đổi nếu bạn dùng tên khác)
    train_optimized("dataset")