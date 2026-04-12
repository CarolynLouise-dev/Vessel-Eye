import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from constant import MODEL_PATH, MODEL_DIR, IMG_SIZE
import riched_image, feature_extract
from riched_image import resize_with_pad


def train_optimized(dataset_path):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    X_features = []
    y_labels = []

    print("--- BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG CHUẨN Y KHOA ---")

    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Thư mục '{dataset_path}' không tồn tại!")
        return

    for label in ['0', '1']:
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

            except Exception as e:
                print(f"❌ Lỗi tại file {filename}: {e}")

    if not X_features:
        print("❌ Không có dữ liệu đặc trưng nào được trích xuất thành công!")
        return

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