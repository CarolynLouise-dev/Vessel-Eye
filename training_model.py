import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import os
import cv2
import numpy as np  # Thêm numpy nếu cần xử lý mảng
from sklearn.ensemble import RandomForestClassifier
from constant import MODEL_PATH, MODEL_DIR, IMG_SIZE
import riched_image, feature_extract


def train_optimized(dataset_path):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    X_features = []
    y_labels = []

    print("--- Bắt đầu trích xuất đặc trưng (Bản tối ưu hóa) ---")

    # Kiểm tra thư mục dataset gốc
    if not os.path.exists(dataset_path):
        print(f"Lỗi: Thư mục '{dataset_path}' không tồn tại!")
        return

    for label in ['0', '1']:
        folder = os.path.join(dataset_path, label)
        if not os.path.exists(folder):
            print(f"Cảnh báo: Thư mục nhãn {label} không tìm thấy tại {folder}")
            continue

        # Lấy danh sách ảnh
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Đang xử lý nhóm nhãn {label}: {len(files)} ảnh")

        for filename in files:
            full_path = os.path.join(folder, filename)

            img = cv2.imread(full_path)
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)

            try:
                # riched_image trả về 3 giá trị: en_green, vessel_mask, skeleton
                en, mask, *_ = riched_image.get_enhanced_vessels(img)

                # feature_extract trả về 2 cụm: [features], (lists_of_diams)
                # Chúng ta chỉ lấy cụm đầu tiên là các chỉ số để train
                result = feature_extract.extract_features(mask, en)
                feats = result[0]

                X_features.append(feats)
                y_labels.append(int(label))

                # Giải phóng bộ nhớ
                del img, en, mask
            except Exception as e:
                print(f"Lỗi tại file {filename}: {e}")

    if not X_features:
        print("❌ Không có dữ liệu đặc trưng nào được trích xuất thành công!")
        return

    print(f"--- Đang huấn luyện với {len(X_features)} mẫu đặc trưng ---")

    # Huấn luyện mô hình RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',  # Quan trọng: Giúp giảm báo động giả
        random_state=42
    )
    model.fit(X_features, y_labels)

    # Lưu model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ XONG: Model đã lưu thành công tại {MODEL_PATH}")


if __name__ == "__main__":
    # Đảm bảo thư mục 'dataset' có cấu trúc dataset/0 và dataset/1
    train_optimized("dataset")