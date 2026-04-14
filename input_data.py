import cv2
import os
import numpy as np
from constant import IMG_SIZE, INPUT_FILL_VALUE


def standardize_fundus_image(img, target_size=IMG_SIZE, fill_value=INPUT_FILL_VALUE):
    """
    Chuẩn hóa ảnh đầu vào theo kích thước cố định nhưng giữ nguyên tỷ lệ khung hình.
    Ảnh được resize theo cạnh dài hơn rồi pad vào canvas vuông để tránh méo mạch máu.
    """
    if img is None:
        return None

    th, tw = target_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(tw / float(w), th / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), fill_value, dtype=np.uint8)

    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def load_and_preprocess_raw(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return standardize_fundus_image(img, IMG_SIZE)

def load_training_dataset(root_path):
    """
    Cấu trúc thư mục:
    /dataset/0 (Bình thường)
    /dataset/1 (Nguy cơ đột quỵ)
    """
    X, y = [], []
    for label in [0, 1]:
        folder = os.path.join(root_path, str(label))
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            img = load_and_preprocess_raw(os.path.join(folder, filename))
            if img is not None:
                X.append(img)
                y.append(label)
    return X, np.array(y)