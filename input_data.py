import cv2
import os
import numpy as np
from constant import IMG_SIZE

def load_and_preprocess_raw(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img

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