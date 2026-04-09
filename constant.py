import os

# Đường dẫn hệ thống
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stroke_risk_model.pkl")

# Tạo thư mục model nếu chưa có
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Tham số hình ảnh
IMG_SIZE = (400, 400)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Ngưỡng y khoa
AV_RATIO_THRESHOLD = 0.65
TORTUOSITY_THRESHOLD = 1.40  # Nâng lên 1.4 như đã thống nhất
MIN_VESSEL_AREA = 100        # Ngưỡng lọc nhiễu chung
DENSITY_THRESHOLD = 0.05
MIN_BRANCH_COUNT = 15

# Ngưỡng lâm sàng
THRESHOLD_TORT_LOCAL = 1.40
THRESHOLD_NARROW_LOCAL = 0.50