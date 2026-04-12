import os

# Đường dẫn hệ thống
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stroke_risk_model.pkl")

# Tạo thư mục model nếu chưa có
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Tham số hình ảnh
IMG_SIZE = (800, 800)
CLAHE_CLIP = 2.5
CLAHE_GRID = (8, 8)

# ==========================================
# HẰNG SỐ CHUẨN Y KHOA TẾ BÀO (Clinical Constants)
# ==========================================
AV_RATIO_THRESHOLD = 0.66        # Ngưỡng hẹp động mạch (Bình thường 0.67 - 0.80)
TORTUOSITY_THRESHOLD = 1.20      # Chỉ số VTI, >1.2 là bắt đầu có nguy cơ xoắn vặn
MIN_VESSEL_AREA = 50            # Lọc nhiễu pixel
DENSITY_THRESHOLD = 0.05         # Mật độ vi mạch tối thiểu
MIN_BRANCH_COUNT = 15            # Số ngã ba tối thiểu
ANGLE_MIN = 45                   # Góc Murray tối thiểu cho nhánh khỏe mạnh
ANGLE_MAX = 105                  # Góc Murray tối đa

# Ngưỡng lâm sàng cục bộ (Vẽ đồ họa)
THRESHOLD_TORT_LOCAL = 1.20
THRESHOLD_NARROW_LOCAL = 0.50