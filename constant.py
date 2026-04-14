import os

# Đường dẫn hệ thống
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH     = os.path.join(MODEL_DIR, "stroke_risk_model.pkl")
AV_MODEL_PATH  = os.path.join(MODEL_DIR, "av_classifier.pkl")   # Phase 3

# Tạo thư mục model nếu chưa có
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Tham số hình ảnh
# IMG_SIZE được dùng xuyên suốt train/inference để giữ phân phối feature ổn định.
IMG_SIZE = (768, 768)
INPUT_FILL_VALUE = 0
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Quyết định lâm sàng: ưu tiên sensitivity ("bắt nhầm còn hơn bỏ sót")
RISK_DECISION_THRESHOLD = 0.35

# Ngưỡng y khoa (toàn cục)
AV_RATIO_THRESHOLD = 0.65
TORTUOSITY_THRESHOLD = 1.40      # phase 1 compactness-based (kept for reference)
MIN_VESSEL_AREA = 100
DENSITY_THRESHOLD = 0.05
MIN_BRANCH_COUNT = 15

# Ngưỡng lâm sàng cục bộ (per-segment, phase 2+)
THRESHOLD_TORT_LOCAL   = 1.50    # Phase 3: arc/chord ratio — bệnh lý thực sự
                                  # (bình thường: 1.0–1.3; xoắn bệnh lý: > 1.5)
THRESHOLD_NARROW_LOCAL = 0.50    # diameter_artery / mean_vein_diameter