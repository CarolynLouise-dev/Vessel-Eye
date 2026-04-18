import cv2
import numpy as np
import warnings
from skimage.filters import frangi, apply_hysteresis_threshold, sato, meijering
from skimage.morphology import skeletonize, remove_small_objects, disk, closing, reconstruction, binary_closing, \
    white_tophat
from skimage.measure import label, regionprops

try:
    from skimage.filters import apply_hysteresis_threshold
    _HYSTERESIS_AVAILABLE = True
except ImportError:
    _HYSTERESIS_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)


def resize_with_pad(img, target_size=(800, 800)):
    """
    Kỹ thuật Letterboxing: Resize ảnh giữ nguyên tỷ lệ (Aspect Ratio)
    bằng cách thêm viền đen, chống bóp méo hình học mạch máu.
    """
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left

    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def get_enhanced_vessels(img):
    # ── 1. TIỀN XỬ LÝ & TÌM MẶT NẠ VÕNG MẠC TỰ ĐỘNG (HOUGH CIRCLE) ──────
    # Trích xuất kênh Green (độ tương phản mạch máu tốt nhất)
    gray = img[:, :, 1]
    h, w = gray.shape

    # Tìm bán kính đĩa võng mạc tự động để loại bỏ viền đen ngoài
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (9, 9), 2),
        cv2.HOUGH_GRADIENT,
        dp=1, minDist=w // 2,
        param1=50, param2=30,
        minRadius=int(min(w, h) * 0.35),
        maxRadius=int(min(w, h) * 0.55)
    )

    retina_mask = np.zeros((h, w), dtype=np.uint8)
    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype(int)
        cv2.circle(retina_mask, (x, y), r - 10, 255, -1)
    else:
        # Fallback nếu không tìm thấy hình tròn
        center = (w // 2, h // 2)
        radius = int(min(w, h) * 0.46)
        cv2.circle(retina_mask, center, radius, 255, -1)

    # ── 2. TĂNG CƯỜNG TƯƠNG PHẢN (CLAHE) & XOÁ NHIỄU NỀN ────────────────
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Làm mịn nhẹ để giảm noise muối tiêu trước khi tách mạch
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # White Top-Hat: Trích xuất mạch sáng, ép toàn bộ nền mờ về 0 (Đen)
    # Đây là bước quan trọng nhất để xóa nhiễu "muối tiêu" ở ảnh CLAHE
    black_bg_vessels = white_tophat(cv2.bitwise_not(blurred), disk(10))
    ui_clahe_display = black_bg_vessels.copy()  # Ảnh [2] BỘ LỌC XANH nền đen

    # ── 3. DÒ MẠCH MÁU CHI TIẾT (MEIJERING FILTER) ─────────────────────
    # Meijering giúp giữ lại các mao mạch mảnh mà không bị đứt đoạn
    vessels_prob = meijering(black_bg_vessels, sigmas=range(1, 5), black_ridges=False)
    v_norm = cv2.normalize(vessels_prob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── 4. PHÂN NGƯỠNG & KẾT HỢP HÌNH THÁI HỌC (MORPHOLOGY) ─────────────
    # Ngưỡng hóa Otsu để tách mạch máu sáng khỏi nền đen
    _, thresh = cv2.threshold(v_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing: Nối các đoạn mạch bị đứt
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # ── 5. LOẠI BỎ ARTIFACT (CONNECTED COMPONENTS) ─────────────────────
    # Loại bỏ các vùng quá nhỏ (nhiễu rời rạc) diện tích dưới 100 pixel
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    vessel_mask = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 100:
            vessel_mask[labels == i] = 255

    # Áp dụng mặt nạ võng mạc tự động để xóa viền
    final_mask = cv2.bitwise_and(vessel_mask, retina_mask)

    # ── 6. LÀM SẮC NÉT MẠCH MÁU & TẠO KHUNG XƯƠNG ────────────────────────
    # Làm sắc nét mạch máu lần cuối (Sharpening)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    vessel_gray = cv2.bitwise_and(gray, final_mask)
    sharpened_vessels = cv2.filter2D(vessel_gray, -1, sharpen_kernel)
    # Chỉ giữ lại phần sắc nét nằm trong mask mạch máu
    vessel_final_display = np.where(final_mask > 0, sharpened_vessels, 0).astype(np.uint8)

    # Tạo khung xương (Skeleton)
    skel = skeletonize(final_mask > 0).astype(np.uint8) * 255

    img_no_bg = cv2.bitwise_and(img, img, mask=retina_mask)

    return ui_clahe_display, final_mask, skel, img_no_bg, retina_mask
