# Vessel-Eye v2.0 — Cải thiện phiên 2026-04-14

> Ghi lại toàn bộ thay đổi thực hiện trong phiên làm việc này.  
> Trạng thái: **Đang hoạt động / Cần cập nhật thêm**

---

## Mục tiêu phiên làm việc

Xuất phát từ nhận xét của người dùng trên các ảnh RAO (Retinal Artery Occlusion):

1. **Pipeline hiển thị chưa thể hiện ý nghĩa lâm sàng** — tên và nội dung 4 panel chưa rõ
2. **Panel phải UI bị trắng** — không đồng màu với nền dark của app
3. **Bản đồ lâm sàng cần zoom/pan** để xem rõ các vùng bất thường
4. **Cần phân đoạn mạch máu rõ ràng** hơn (B&W sắc nét)
5. **Cần đánh dấu rõ điểm hẹp, phồng, đứt đoạn** trên ảnh

---

## Thay đổi chi tiết theo file

### 1. `draw.py` — Viết lại hoàn toàn

#### Thêm mới
| Hàm | Mô tả |
|-----|-------|
| `_apply_colormap_jet(value_norm)` | Chuyển giá trị 0→1 thành màu BGR theo jet colormap (xanh→vàng→đỏ) |
| `_make_colorbar(height, width)` | Tạo thanh colorbar dọc cho legend |
| `draw_av_calibre_map(...)` | **Panel 2**: Vẽ A/V trên nền đen, màu nhiệt theo đường kính. Động mạch: warm palette; tĩnh mạch: cool palette |
| `draw_discontinuity_map(...)` | **Panel 3 (cũ)**: Skeleton xanh lá + gap đứt đoạn tô đỏ, khoanh vòng tròn |
| `draw_vessel_segmentation(...)` | **Panel 3 (mới)**: Phân đoạn mạch B&W cực rõ — CC filter, close, dilate nhẹ, CLAHE blend |
| `draw_diameter_heatmap(...)` | **Panel 4**: Heat-map đường kính dọc skeleton, marker HEP/PHONG |

#### Cải thiện
| Hàm | Thay đổi |
|-----|---------|
| `draw_optic_disc_vis(...)` | Thêm crosshair, Zone B rõ hơn, label tiếng Anh |
| `draw_feature_map(...)` | Bản đồ lâm sàng: nền làm mờ ×0.55, marker XOAN/HEP/DUT lớn hơn, danger zone overlay đỏ bán trong suốt, OD + Zone B |
| `draw_discontinuity_map(...)` | Siết lọc: min area `4 → 18`, giới hạn vòng tròn `60 → 20`, endpoint chỉ vẽ gần gap thực |

---

### 2. `riched_image.py` — Cải thiện FOV Masking

**Hàm `_build_fov_mask(img)`** — Thêm **Bước 3: Ellipse fitting**:

```python
# Fit ellipse lên contour lớn nhất → mask tròn chính xác hơn
# Shrink 4% để cắt viền sáng ngoài
# Kết hợp: bitwise_and(fov_threshold, fov_ellipse)
# Dùng nếu tỉ lệ overlap > 60%
```

**Mục đích**: Loại bỏ viền sáng tròn bên ngoài ảnh fundus, tránh bắt nhầm thành mạch máu.

---

### 3. `main.py` — Viết lại hoàn toàn

#### Class `ZoomableImageLabel` (mới)
Widget tùy chỉnh thay thế `QLabel` tĩnh cho bản đồ lâm sàng:

| Tính năng | Cách dùng |
|-----------|-----------|
| Zoom | `Ctrl + Scroll` |
| Pan | Click & kéo |
| Fullscreen | Nút **🔍 Phóng to** |
| Re-entry guard | Flag `_rendering` tránh vòng lặp resize vô hạn |

**Các lỗi đã fix trong quá trình phát triển:**

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `sizeHint()` trả về `(0,0)` → fullscreen chỉ chiếm nửa màn hình | `minimumWidth/Height` = 0 khi không set minimum | Override `sizeHint()` trả về `max(minimumWidth, 260)` |
| Double-click pan → mở fullscreen | `mouseDoubleClickEvent` trigger khi đang kéo | Xóa double-click, thay bằng nút riêng |
| Zoom tự giãn liên tục | Loop: `resizeEvent → setPixmap → sizeHint thay đổi → resize` | `QLabel.setPixmap()` trực tiếp (bypass override), flag `_rendering` |
| `UnboundLocalError: 'i'` | List `abnormal_flags` dùng `i` trước `for i in enumerate(...)` | Xóa list thừa |

#### Class `_FullscreenImageDialog` (mới)
Dialog phóng to toàn màn hình:
- `ZoomableImageLabel` bên trong với `SizePolicy.Expanding` + `stretch=1`
- ESC để đóng
- Hint text ở dưới

#### UI Fixes

| Vấn đề | Trước | Sau |
|--------|-------|-----|
| Panel phải bị trắng | `result_content` (QWidget) không có stylesheet | `setStyleSheet("QWidget { background: #0f172a; }")` |
| ScrollArea trắng | `QScrollArea` mặc định trắng | `setStyleSheet("background: #0f172a; border: none;")` |
| Tên panel không có ý nghĩa | `MORPHOLOGICAL CLOSING & DISCONTINUITY` | `🔬 PHÂN ĐOẠN MẠCH — Trích xuất mạch máu rõ nét (B&W)` |

#### Tên 4 Pipeline Panel (cập nhật)

| # | Tên mới | Nội dung |
|---|---------|---------|
| 1 | 🎯 OPTIC DISC — Vùng Zone B phân tích | Fundus + vòng OD cam + Zone B xanh lá |
| 2 | 🩸 A/V CALIBRE — Đường kính động/tĩnh mạch | Heat-map calibre nền đen, warm=artery, cool=vein |
| 3 | 🔬 PHÂN ĐOẠN MẠCH — Trích xuất mạch máu rõ nét | B&W sắc nét, CC-filtered, CLAHE-enhanced |
| 4 | 🌡 HEAT-MAP HẸP/PHỒNG — Biến đổi đường kính | Màu jet theo đường kính, marker HEP/PHONG |

---

## Ghi chú lâm sàng (để phát triển tiếp)

### ✨ Cập nhật bổ sung (Fixes sau review)
Sau khi review các lỗi "Mạch bị phồng to", "Kết luận ML sai lệch với RAO", và "Thuật toán Hysteresis Thresholding":

1. **Fix Hysteresis Thresholding (`riched_image.py`)**: 
   - Thay thế `_adaptive_vessel_threshold` bằng `_hysteresis_vessel_threshold`.
   - Dùng High threshold (percentile 88) làm hạt giống và Low threshold (percentile 72) để kết nối. Giúp các đoạn mạch thưa thớt ở bệnh RAO không bị đứt gãy.
2. **Fix Nhận diện cực đoan RAO (`main.py`)**:
   - Bổ sung override cứng bỏ qua model ML nếu phát hiện RAO/CRVO cực đoan:
      - AV Ratio cực thấp (< 0.3)
      - Tổ hợp (CRAE < 2.0 và CRVE > 3.0) -> Động mạch biến mất.
      - Fractal cực thấp (< 1.20) và hẹp lan tỏa.
   - Khi phát hiện, ép phần trăm rủi ro lên tối thiểu 88% (NGUY CƠ CAO) và hiển thị cảnh báo đỏ trên bảng metrics.
3. **Fix Vessel Thickness (`draw.py`)**:
   - Sửa Panel 3 "Phân đoạn mạch máu B&W". Mạch máu từng bị nở do double-processing (morphological close kép).
   - Xóa bỏ close & dilate dư thừa khỏi `draw_vessel_segmentation` và áp dụng erosion nhẹ (1px) để trả độ sắc nét nguyên bản cho mạch.

### Điểm cần cải thiện thêm

- [x] **RAO detection**: Đã thêm override rule bảo vệ.
- [ ] **False positive discontinuity**: Panel đứt đoạn vẫn tạo ra nhiều vòng tròn kể cả sau khi siết. Cân nhắc thêm bộ lọc dựa trên skeleton length thay vì gap area.
- [ ] **Vessel width calibration**: CRAE/CRVE hiện đo bằng pixel cross-section. Nên chuẩn hóa theo FOV size để có giá trị tuyệt đối (μm).
- [ ] **OD detection accuracy**: Một số ảnh OD bị xác định sai vị trí. Xem xét dùng bright blob + vascular convergence kết hợp.

### Ngưỡng lâm sàng hiện tại (constant.py)

```python
RISK_DECISION_THRESHOLD = 0.35  # Sensitivity cao: bắt nhầm hơn bỏ sót
AV_RATIO_THRESHOLD = 0.65       # < 0.65 → hẹp tiểu động mạch
THRESHOLD_TORT_LOCAL = 1.50     # arc/chord ratio: > 1.5 → xoắn bệnh lý
THRESHOLD_NARROW_LOCAL = 0.50   # diam_artery / mean_vein: < 0.5 → hẹp
```

---

## Files đã thay đổi

```
Vessel-Eye/
├── main.py          ← Viết lại hoàn toàn + Thêm RAO Rule Override + UI Fixes
├── draw.py          ← Viết lại hoàn toàn + Sửa Vessel Segment Thickness
└── riched_image.py  ← Cải thiện _build_fov_mask() + Thêm Hysteresis Thresholding
```

---

*Ghi bởi Antigravity — 14/04/2026*
