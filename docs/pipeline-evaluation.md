# Báo Cáo Đánh Giá Pipeline — RETINA-SCAN (Vessel-Eye)

> **Ngày:** 11 tháng 4, 2026
> **Người lập:** AI Code Review
> **Mục đích:** Đánh giá kỹ thuật làm cơ sở trình bày với team và lập kế hoạch nâng cấp

---

## 1. Tổng Quan Dự Án

**RETINA-SCAN** là hệ thống phân tích ảnh võng mạc (fundus image) nhằm trích xuất các đặc trưng hình thái mạch máu và dự đoán nguy cơ đột quỵ bằng Machine Learning.

### Pipeline Xử Lý

```
Ảnh Võng Mạc (Fundus Image)
          ↓
Tạo FOV Mask                    (riched_image.py)
          ↓
Trích Kênh Xanh Lá (Green)      (riched_image.py)
          ↓
Tăng Cường Tương Phản (CLAHE)   (riched_image.py)
          ↓
Top-hat Morphology               (riched_image.py)
          ↓
Ngưỡng Thích Nghi               (riched_image.py)
          ↓
Skeletonization                  (riched_image.py)
          ↓
Trích Xuất Đặc Trưng             (feature_extract.py)
          ↓
Phân Loại RandomForest           (training_model.py)
          ↓
Hiển Thị GUI                     (main.py / draw.py)
```

### Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|---|---|
| Giao diện | PyQt6 |
| Xử lý ảnh | OpenCV, scikit-image |
| Machine Learning | scikit-learn (RandomForest) |
| Tiện ích | NumPy, joblib |

---

## 2. Tóm Tắt Đánh Giá

| Hạng mục | Kết quả | Ghi chú |
|---|---|---|
| Ý tưởng tổng thể | ✅ Có cơ sở | Tiếp cận có nền tảng y học |
| Cấu trúc code | ✅ Tốt | Phân chia module rõ ràng |
| Tiền xử lý ảnh | ⚠️ Cần sửa | Vấn đề chiều ngưỡng threshold |
| Trích xuất đặc trưng | ❌ Sai phương pháp | Phân loại A/V và metric tortuosity |
| Validation mô hình ML | ❌ Thiếu hoàn toàn | Không có train/test split hay metric |
| Độ tin cậy lâm sàng | ❌ Thấp | Hệ quả của các vấn đề trên |

> **Kết luận chung:** Pipeline phù hợp làm **proof-of-concept / demo học thuật**. Cần cải thiện đáng kể ở phần trích xuất đặc trưng và validation trước khi có giá trị lâm sàng thực sự.

---

## 3. Điểm Mạnh

### 3.1 Kênh Xanh Lá (Green Channel) — Lựa Chọn Đúng Chuẩn

Sử dụng kênh xanh lá là **tiêu chuẩn được thiết lập** trong nghiên cứu phân đoạn mạch máu võng mạc (Staal et al. 2004, Soares et al. 2006). Hemoglobin hấp thụ ánh sáng xanh lá mạnh nhất, tạo ra độ tương phản tốt nhất giữa mạch máu và nền.

### 3.2 CLAHE — Cấu Hình Phù Hợp

`clipLimit=2.5`, `tileGridSize=(8,8)` là các tham số được trích dẫn rộng rãi và phù hợp với ảnh y tế. Quan trọng hơn, CLAHE được áp dụng **sau** khi tạo FOV mask — đây là thứ tự đúng, tránh khuếch đại nhiễu ở viền ảnh.

### 3.3 FOV Masking — Loại Bỏ Nền Đen

Tạo mặt nạ vùng nhìn (Field-of-View) để loại bỏ phần nền tối trước khi xử lý là bước quan trọng mà nhiều implementation đơn giản bỏ qua. Bước này ngăn chặn việc phát hiện "mạch máu giả" ở biên ảnh.

### 3.4 Top-hat Morphology — Làm Nổi Mạch Máu Mảnh

Biến đổi Top-hat là kỹ thuật được ghi nhận để làm nổi bật các cấu trúc sáng, mảnh như mạch máu võng mạc trên nền thay đổi chậm.

---

## 4. Các Vấn Đề Nghiêm Trọng

### 4.1 🔴 Adaptive Threshold — Nguy Cơ Đảo Ngược Mask

**Vị trí:** `riched_image.py`

```python
vessel_mask = cv2.adaptiveThreshold(
    denoise,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,   # ← pixel sáng hơn trung bình cục bộ → 255
    17,
    2
)
```

**Vấn đề:** Trong kênh xanh lá của ảnh fundus, **mạch máu tối hơn nền**, không phải sáng hơn. `THRESH_BINARY` đánh dấu pixel *sáng hơn* ngưỡng thích nghi cục bộ là foreground (255). Điều này có thể tạo ra **mask đảo ngược** — nền là foreground còn mạch máu là background.

**Minh chứng:** Hầu hết các baseline xử lý ảnh fundus được công bố (DRIVE, STARE, CHASE_DB1) đều dùng `THRESH_BINARY_INV` khi làm việc trực tiếp từ kênh xanh lá.

**Cách sửa:**
- **Phương án A:** Đổi thành `cv2.THRESH_BINARY_INV`
- **Phương án B:** Thay Top-hat bằng **Black-hat** morphology (đảo chiều tương phản mạch máu/nền), rồi giữ nguyên `THRESH_BINARY`

---

### 4.2 🔴 Phân Loại Động Mạch/Tĩnh Mạch — Không Tin Cậy Về Y Tế

**Vị trí:** `feature_extract.py`

```python
v_pix_mean = np.median(vessel_pixels)  # trung vị cường độ toàn cục

if avg_int > v_pix_mean:
    a_diams.append(...)   # → phân loại là "động mạch"
else:
    v_diams.append(...)   # → phân loại là "tĩnh mạch"
```

**Vấn đề:** Cách phân loại động mạch/tĩnh mạch chỉ dựa trên **cường độ so với trung vị toàn cục sau CLAHE** là không đủ tin cậy vì:

- Sau CLAHE, tương phản cục bộ bị chuẩn hóa, làm sai lệch cường độ toàn cục
- Sự khác biệt sáng/tối giữa động mạch và tĩnh mạch (hiệu ứng copper-wire reflex) rất vi tế và phụ thuộc vào thiết bị chụp
- Các nghiên cứu học thuật đạt chuẩn dùng classifier riêng kết hợp: gradient màu sắc, profile độ rộng mạch, kết nối đồ thị, vị trí tương đối với đĩa thị

**Tác động:** **AVR (Artery-Vein Ratio)** — chỉ số lâm sàng quan trọng nhất trong pipeline — được tính từ kết quả phân loại sai này, dẫn đến sai số hệ thống lớn.

**Tài liệu tham khảo:** Dashtbozorg et al. (2014), Hu et al. (2013) dùng classifier SVM/CNN với nhiều đặc trưng đầu vào chuyên biệt cho bài toán phân loại A/V.

---

### 4.3 🔴 Connected Components — Không Phù Hợp Với Cấu Trúc Mạng Mạch Máu

**Vị trí:** `feature_extract.py`

```python
label_img = label(binary_mask)
regions = regionprops(label_img)
# Mỗi region được giả định là một đoạn mạch độc lập
```

**Vấn đề:** Mạng mạch máu võng mạc là một **đồ thị liên thông cao** với nhiều điểm phân nhánh và giao cắt. Trong thực tế, phần lớn mạch máu tạo thành **1–3 connected component khổng lồ**, không phải các đoạn rời rạc. Hệ quả:

- Mỗi `region` là một khối blob lớn không đều, không tương ứng với một đoạn mạch
- `axis_minor_length` của một blob kéo dài qua nhiều phân nhánh không phải đường kính mạch máu hợp lệ
- Các histogram đường kính được xây dựng từ đây không có ý nghĩa thống kê

**Cách đúng:** Phân tích **đồ thị skeleton** bằng cách phát hiện điểm phân nhánh (branch points), rồi đo thuộc tính trên từng đoạn giữa hai nút (dùng thư viện `skan`).

---

### 4.4 🔴 Skeleton Tính Xong Nhưng Không Được Dùng Trong Feature Extraction

**Vị trí:** `riched_image.py` → `feature_extract.py`

```python
# riched_image.py — skeleton CÓ được tính:
skeleton = skeletonize(vessel_mask > 0).astype(np.uint8) * 255
return en_green, vessel_mask, skeleton, img_no_bg, fov_mask

# feature_extract.py — skeleton KHÔNG được nhận:
def extract_features(binary_mask, en_green):
    # skeleton không bao giờ được sử dụng ở đây
```

**Vấn đề:** Skeleton là cấu trúc quan trọng nhất để tính tortuosity và độ dài mạch máu chính xác, nhưng không được truyền vào hay sử dụng trong `feature_extract.py`. Bước skeletonization tốn kém về tính toán chỉ được dùng để hiển thị trên GUI — lãng phí hoàn toàn về mặt feature engineering.

---

### 4.5 🟡 Metric Tortuosity — Chỉ Số Gián Tiếp, Sai Định Nghĩa Y Học

**Vị trí:** `feature_extract.py`

```python
t_score = (reg.perimeter ** 2) / (4 * np.pi * reg.area)
```

**Vấn đề:** Công thức này là **Isoperimetric Quotient** (còn gọi là Compactness/Circularity) — đo mức độ "tròn" hay "gọn" của một vùng, **không phải** độ cong đường đi của mạch máu. Một mạch thẳng dày và một mạch cong mảnh có thể cho ra cùng giá trị.

**Metric tortuosity đúng về mặt lâm sàng:**

$$T = \frac{L_{arc}}{L_{chord}}$$

Trong đó $L_{arc}$ là độ dài đường đi theo skeleton và $L_{chord}$ là khoảng cách thẳng giữa hai đầu mút. Tính toán này yêu cầu phân tích đồ thị skeleton.

---

### 4.6 🟡 Mô Hình ML — Hoàn Toàn Thiếu Validation

**Vị trí:** `training_model.py`

```python
model.fit(X_features, y_labels)
joblib.dump(model, MODEL_PATH)
# ← Không chia train/test. Không cross-validation. Không in bất kỳ metric nào.
```

**Vấn đề:** Không có bất kỳ chỉ số đánh giá nào được đo lường. Với dataset y tế nhỏ, RandomForest có thể âm thầm overfit mà không phát hiện được.

**Yêu cầu tối thiểu cho ML y tế:**

| Thành phần còn thiếu | Đề xuất |
|---|---|
| Chia tập dữ liệu | 80/20 stratified split |
| Cross-validation | 5-fold stratified CV |
| Metric đầu ra | Accuracy, AUC-ROC, F1, Sensitivity, Specificity |
| Ma trận nhầm lẫn | Bắt buộc cho bài toán phân loại y tế |

---

## 5. So Sánh Với Nghiên Cứu Hiện Tại

| Thành phần | Dự án này | State-of-the-Art (2024) |
|---|---|---|
| Phân đoạn mạch máu | Adaptive Threshold | U-Net, TransUNet (DRIVE F1 > 0.82) |
| Phân loại A/V | So sánh trung vị cường độ | CNN + Phân tích đồ thị |
| Tortuosity | Isoperimetric Quotient | Tỷ lệ arc/chord trên skeleton |
| Đo đường kính | Bounding box của region | Mặt cắt ngang perpendicular với skeleton |
| Số đặc trưng | 4 | 20–100+ |
| Mô hình | RandomForest | CNN end-to-end (Poplin et al. 2018, Google) |
| Validation | Không có | 5-fold CV + tập test ngoài |

**Tài liệu quan trọng:** Poplin et al. (2018) — *"Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning"* (Nature Biomedical Engineering) đạt kết quả mạnh trong dự đoán nguy cơ tim mạch trực tiếp từ ảnh fundus bằng CNN, không cần trích xuất đặc trưng trung gian.

---

## 6. Lộ Trình Nâng Cấp (Đề Xuất)

### Giai Đoạn 1 — Sửa Nhanh (Ít công sức, hiệu quả cao)

| # | Thay đổi | File | Công sức |
|---|---|---|---|
| 1.1 | Sửa chiều threshold (`THRESH_BINARY_INV`) | `riched_image.py` | 1 dòng |
| 1.2 | Thêm train/test split + in metric sau training | `training_model.py` | ~20 dòng |
| 1.3 | Truyền skeleton vào `extract_features()` | `feature_extract.py`, `riched_image.py` | Refactor nhỏ |

### Giai Đoạn 2 — Cải Tổ Feature Extraction (Công sức trung bình)

| # | Thay đổi | Hướng tiếp cận |
|---|---|---|
| 2.1 | Thay connected component bằng phân tích đồ thị skeleton | Dùng thư viện `skan` |
| 2.2 | Sửa tortuosity theo tỷ lệ arc/chord | Dựa trên từng đoạn skeleton |
| 2.3 | Sửa đo đường kính mạch theo mặt cắt ngang | Perpendicular với đường tâm skeleton |

### Giai Đoạn 3 — Phân Loại Động Mạch/Tĩnh Mạch (Công sức cao, cần dữ liệu nhãn)

| # | Thay đổi | Hướng tiếp cận |
|---|---|---|
| 3.1 | Xây dựng classifier A/V riêng | SVM hoặc CNN nhẹ trên patch mạch máu |
| 3.2 | Áp dụng hiệu chỉnh dựa trên đồ thị | Làm mượt nhãn A/V theo kết nối mạch |
| 3.3 | Tính lại AVR từ nhãn đã hiệu chỉnh | Validate lại với chỉ số lâm sàng |

### Giai Đoạn 4 — Nâng Cấp Mô Hình (Tùy chọn, dài hạn)

| # | Thay đổi | Hướng tiếp cận |
|---|---|---|
| 4.1 | Thay RandomForest bằng deep learning | Fine-tune EfficientNet hoặc ResNet |
| 4.2 | Xem xét hướng tiếp cận end-to-end | Loại bỏ trích xuất đặc trưng trung gian |

---

## 7. Đánh Giá Rủi Ro

| Rủi ro | Mức độ | Biện pháp giảm thiểu |
|---|---|---|
| Mask mạch máu bị đảo ngược → đặc trưng sai hoàn toàn | Cao | Sửa threshold + kiểm tra trực quan |
| AVR không chính xác do phân loại A/V sai | Cao | Thực hiện Giai đoạn 3 |
| Overfitting do thiếu validation | Cao | Thực hiện Giai đoạn 1.2 |
| Tortuosity không phản ánh định nghĩa lâm sàng | Trung bình | Thực hiện Giai đoạn 2.2 |
| Dataset nhỏ | Trung bình | Data augmentation, transfer learning |

---

## 8. Tài Liệu Tham Khảo

1. Staal, J. et al. (2004). *Ridge-based vessel segmentation in color images of the retina.* IEEE Transactions on Medical Imaging.
2. Soares, J. et al. (2006). *Retinal vessel segmentation using the 2-D Gabor wavelet and supervised classification.* IEEE Transactions on Medical Imaging.
3. Dashtbozorg, B. et al. (2014). *An automatic graph-based approach for artery/vein classification in retinal images.* IEEE Transactions on Medical Imaging.
4. Poplin, R. et al. (2018). *Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning.* Nature Biomedical Engineering.
5. Hu, Q. et al. (2013). *Automated separation of binary overlapping trees in avascular AOSLO images.* MICCAI.

---

*Tài liệu được lập cho mục đích review nội bộ nhóm. Không dùng cho mục đích lâm sàng.*
