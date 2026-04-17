# VESSEL-EYE: Phân tích mạch máu võng mạc

Desktop app phân tích ảnh đáy mắt (fundus) — **tập trung xử lý ảnh**, ML chỉ hỗ trợ phân loại.
*Được Review và Tối Ưu Hóa bởi GitHub Copilot.*

## Pipeline xử lý ảnh chính (Bản cập nhật mới nhất cho Claude)

1. **Tiền xử lý (Preprocessing)**:
   - Chuẩn hóa đầu vào kích thước 768×768.
   - **Robust FOV Mask**: Tách vùng FOV bằng large Gaussian blur, kết hợp Threshold (low + Otsu), sau đó Morphological cleanup và Erode nhẹ mask để tránh artifact viền.
   - **Kênh Green (Green Channel)**: Sử dụng kênh xanh lá để có độ tương phản mạch/nền cao nhất.
   - **Chuẩn hóa ánh sáng (Background Illumination Correction - FOV)**: Suy diễn background sáng (mây sáng) qua Morphological OPEN (kernel lớn) và Gaussian Blur lớn để trừ dần cho Green channel, lấy lại gradient trung tâm ra rìa. In-mask intensity normalization (phân vị 1-99) chuẩn hóa độ tương phản và Gamma normalization để scale độ sáng mean intensity về ~128.
   - **Tăng cường tương phản cục bộ (CLAHE)**: `clipLimit=2.5`, `tileGridSize=(8, 8)` triệt tiêu chênh lệch tối ưu quanh các bó mạch.

2. **Chắt lọc & Tăng cường mạch máu (Vessel Enhancement)**:
   - **Hessian Multi-scale Frangi Filter**: Áp dụng dải sigma rộng `sigmas=1..8` để detect từ mao mạch biên đến tĩnh mạch trung tâm mà không bị bỏ sót.
   - **Multi-scale Black-Hat (9px, 15px)**: Phép toán đồng hành cùng Frangi để đảm bảo mạch mỏng không bị nhiễu do sigma lớn nuốt.

3. **Phân đoạn sắc nét (Segmentation)**:
   - **Adaptive Thresholding**: Tính toán ngưỡng trên phân phối FOV pixels, phối hợp linh hoạt chuyển đổi giữa Otsu và Percentiles để base density luôn ổn định ~5-18% diện tích mạch.
   - **Morphology Cleaning**: Dùng Morphological CLOSING (3x3 ellipse) với 2 vòng lặp để nối liền các mạch đứt mảnh liti. Không dùng Open để mạch mỏng 1-2px không vỡ.
   - **Connected Component (CC) Filter**: Loại bỏ vùng pixel cô lập rác (diện tích < 40px) và không dính viền.

4. **Skeleton & Pruning**:
   - Rút trích khung xương (skeletonize).
   - Prune cắt các cành dăm nhiễu (<12px).

5. **Phát hiện Optic Disc (OD)**:
   - Đánh giá HSV/LAB scoring, loại trừ mạch máu, thêm chấm ứng viên theo tương phản vòng đĩa và hỗ trợ biên để xác định đĩa thị. Định hình Zone B.

6. **Đánh giá chất lượng ảnh theo hướng lâm sàng (Soft Quality Triage)**:
   - Chấm focus, contrast, illumination, FOV coverage và độ thấy được mạch máu.
   - Không loại cứng mọi ảnh "ít thấy mạch" vì đây cũng có thể là tín hiệu bệnh lý.
   - Kết quả được hiển thị như độ tin cậy và cảnh báo đọc kết quả, không thay vector feature hiện tại của model.

7. **Đứt đoạn & Hình thái (Closing & Discontinuity)**:
   - Khám phá gap map bằng phép Morphological Closing trên skeleton gốc để hiển thị rõ các đoạn đứt gãy.

8. **Bản đồ lâm sàng & Phân tích (A/V)**:
   - Tính A/V ratio, mức xoắn vặn (Tortuosity), chỉ số hẹp động mạch.

## Giao diện (4 ảnh chính)

| Optic Disc Detection | Vessel Segmentation |
|---|---|
| **Skeleton** | **Closing & Discontinuity** |

## Quick start

```bash
pip install -r requirements.txt
python training_model.py   # Huấn luyện model trên dataset/0 và dataset/1 (tùy chọn)
python main.py             # Khởi chạy giao diện
```

Nếu muốn train lại với backend AutoMorph DL thay vì pipeline classical:

```bash
python training_model.py --backend automorph --n_models 3 --device cpu
```

## Cấu trúc

```
main.py              # PyQt6 desktop UI
riched_image.py      # Pipeline tăng cường + segmentation (CORE - Updated by GitHub Copilot)
anatomy.py           # Optic Disc detection + Zone B
draw.py              # Visualization + clinical map
feature_extract.py   # 10D feature vector
av_classifier.py     # A/V SVM classifier
training_model.py    # RandomForest training
evaluate_testset.py  # Test set evaluation
constant.py          # System constants
input_data.py        # Image I/O
```
