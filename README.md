# VESSEL-EYE: Phân tích mạch máu võng mạc

Desktop app phân tích ảnh đáy mắt fundus theo hướng image-processing-first. ML chỉ là lớp phân loại nguy cơ ở cuối pipeline, không thay thế cho các bản đồ hình thái học.

## Tổng quan luồng xử lý

Pipeline hiện tại có 2 nhánh chính sau khi segment mạch:

1. `final vessel mask`: mask sau cleanup và morphology, dùng cho segmentation hiển thị, skeleton chính, đo calibre, A/V và đa số feature toàn cục.
2. `raw pre-close vessel mask`: mask trước bước `MORPH_CLOSE`, dùng riêng để phát hiện discontinuity và endpoint gap tiềm ẩn mà phép close có thể làm mất.

Điểm quan trọng: nếu chỉ nhìn skeleton cuối cùng thì một số đoạn gap nhỏ có thể biến mất. Vì vậy discontinuity không còn dựa duy nhất vào final skeleton nữa.

## Chi tiết pipeline classical

### 1. Chuẩn hóa đầu vào

- Ảnh được đưa về kích thước `768 x 768` trong [input_data.py](input_data.py).

### 2. FOV mask

- [riched_image.py](riched_image.py) tạo FOV robust bằng Gaussian blur lớn trên grayscale.
- Kết hợp threshold thấp và Otsu để lấy vùng võng mạc.
- Giữ connected component lớn nhất.
- Dùng close/open với kernel ellipse lớn để làm mượt vùng FOV.
- Nếu đủ điều kiện thì fit ellipse và co nhẹ ellipse để cắt viền sáng ngoài fundus.
- Sau đó erode nhẹ thành `proc_mask` để tránh bắt artifact sát rìa.

### 3. Green-channel enhancement

- Chỉ dùng kênh xanh lá vì tương phản vessel/background tốt nhất.
- Ước lượng nền sáng bằng morphological opening kernel lớn + Gaussian blur lớn.
- Trừ nền để giảm illumination gradient trong cùng ảnh.
- Chuẩn hóa intensity trong mask theo percentile.
- Gamma normalization để đưa mean intensity về mức ổn định hơn.
- CLAHE với `clipLimit=2.5`, `tileGridSize=(8, 8)` để tăng local contrast.

### 4. Vessel enhancement

- Frangi đa tỉ lệ với `sigmas = 1..8` để bắt cả mạch lớn lẫn mạch nhỏ.
- Black-hat đa scale `(9, 15)` để hỗ trợ mạch mảnh hoặc đoạn Frangi yếu.
- Hai đáp ứng được trộn lại, normalize trong `proc_mask`, rồi Gaussian blur nhẹ.

### 5. Threshold và tách 2 nhánh mask

- Threshold chính là hysteresis threshold: seed mạnh + vùng yếu có kết nối.
- Nếu hysteresis ra quá ít pixel thì fallback sang adaptive threshold dựa trên Otsu/percentile.
- Sau threshold, pipeline tạo:

`raw_vessel_mask`
- Là mask sau threshold và CC filtering, nhưng trước `MORPH_CLOSE`.
- Dùng cho discontinuity vì tại đây gap thật vẫn còn được giữ lại nếu closing chưa nối mất.

`final vessel mask`
- Tạo bằng `MORPH_CLOSE (3x3 ellipse)` trên `raw_vessel_mask`.
- Sau đó CC filtering lại để bỏ noise và component không hợp lệ.
- Dùng cho segmentation B/W, diameter measurement, A/V and most global features.

### 6. Skeleton

- Skeleton cuối cùng được tạo từ `final vessel mask`.
- Sau đó prune nhẹ bằng connected component area threshold để bỏ cành quá ngắn.

## Discontinuity hiện hoạt động như thế nào

### Vấn đề gốc

- Nếu một gap nhỏ bị `MORPH_CLOSE` nối lại trước khi skeletonize, thì final skeleton có thể trông liền mạch.
- Khi đó một logic discontinuity chỉ nhìn final skeleton sẽ bỏ sót các đoạn như mạch mờ, sắp mất, hoặc vừa bị reconnect bởi morphology.

### Cách xử lý hiện tại

Trong [feature_extract.py](feature_extract.py):

1. Tạo endpoint candidates từ skeleton.
2. Dùng graph pairing để ghép các endpoint đối diện nhau theo khoảng cách, hướng tiếp tuyến và hành lang hỗ trợ quanh vessel.
3. Dùng thêm morphology fallback khi graph không tìm được bridge hợp lệ.
4. Nếu có `raw_vessel_mask`, discontinuity chạy thêm trên nhánh pre-close thay vì chỉ nhìn final mask.
5. Đồng thời so sánh `raw_vessel_mask` với `final vessel mask` để tìm những pixel bridge do morphology tạo ra. Đây là tín hiệu trực tiếp cho loại gap “bị close che mất”.

Nói ngắn gọn:

- `final mask` giúp ổn định segmentation tổng thể.
- `raw mask` giữ lại thông tin gap thật cho discontinuity.

### Hệ quả thực tế

- Một số gap nhỏ sẽ vẫn không bị bắt nếu threshold ban đầu đã không giữ được đoạn mạch đó.
- Nhưng ít nhất pipeline không còn bị mù hoàn toàn trước các gap bị nối mất bởi `close` như trước.

## Optic disc và Zone B

Trong [anatomy.py](anatomy.py):

- Heuristic OD không còn nhìn vùng sáng đơn thuần.
- Ứng viên OD được chấm bằng bright score, rim contrast, circularity, centrality và vessel convergence map.
- Nếu deep OD từ AutoMorph quá lệch hoặc mask quá bất thường, main app sẽ bỏ và quay về heuristic vessel-guided.
- Zone B được dựng từ tâm và bán kính OD, rồi dùng để lọc marker và trích xuất feature quanh gai thị.

## A/V, diameter, tortuosity, clinical map

Trong [draw.py](draw.py):

- Skeleton segments được dựng từ `final skeleton`.
- Segment được phân loại artery/vein bằng deep A/V mask nếu có, nếu không thì fallback sang SVM/classical brightness rule.
- Diameter heatmap tô màu theo đường kính đo thực trên cross-section vuông góc skeleton.
- Marker narrowing/dilation/tortuosity/gap được lọc theo FOV, Zone B và vùng loại trừ quanh optic disc để tránh map quá rối.
- Tất cả panel giữa đều có thể `Phong to` hoặc double-click để mở fullscreen.

## Feature vector dùng cho ML

Vector feature 10 chiều trong [feature_extract.py](feature_extract.py):

1. `AVR`
2. `CRAE`
3. `CRVE`
4. `Tortuosity`
5. `StdTortuosity`
6. `VesselDensity`
7. `FractalDim`
8. `Discontinuity`
9. `EndpointGapScore`
10. `WhiteningScore`

Lưu ý:

- Marker trên map là lớp diễn giải cục bộ.
- Quyết định nguy cơ đỏ/vàng/xanh ở UI là từ model ML chạy trên toàn bộ vector đặc trưng.
- Vì vậy có thể xảy ra trường hợp map khoanh ít nhưng xác suất vẫn cao nếu đặc trưng toàn cục bất thường.

## Quality triage

Trong [quality_assessment.py](quality_assessment.py):

- Chấm focus, contrast, illumination, FOV coverage, vessel visibility.
- Không loại cứng mọi ảnh ít thấy mạch vì đó cũng có thể là dấu hiệu bệnh lý.
- Kết quả được dùng như confidence / warning trong UI, không thêm trực tiếp vào vector 10D hiện tại.

## Training và evaluation

### Train

Trong [training_model.py](training_model.py):

- Trích xuất feature trên toàn bộ `dataset/0` và `dataset/1`.
- Train `RandomForestClassifier` với `class_weight='balanced'`.
- Có 5-fold CV, hold-out 80/20, và lưu model vào `models/stroke_risk_model.pkl`.
- Nếu train lại bây giờ, pipeline feature sẽ dùng cả `raw_vessel_mask` cho discontinuity thay vì luồng cũ chỉ nhìn final skeleton.

### Evaluate

- [evaluate_testset.py](evaluate_testset.py) dùng cùng pipeline feature để đánh giá trên `test/0` và `test/1`.
- Kết quả lưu vào `reports/test_metrics.json`.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Train classical:

```bash
python training_model.py --backend classical
```

Train AutoMorph-assisted:

```bash
python training_model.py --backend automorph --n_models 3 --device cpu
```

Evaluate test set:

```bash
python evaluate_testset.py
```

## Cấu trúc file chính

```text
main.py              UI PyQt6 và orchestration toàn pipeline
riched_image.py      Preprocessing classical, vessel enhancement, raw/final masks, skeleton
deep_backend.py      AutoMorph vessel segmentation backend
anatomy.py           Optic disc detection và Zone B
feature_extract.py   10D feature vector, discontinuity logic, endpoint analysis
draw.py              Visualization panels, clinical map, pathology findings
av_backend.py        AutoMorph artery/vein backend
av_classifier.py     A/V fallback classifier
quality_assessment.py Soft image quality triage
training_model.py    Huấn luyện Random Forest
evaluate_testset.py  Đánh giá test set
benchmark_comparison.py So sánh classical vs AutoMorph backend
constant.py          Hằng số hệ thống và threshold
input_data.py        Chuẩn hóa input image
```
