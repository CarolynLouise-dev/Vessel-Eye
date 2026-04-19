# Vessel-Eye Presentation Content

Tài liệu này chỉ giữ `nội dung slide`, không kèm script nói.

Nguyên tắc dùng xuyên suốt:

- `Kết luận cuối` không đi thẳng từ ảnh sang nhãn.
- Hệ thống trước hết `đo các thông số lâm sàng` từ ảnh võng mạc.
- Sau đó `kết hợp các dấu hiệu này bằng Machine Learning` để ra xác suất nguy cơ cuối cùng.
- `Demo hiện tại` dùng `10 đặc trưng` trong vector ML.
- Trên `UI góc phải`, hệ thống còn hiển thị thêm các chỉ số hỗ trợ đọc kết quả như `độ tin cậy ảnh` và `độ tin cậy đĩa thị`.
- Không dùng claim `Recall = 91%`.

---

## Slide 1. Các thông số lâm sàng đang được đánh giá và trích xuất cho học máy

### Tiêu đề slide

`Các thông số lâm sàng dùng để đánh giá nguy cơ trên UI và trong mô hình`

### Ý chính mở slide

- Chuẩn đánh giá của hệ thống là: `đo các thông số lâm sàng từ ảnh` rồi `kết hợp chúng bằng ML` để ra kết luận cuối cùng.
- Vì vậy cần chỉ rõ `từng thông số là gì`, `được trích xuất như thế nào`, và `cái nào đang đi vào mô hình`.

### Các thông số đang hiển thị ở UI góc phải

| STT | Đặc trưng / chỉ số | Công thức / kỹ thuật | Ngưỡng / ý nghĩa | Vai trò |
| --- | --- | --- | --- | --- |
| 1 | Độ tin cậy ảnh | quality score = `0.30 focus + 0.25 contrast + 0.20 illumination + 0.15 coverage + 0.10 vessel visibility` | `< 0.42`: low-confidence, `0.42 - 0.68`: usable-with-caution, `>= 0.68`: usable; nếu `FOV < 0.25` hoặc focus và contrast cùng rất thấp thì reject | hỗ trợ đọc kết quả |
| 2 | Độ tin cậy đĩa thị | optic disc detection + confidence | UI đang dùng mốc tham chiếu `> 40%`; nếu detector fallback thì confidence có thể chỉ khoảng `0.15` | hỗ trợ giải phẫu |
| 3 | A/V Ratio (AVR) | `CRAE / CRVE` | ngưỡng toàn cục trong project: `AVR < 0.65` gợi ý hẹp động mạch; ở marker cục bộ còn dùng `diameter_artery / mean_vein_diameter < 0.50` | vào ML |
| 4 | CRAE | trung bình đường kính động mạch trong Zone B | không có hard threshold toàn cục; UI ghi `tham chiếu nội bộ`, dùng để so sánh tương đối giữa các ca | vào ML |
| 5 | CRVE | trung bình đường kính tĩnh mạch trong Zone B | không có hard threshold toàn cục; UI ghi `tham chiếu nội bộ`, dùng cùng AVR để đọc mất cân đối A/V | vào ML |
| 6 | Độ cong trung bình | `arc length / chord length` theo từng segment | ngưỡng bệnh lý cục bộ trong code: `> 1.50`; comment code ghi bình thường khoảng `1.0 - 1.3`, xoắn bệnh lý khi `> 1.5` | vào ML |
| 7 | Biến thiên độ cong | `std(tortuosity)` | UI dùng mốc tham chiếu `Std < 0.20`; cao hơn gợi ý phân bố xoắn không ổn định | vào ML |
| 8 | Mật độ vi mạch | `vessel pixels / zone area` | ngưỡng toàn cục trong project: `< 0.05` tức `< 5%` gợi ý mạng mạch thưa; quality module cũng xem `vessel_density < 0.045` là rất thấp | vào ML |
| 9 | Fractal dimension | box-counting trên mạng mạch | UI hiển thị khoảng tham khảo `~1.3 - 1.7`; giảm thấp gợi ý mất độ phức tạp phân nhánh | vào ML |
| 10 | Discontinuity score | gap-weighted score trên skeleton graph | UI dùng mốc `Discontinuity < 0.15`; cao hơn gợi ý đứt đoạn mạch tăng | vào ML |
| 11 | Endpoint gap score | khoảng cách giữa các endpoint song song/ngược chiều | UI dùng mốc `Endpoint < 0.05`; cao hơn gợi ý gap bất thường hoặc tắc đoạn | vào ML |

### Đặc trưng nội bộ có trong vector ML nhưng không nhấn mạnh trên bảng UI

| Đặc trưng | Công thức / kỹ thuật | Ý nghĩa |
| --- | --- | --- |
| WhiteningScore | high-intensity region ngoài optic disc và ngoài vessel mask | project không đặt hard threshold toàn cục trong UI; score càng cao càng gợi ý phù trắng võng mạc, đặc biệt quan trọng trong CRAO cấp |

### Cách trích xuất để đưa vào học máy

- Tất cả đặc trưng đều được đo trên `vessel mask`, `skeleton graph`, `optic disc`, và `Zone B` để giữ chuẩn so sánh giữa các ảnh.
- `Zone B` trong code được tạo theo vòng nhẫn từ `1.0 x bán kính đĩa thị` đến `2.0 x bán kính đĩa thị`.
- `CRAE`, `CRVE`, `AVR`: đo caliber động mạch và tĩnh mạch trên vùng chuẩn hóa quanh đĩa thị.
- `Tortuosity`, `StdTortuosity`: tính theo từng nhánh mạch từ `arc length / chord length` trên skeleton.
- `VesselDensity`, `FractalDim`: đo độ dày và độ phức tạp của toàn mạng mạch trong vùng phân tích.
- `Discontinuity`, `EndpointGapScore`: phát hiện gap, endpoint bất thường và mức độ đứt đoạn trên đồ thị mạch.
- `WhiteningScore`: phát hiện vùng sáng bất thường ngoài optic disc và ngoài mạng mạch.

### Chốt nội dung slide

- `Thông số lâm sàng` là phần nền tảng.
- `Machine Learning` không thay thế các chỉ số này, mà học cách kết hợp chúng để ra kết luận cuối cùng.

---

## Slide 2. Vì sao phải kết hợp 6 dấu hiệu hình thái với nhau và với Machine Learning

### Tiêu đề slide

`Đánh giá theo 6 dấu hiệu hình thái RAO và lý do phải kết hợp bằng ML`

### Dấu hiệu hình thái RAO — chi tiết từng loại

#### DẤU HIỆU 1 — HẸP ĐỘNG MẠCH

- Sinh lý: tắc nghẽn làm áp lực giảm, lòng mạch co lại.
- IP: đo cross-section vuông góc với skeleton để lấy caliber.
- Đặc trưng liên quan: `CRAE`, `AVR`.
- Ghi chú: chỉ tính trong `Zone B` để so sánh được giữa các người bệnh.
- Trong code còn có marker hẹp cục bộ khi `diameter_artery / mean_vein_diameter < 0.50`.

#### DẤU HIỆU 2 — ĐỨT ĐOẠN MẠCH

- Sinh lý: đoạn sau chỗ tắc có thể không còn hiện rõ trên ảnh.
- IP: dùng `closing(skeleton) - skeleton`, endpoint analysis, vector tangent để tìm cặp gap song song ngược chiều.
- Đặc trưng liên quan: `Discontinuity`, `EndpointGapScore`.

#### DẤU HIỆU 3 — TĂNG ĐỘ XOẮN

- Sinh lý: tăng huyết áp và biến đổi thành mạch làm nhánh mạch mất đàn hồi.
- IP: tính `arc_length / chord_length` theo từng segment trên skan graph.
- Đặc trưng liên quan: `Tortuosity`, `StdTortuosity`.
- Mốc tham khảo: code dùng ngưỡng cục bộ `> 1.50`; comment trong project ghi bình thường khoảng `1.0 - 1.3`, xoắn bệnh lý khi `> 1.5`.

#### DẤU HIỆU 4 — PHÙ TRẮNG VÕNG MẠC

- Sinh lý: thiếu oxy làm tế bào phù nề, tạo vùng trắng võng mạc.
- IP: phát hiện high-intensity blob ngoài optic disc và ngoài vessel mask bằng connected components + area filter.
- Đặc trưng liên quan: `WhiteningScore`.
- Ý nghĩa: là dấu hiệu rất đặc trưng trong CRAO cấp tính.

#### DẤU HIỆU 5 — GIẢM MẬT ĐỘ MẠCH

- Sinh lý: các nhánh mạch nhỏ xẹp đi, làm mạng lưới thưa hơn.
- IP: `vessel_pixels / FOV_area` để tính density; box-counting để lấy fractal dimension.
- Đặc trưng liên quan: `VesselDensity`, `FractalDim`.
- Ngưỡng đang dùng trong project: `VesselDensity < 0.05`; quality module còn gắn cờ mật độ rất thấp khi `< 0.045`.

#### DẤU HIỆU 6 — MẤT CÂN ĐỐI A/V

- Sinh lý: động mạch teo nhỏ, tĩnh mạch có thể giãn bù trừ.
- IP: đo riêng `CRAE` và `CRVE`, không chỉ nhìn ratio mà còn nhìn absolute value.
- Đặc trưng liên quan: `AVR`, `CRAE`, `CRVE`.
- Ngưỡng toàn cục đang dùng cho ratio là `AVR < 0.65`.

### Tại sao phải kết hợp cả 6 dấu hiệu?

- `Đứt đoạn đơn lẻ` có thể do ảnh mờ hoặc phân đoạn chưa tốt, nên dễ false positive.
- `AVR thấp đơn lẻ` vẫn có thể là biến thiên bình thường giữa các người bệnh.
- `Độ xoắn tăng đơn lẻ` chưa đủ để kết luận vì còn phụ thuộc từng nhánh mạch riêng lẻ.
- `Whitening đơn lẻ` có thể bị nhiễu bởi vùng sáng hoặc phản xạ.
- Khi `nhiều dấu hiệu xuất hiện cùng lúc`, độ tin cậy chẩn đoán tăng rõ rệt hơn rất nhiều so với nhìn từng chỉ số rời rạc.

### Vì sao phải kết hợp với Machine Learning?

- Quan hệ giữa các dấu hiệu là `đa biến`, không nên chốt bằng một ngưỡng đơn lẻ.
- Random Forest nhận `vector đặc trưng` chứ không nhận ảnh thô.
- Mô hình học `trọng số kết hợp` giữa các dấu hiệu từ dữ liệu thật.
- Đầu ra là `xác suất nguy cơ`.
- Sau đó hệ thống dùng `decision threshold = 0.35` để phù hợp hơn với mục tiêu sàng lọc.

### Chốt nội dung slide

- Hệ thống không kết luận từ một dấu hiệu riêng lẻ.
- Hệ thống kết luận bằng cách `đo nhiều dấu hiệu lâm sàng`, rồi `kết hợp chúng bằng ML` để ra nguy cơ cuối cùng.

---

## Slide 3. Kết quả mô hình hiện tại

### Tiêu đề slide

`Kết quả mô hình Random Forest trên bộ dữ liệu hiện tại`

### Nội dung slide

- Dataset: `455 ảnh`
- Phân bố nhãn: `356 normal`, `99 high-risk`
- 5-fold cross-validation:
  - Accuracy = `0.8549 ± 0.0377`
  - F1 = `0.6548 ± 0.0689`
  - Recall = `0.6268 ± 0.0905`
  - AUC-ROC = `0.8488 ± 0.0500`
- Hold-out mặc định:
  - Accuracy = `0.8571`
  - F1 = `0.6486`
  - Sensitivity = `0.6000`
  - Specificity = `0.9296`
- Deploy threshold `0.35`:
  - Accuracy = `0.8352`
  - Recall = `0.7000`
  - Specificity = `0.8732`

### Thông điệp slide

- Mô hình đang được dùng theo hướng `ưu tiên sàng lọc`, nên ngưỡng `0.35` được chọn để tăng khả năng không bỏ sót ca nguy cơ.

---

## Slide 4. Đầu ra trực quan của hệ thống

### Tiêu đề slide

`Kết quả đầu ra không chỉ là nhãn, mà là bộ bằng chứng trực quan`

### Nội dung slide

- `Vessel mask`: tách cây mạch máu phục vụ đo lường.
- `A/V map`: phân loại động mạch và tĩnh mạch để đo CRAE, CRVE, AVR.
- `Optic disc + Zone B`: chuẩn hóa vùng phân tích.
- `Structural map`: đánh dấu xoắn và đứt đoạn mạch.
- `Diameter / calibre map`: cho thấy thay đổi caliber động mạch, tĩnh mạch.
- `Clinical table`: hiển thị các chỉ số lâm sàng ở UI góc phải.
- `Final risk probability`: xác suất nguy cơ cuối cùng do mô hình trả ra.

### Thông điệp slide

- Điểm mạnh của hệ thống là `giải thích được`, không chỉ cho ra một nhãn đen hộp.

---

## Slide 5. Hạn chế hiện tại

### Tiêu đề slide

`Hạn chế hiện tại của hệ thống`

### Nội dung slide

1. `Dữ liệu và nhãn`
  - Dataset còn lệch lớp.
  - Nhãn nguy cơ chưa phải ground truth lâm sàng quy mô lớn từ bác sĩ.

2. `Độ vững của pipeline ảnh`
  - Ảnh mờ, lóa, hoặc giao cắt mạch phức tạp vẫn có thể làm sai A/V, optic disc và discontinuity.
  - Một số ca chất lượng thấp có thể vừa là nhiễu kỹ thuật vừa là dấu hiệu bệnh lý.

3. `Đa mô thức chưa hoàn chỉnh`
  - Demo hiện tại chủ yếu dựa trên đặc trưng ảnh võng mạc.
  - Chưa tích hợp trực tiếp huyết áp, BMI, bệnh sử vào vector dự đoán cuối.

4. `Kiểm định ngoài tập dữ liệu`
  - Chưa có external validation đủ mạnh trên tập bệnh viện độc lập.

---

## Slide 6. Định hướng tương lai

### Tiêu đề slide

`Định hướng hoàn thiện hệ thống`

### Nội dung slide

1. Mở rộng từ `10 đặc trưng demo hiện tại` sang `khung 12 chỉ số` hoàn chỉnh hơn.
2. Tích hợp thêm `clinical metadata`: huyết áp, BMI, bệnh sử.
3. Chuẩn hóa dữ liệu theo hồ sơ bệnh nhân và theo dõi dọc theo thời gian.
4. Thu thập thêm nhãn bác sĩ và kiểm định ngoài bệnh viện.
5. Tối ưu backend để hỗ trợ telemedicine và phân tích nhanh hơn.

---

## Slide 7. Demo

### Tiêu đề slide

`Demo quy trình đánh giá từ ảnh đến kết luận cuối`

### Nội dung slide

- Bước 1: nạp ảnh fundus.
- Bước 2: xem vessel segmentation và A/V map.
- Bước 3: xem optic disc, Zone B, structural map và bảng chỉ số lâm sàng.
- Bước 4: đọc xác suất nguy cơ cuối cùng.
- Bước 5: nhấn mạnh logic đánh giá: `chỉ số lâm sàng + ML -> kết luận cuối`.