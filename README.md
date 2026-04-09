Dưới đây là **README.md chuẩn cho project của bạn** (GUI + phân tích ảnh võng mạc + dự đoán nguy cơ đột quỵ). Bạn có thể **copy trực tiếp vào file `README.md` trong repo GitHub**.

---

# 🧠 RETINA-SCAN

### Stroke Risk Detection from Retinal Fundus Images

**RETINA-SCAN** là hệ thống phân tích **ảnh võng mạc (fundus image)** để trích xuất các đặc trưng hình thái mạch máu và dự đoán **nguy cơ đột quỵ** bằng Machine Learning.

Ứng dụng có giao diện GUI trực quan giúp:

* tải ảnh võng mạc
* phân đoạn mạch máu
* trích xuất chỉ số y sinh
* hiển thị bản đồ lâm sàng
* dự đoán nguy cơ đột quỵ

---

# 📷 Demo Pipeline

Pipeline xử lý ảnh:

```
Fundus Image
     ↓
Green Channel Extraction
     ↓
CLAHE Contrast Enhancement
     ↓
Top-hat Morphology
     ↓
Vessel Segmentation
     ↓
Skeletonization
     ↓
Feature Extraction
     ↓
Stroke Risk Prediction
```

---

# 🧪 Các chỉ số sinh học được phân tích

Hệ thống trích xuất **4 đặc trưng chính** từ mạch máu võng mạc:

| Feature                     | Ý nghĩa                                |
| --------------------------- | -------------------------------------- |
| **AVR (Artery-Vein Ratio)** | Tỷ lệ đường kính động mạch / tĩnh mạch |
| **Tortuosity**              | Độ cong mạch máu                       |
| **Std Tortuosity**          | Độ biến thiên độ cong                  |
| **Vessel Density**          | Mật độ mạch máu                        |

Các chỉ số này có liên quan đến:

* tăng huyết áp
* bệnh tim mạch
* nguy cơ đột quỵ

---

# 🧠 AI Model

Model sử dụng:

```
RandomForestClassifier
```

Cấu hình:

```
n_estimators = 100
max_depth = 12
class_weight = balanced
```

Input của model:

```
[AVR, Tortuosity, StdTortuosity, VesselDensity]
```

Output:

```
Stroke Risk Probability
```

---

# 🖥️ Giao diện ứng dụng

GUI được xây dựng bằng:

```
PyQt6
```

Ứng dụng hiển thị:

* ảnh gốc
* ảnh tăng cường tương phản
* phân đoạn mạch máu
* skeleton mạch
* bản đồ lâm sàng

Ngoài ra còn hiển thị:

* bảng chỉ số định lượng
* kết luận lâm sàng từ AI

---

# ⚙️ Cài đặt

## 1️⃣ Clone repository

```
git clone https://github.com/CarolynLouise-dev/Vessel-Eye.git
cd retina-scan
```

---

## 2️⃣ Cài đặt thư viện

```
pip install opencv-python
pip install numpy
pip install scikit-image
pip install scikit-learn
pip install pyqt6
pip install joblib
```

---

# 🏋️ Huấn luyện model

Chuẩn bị dataset:

```
dataset/
   ├── 0/
   │   ├── img1.jpg
   │   └── ...
   │
   └── 1/
       ├── img2.jpg
       └── ...
```

Chạy training:

```
python training_model.py
```

Model sẽ được lưu tại:

```
model/stroke_model.pkl
```

---

# 🚀 Chạy ứng dụng

```
python main.py
```

Sau đó:

1. tải ảnh võng mạc
2. hệ thống tự động phân tích
3. hiển thị kết quả và dự đoán

---

# 📊 Dataset đề xuất

Các dataset retinal phổ biến:

* DRIVE
* STARE
* CHASE_DB1

Các dataset này chứa **ảnh võng mạc đã được annotate mạch máu**.

---

# 🔬 Công nghệ sử dụng

```
Python
OpenCV
Scikit-image
Scikit-learn
PyQt6
NumPy
```

---

# 📈 Hướng phát triển

Trong tương lai hệ thống có thể cải tiến:

* Deep Learning vessel segmentation (U-Net)
* Frangi vessel filter
* Optic disc detection
* CRAE / CRVE measurement
* Stroke risk deep model

---

# ⚠️ Lưu ý

Hệ thống này **chỉ mang tính nghiên cứu và hỗ trợ**.

Không thay thế cho chẩn đoán y khoa chuyên nghiệp.

---



