import sys, cv2, numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import riched_image_check
import warnings
import os

# Ẩn cảnh báo Deprecation từ sip/PyQt6
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Một số lỗi từ driver hoặc môi trường cũng có thể ẩn bằng dòng này
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.vulkan=false"

import sys, cv2, numpy as np
# --- STYLE SHEET CHUẨN ---
STYLESHEET = """
    QMainWindow { background-color: #0f172a; }
    QScrollArea { border: none; background-color: #1e293b; }
    QWidget#sideBar { background-color: #1e293b; border-right: 1px solid #334155; }
    QLabel { color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }
    QLabel#title { font-size: 18px; font-weight: bold; color: #38bdf8; margin-bottom: 15px; }
    QLabel#imgTitle { 
        font-size: 11px; font-weight: bold; color: #38bdf8; 
        background: #0f172a; padding: 6px; border-radius: 4px; 
        border: 1px solid #334155; 
    }
    QGroupBox { 
        color: #94a3b8; font-weight: bold; border: 1px solid #334155; 
        margin-top: 15px; padding-top: 15px; border-radius: 8px; 
    }
    QPushButton#loadBtn {
        background-color: #0ea5e9; color: white; border-radius: 6px;
        padding: 12px; font-weight: bold; font-size: 13px; border: none;
    }
    QPushButton#loadBtn:hover { background-color: #0284c7; }
    QSlider::handle:horizontal {
        background: #38bdf8; border: 1px solid #0ea5e9;
        width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #334155; height: 4px; background: #334155; border-radius: 2px;
    }
"""


class Worker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, img, params):
        super().__init__()
        self.img, self.params = img, params

    def run(self):
        try:
            steps = riched_image_check.get_processing_steps(self.img, self.params)
            self.result_ready.emit(steps)
        except Exception as e:
            print(f"Error: {e}")


class CheckApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Định nghĩa bộ giá trị mặc định chuẩn từ riched_image.py
        self.DEFAULTS = {
            'fov_blur': 35,
            'fov_thr': 12,
            'pre_blur': 3,
            'illu': 41,
            'clahe': 25,  # Tương ứng 2.5
            'fr_b1': 5,  # Tương ứng 0.5
            'fr_b2': 15,  # Tương ứng 15 (gamma)
            'fr_scale': 7,
            'hyst_low': 40,
            'min_area': 60,
            'skel_prune': 12
        }

        self.setWindowTitle("Vessel-Eye Pro | Parameter Tuner")
        self.img_raw = None
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_processing)
        self.labels = []
        self.initUI()
        self.setStyleSheet(STYLESHEET)

    def initUI(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- SIDEBAR ---
        side_bar = QWidget();
        side_bar.setObjectName("sideBar")
        side_bar.setFixedWidth(340)
        side_layout = QVBoxLayout(side_bar)

        # --- Cập nhật Sidebar ---
        side_layout.addWidget(QLabel("⚙️ ALGORITHM TUNER", objectName="title"))

        self.btn_load = QPushButton("📂 UPLOAD FUNDUS IMAGE", objectName="loadBtn")
        self.btn_load.clicked.connect(self.load_image)
        side_layout.addWidget(self.btn_load)

        # THÊM NÚT RESET
        self.btn_reset = QPushButton("🔄 RESET TO DEFAULTS")
        self.btn_reset.setStyleSheet("""
                    QPushButton {
                        background-color: #334155; color: #e2e8f0; border-radius: 6px;
                        padding: 8px; font-weight: bold; margin-top: 5px; border: 1px solid #475569;
                    }
                    QPushButton:hover { background-color: #475569; }
                """)
        self.btn_reset.clicked.connect(self.reset_settings)
        side_layout.addWidget(self.btn_reset)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #64748b; font-size: 11px;")
        side_layout.addWidget(self.status_label)

        scroll = QScrollArea();
        scroll_content = QWidget();
        s_layout = QVBoxLayout(scroll_content)

        # KHÔI PHỤC DEFAULT OPTIONS TỪ RICHED_IMAGE.PY
        g1 = QGroupBox("STEP 1: FOV & SMOOTHING")
        l1 = QVBoxLayout();
        g1.setLayout(l1)
        self.sl_fov_blur = self.create_ctrl(l1, "FOV Blur Mask", 5, 101, 35)  # Default 35
        self.sl_fov_thr = self.create_ctrl(l1, "FOV Threshold", 1, 50, 12)  # Default 12
        self.sl_pre_blur = self.create_ctrl(l1, "Smoothing (GaussianBlur)", 1, 15, 3)  # Default 3

        g2 = QGroupBox("STEP 2: ENHANCEMENT")
        l2 = QVBoxLayout();
        g2.setLayout(l2)
        self.sl_illu = self.create_ctrl(l2, "Illumination K", 11, 201, 41)  # Default 41
        self.sl_clahe = self.create_ctrl(l2, "CLAHE Clip (x10)", 5, 80, 25)  # Default 2.5 (25/10)

        g3 = QGroupBox("STEP 3: FRANGI (VESSELNESS)")
        l3 = QVBoxLayout();
        g3.setLayout(l3)
        self.sl_fr_b1 = self.create_ctrl(l3, "Beta 1 (Blobness x0.1)", 1, 20, 5)  # Default 0.5 (5/10)
        self.sl_fr_b2 = self.create_ctrl(l3, "Beta 2 (Noise x0.01)", 1, 100, 15)  # Default 15 (as gamma)
        self.sl_fr_scale = self.create_ctrl(l3, "Frangi Max Sigma", 1, 15, 7)  # Default 7 (sigmas 1->7)

        g4 = QGroupBox("STEP 4: SEGMENTATION")
        l4 = QVBoxLayout();
        g4.setLayout(l4)
        self.sl_hyst_low = self.create_ctrl(l4, "Hysteresis/Binary Low", 5, 150, 40)  # Default ~40
        self.sl_min_area = self.create_ctrl(l4, "Min Vessel Area (Clean)", 10, 800, 60)  # Default ~60
        self.sl_skel_prune = self.create_ctrl(l4, "Skeleton Pruning", 1, 100, 12)  # Default 12

        for g in [g1, g2, g3, g4]: s_layout.addWidget(g)
        s_layout.addStretch()
        scroll.setWidget(scroll_content);
        scroll.setWidgetResizable(True)
        side_layout.addWidget(scroll)
        layout.addWidget(side_bar)

        # --- DISPLAY GRID ---
        display_area = QWidget()
        grid = QGridLayout(display_area)
        grid.setSpacing(15)

        step_names = ["FOV Mask", "ROI Extract", "Green Blur", "Illu. Fixed", "CLAHE", "Frangi Prob", "Binary",
                      "Skeleton"]

        for i in range(8):
            container = QFrame()
            container.setMinimumSize(280, 240)
            container.setStyleSheet("background: #1e293b; border: 1px solid #334155; border-radius: 8px;")
            v = QVBoxLayout(container)

            title = QLabel(step_names[i], objectName="imgTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            img_lab = QLabel()
            img_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

            v.addWidget(title)
            v.addWidget(img_lab, stretch=1)
            grid.addWidget(container, i // 4, i % 4)
            self.labels.append(img_lab)

        layout.addWidget(display_area, stretch=1)
        self.setCentralWidget(main_widget)

    def create_ctrl(self, layout, name, min_v, max_v, def_v):
        h = QHBoxLayout()
        l_n = QLabel(name);
        l_n.setStyleSheet("font-size: 10px; color: #94a3b8;")
        l_v = QLabel(str(def_v));
        l_v.setStyleSheet("color: #38bdf8; font-weight: bold;")
        h.addWidget(l_n);
        h.addStretch();
        h.addWidget(l_v)
        layout.addLayout(h)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_v, max_v);
        s.setValue(def_v)
        s.valueChanged.connect(lambda v: (l_v.setText(str(v)), self.trigger_processing()))
        layout.addWidget(s)
        return s

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Fundus Image", "", "Images (*.png *.jpg *.tif)")
        if path:
            self.img_raw = cv2.imread(path)
            self.trigger_processing()

    def trigger_processing(self):
        self.timer.start(250)

    def start_processing(self):
        if self.img_raw is None: return
        p = {
            'fov_blur': self.sl_fov_blur.value(),
            'fov_low_thr': self.sl_fov_thr.value(),
            'pre_blur': self.sl_pre_blur.value(),
            'illu_k': self.sl_illu.value(),
            'clahe_clip': self.sl_clahe.value() / 10.0,
            'fr_b1': self.sl_fr_b1.value() * 0.1,
            'fr_b2': self.sl_fr_b2.value() * 1.0,  # gamma gốc là 15, không cần chia nhỏ quá
            'fr_scale': self.sl_fr_scale.value(),
            'hyst_low': self.sl_hyst_low.value(),
            'min_area': self.sl_min_area.value(),
            'skel_prune': self.sl_skel_prune.value()
        }
        self.worker = Worker(self.img_raw, p)
        self.worker.result_ready.connect(self.update_images)
        self.worker.start()

    def update_images(self, steps):
        for i, img in enumerate(steps):
            if i < len(self.labels):
                h, w = img.shape[:2]
                fmt = QImage.Format.Format_BGR888 if len(img.shape) == 3 else QImage.Format.Format_Grayscale8
                qimg = QImage(img.data, w, h, img.strides[0], fmt)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.labels[i].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.labels[i].setPixmap(pix)
        self.status_label.setText("Updated.")

    # Hàm thực hiện reset
    def reset_settings(self):
        self.sl_fov_blur.setValue(self.DEFAULTS['fov_blur'])
        self.sl_fov_thr.setValue(self.DEFAULTS['fov_thr'])
        self.sl_pre_blur.setValue(self.DEFAULTS['pre_blur'])
        self.sl_illu.setValue(self.DEFAULTS['illu'])
        self.sl_clahe.setValue(self.DEFAULTS['clahe'])
        self.sl_fr_b1.setValue(self.DEFAULTS['fr_b1'])
        self.sl_fr_b2.setValue(self.DEFAULTS['fr_b2'])
        self.sl_fr_scale.setValue(self.DEFAULTS['fr_scale'])
        self.sl_hyst_low.setValue(self.DEFAULTS['hyst_low'])
        self.sl_min_area.setValue(self.DEFAULTS['min_area'])
        self.sl_skel_prune.setValue(self.DEFAULTS['skel_prune'])

        self.status_label.setText("Parameters reset to defaults.")
        self.trigger_processing()

    # (Các hàm khác: create_ctrl, load_image, update_images... giữ nguyên)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CheckApp();
    win.showMaximized()
    sys.exit(app.exec())