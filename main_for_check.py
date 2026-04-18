import sys, cv2, numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import riched_image_check
import warnings

# Ẩn cảnh báo SIP
warnings.filterwarnings("ignore", category=DeprecationWarning)

STYLESHEET = """
    QMainWindow { background-color: #0f172a; }
    QScrollArea { border: none; background-color: #1e293b; }
    QWidget#sideBar { background-color: #1e293b; border-right: 1px solid #334155; }
    QLabel { color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }
    QLabel#title { font-size: 18px; font-weight: bold; color: #38bdf8; margin-bottom: 15px; }
    QLabel#imgTitle { 
        font-size: 10px; font-weight: bold; color: #38bdf8; 
        background: #0f172a; padding: 4px; border-radius: 4px; 
        border: 1px solid #334155; 
    }
    QGroupBox { 
        color: #94a3b8; font-weight: bold; border: 1px solid #334155; 
        margin-top: 12px; padding-top: 15px; border-radius: 8px; 
    }
    QPushButton#loadBtn {
        background-color: #0ea5e9; color: white; border-radius: 6px;
        padding: 10px; font-weight: bold; font-size: 12px; border: none;
    }
    QPushButton#resetBtn {
        background-color: #334155; color: #cbd5e1; border-radius: 6px;
        padding: 8px; font-weight: bold; border: 1px solid #475569;
    }
    QSlider::handle:horizontal {
        background: #38bdf8; width: 14px; height: 14px; border-radius: 7px;
    }
"""


class Worker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, img, params):
        super().__init__();
        self.img = img;
        self.params = params

    def run(self):
        try:
            steps = riched_image_check.get_processing_steps(self.img, self.params)
            self.result_ready.emit(steps)
        except Exception as e:
            print(f"Processing Error: {e}")


class CheckApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.DEFAULTS = {
            'f_blur': 35, 'f_thr': 12, 'illu': 41,
            'clahe': 25, 'p_blur': 3, 'fr_b1': 5,
            'fr_b2': 15, 'fr_s': 7, 'h_low': 40,
            'm_area': 60, 'prune': 12
        }
        self.img_raw = None
        self.timer = QTimer();
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.start_processing)
        self.labels = []
        self.initUI()
        self.setStyleSheet(STYLESHEET)

    def initUI(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # SIDEBAR
        side_bar = QWidget(objectName="sideBar");
        side_bar.setFixedWidth(320)
        side_layout = QVBoxLayout(side_bar)
        side_layout.addWidget(QLabel("⚙️ OPTIMIZED TUNER", objectName="title"))

        self.btn_load = QPushButton("📂 LOAD FUNDUS IMAGE", objectName="loadBtn")
        self.btn_load.clicked.connect(self.load_image)
        side_layout.addWidget(self.btn_load)

        self.btn_reset = QPushButton("🔄 RESET TO DEFAULTS", objectName="resetBtn")
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        side_layout.addWidget(self.btn_reset)

        scroll = QScrollArea();
        scroll_content = QWidget();
        s_layout = QVBoxLayout(scroll_content)

        # UI Groups (Sắp xếp theo thứ tự đề xuất)
        g1 = QGroupBox("1. FOV & GREEN")
        l1 = QVBoxLayout();
        g1.setLayout(l1)
        self.sl_f_blur = self.create_ctrl(l1, "FOV Mask Blur", 5, 101, self.DEFAULTS['f_blur'])
        self.sl_f_thr = self.create_ctrl(l1, "FOV Threshold", 1, 50, self.DEFAULTS['f_thr'])

        g2 = QGroupBox("2. ENHANCE (FIXED ORDER)")
        l2 = QVBoxLayout();
        g2.setLayout(l2)
        self.sl_illu = self.create_ctrl(l2, "Illumination K", 11, 201, self.DEFAULTS['illu'])
        self.sl_clahe = self.create_ctrl(l2, "CLAHE Clip (x10)", 5, 80, self.DEFAULTS['clahe'])
        self.sl_p_blur = self.create_ctrl(l2, "Final Smoothing", 1, 15, self.DEFAULTS['p_blur'])

        g3 = QGroupBox("3. FRANGI VESSELNESS")
        l3 = QVBoxLayout();
        g3.setLayout(l3)
        self.sl_fr_b1 = self.create_ctrl(l3, "Beta 1 (Blobness)", 1, 20, self.DEFAULTS['fr_b1'])
        self.sl_fr_b2 = self.create_ctrl(l3, "Beta 2 (Noise)", 1, 100, self.DEFAULTS['fr_b2'])
        self.sl_fr_s = self.create_ctrl(l3, "Max Sigma", 1, 15, self.DEFAULTS['fr_s'])

        g4 = QGroupBox("4. CLEANING")
        l4 = QVBoxLayout();
        g4.setLayout(l4)
        self.sl_h_low = self.create_ctrl(l4, "Threshold Low", 5, 150, self.DEFAULTS['h_low'])
        self.sl_m_area = self.create_ctrl(l4, "Min Vessel Area", 10, 800, self.DEFAULTS['m_area'])
        self.sl_prune = self.create_ctrl(l4, "Skeleton Pruning", 1, 100, self.DEFAULTS['prune'])

        for g in [g1, g2, g3, g4]: s_layout.addWidget(g)
        s_layout.addStretch()
        scroll.setWidget(scroll_content);
        scroll.setWidgetResizable(True)
        side_layout.addWidget(scroll);
        layout.addWidget(side_bar)

        # GRID DISPLAY
        display_area = QWidget();
        grid = QGridLayout(display_area);
        grid.setSpacing(10)
        names = ["1. FOV Mask", "2. Green Channel", "3. Illu Fixed", "4. CLAHE", "5. Blur Final", "6. Frangi",
                 "7. Binary", "8. Skel Clean"]
        for i in range(8):
            container = QFrame()
            container.setMinimumSize(280, 240)
            container.setStyleSheet("background: #1e293b; border: 1px solid #334155; border-radius: 6px;")
            v = QVBoxLayout(container)
            t = QLabel(names[i], objectName="imgTitle");
            t.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l = QLabel();
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            v.addWidget(t);
            v.addWidget(l, 1)
            grid.addWidget(container, i // 4, i % 4);
            self.labels.append(l)

        layout.addWidget(display_area, 1)
        self.setCentralWidget(main_widget)

    def create_ctrl(self, layout, name, min_v, max_v, def_v):
        h = QHBoxLayout();
        n = QLabel(name);
        v = QLabel(str(def_v))
        n.setStyleSheet("font-size: 10px; color: #94a3b8;");
        v.setStyleSheet("color: #38bdf8; font-weight: bold;")
        h.addWidget(n);
        h.addStretch();
        h.addWidget(v);
        layout.addLayout(h)
        s = QSlider(Qt.Orientation.Horizontal);
        s.setRange(min_v, max_v);
        s.setValue(def_v)
        s.valueChanged.connect(lambda val: (v.setText(str(val)), self.timer.start(250)))
        layout.addWidget(s);
        return s

    def reset_to_defaults(self):
        d = self.DEFAULTS
        self.sl_f_blur.setValue(d['f_blur']);
        self.sl_f_thr.setValue(d['f_thr'])
        self.sl_illu.setValue(d['illu']);
        self.sl_clahe.setValue(d['clahe'])
        self.sl_p_blur.setValue(d['p_blur']);
        self.sl_fr_b1.setValue(d['fr_b1'])
        self.sl_fr_b2.setValue(d['fr_b2']);
        self.sl_fr_s.setValue(d['fr_s'])
        self.sl_h_low.setValue(d['h_low']);
        self.sl_m_area.setValue(d['m_area'])
        self.sl_prune.setValue(d['prune'])

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.tif)")
        if path: self.img_raw = cv2.imread(path); self.start_processing()

    def start_processing(self):
        if self.img_raw is None: return
        p = {
            'fov_blur': self.sl_f_blur.value(), 'fov_low_thr': self.sl_f_thr.value(),
            'illu_k': self.sl_illu.value(), 'clahe_clip': self.sl_clahe.value() / 10.0,
            'pre_blur': self.sl_p_blur.value(), 'fr_b1': self.sl_fr_b1.value() * 0.1,
            'fr_b2': self.sl_fr_b2.value(), 'fr_scale': self.sl_fr_s.value(),
            'hyst_low': self.sl_h_low.value(), 'min_area': self.sl_m_area.value(),
            'skel_prune': self.sl_prune.value()
        }
        self.worker = Worker(self.img_raw, p);
        self.worker.result_ready.connect(self.update_images);
        self.worker.start()

    def update_images(self, steps):
        for i, img in enumerate(steps):
            if i < len(self.labels):
                h, w = img.shape[:2];
                fmt = QImage.Format.Format_BGR888 if len(img.shape) == 3 else QImage.Format.Format_Grayscale8
                qimg = QImage(img.data, w, h, img.strides[0], fmt)
                pix = QPixmap.fromImage(qimg).scaled(self.labels[i].size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
                self.labels[i].setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv);
    win = CheckApp();
    win.showMaximized();
    sys.exit(app.exec())