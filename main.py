# main.py — Vessel-Eye: Retinal Vessel Image Processing
import sys, cv2, joblib, os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from constant import MODEL_PATH, IMG_SIZE, RISK_DECISION_THRESHOLD
import riched_image, feature_extract, draw
import av_classifier
import input_data



class StrokeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VESSEL-EYE: PHÂN TÍCH MẠCH MÁU VÕNG MẠC")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 720)
        self.setStyleSheet("QMainWindow { background-color: #0f172a; }")
        self.av_model = av_classifier.load_av_classifier()
        self._last_image_refs = {}
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        main_layout.addWidget(splitter)

        # --- CỘT 1: NHẬP LIỆU + BẢN ĐỒ LÂM SÀNG ---
        left_container = QFrame()
        left_container.setStyleSheet("background:#111827; border:1px solid #1f2937; border-radius:10px;")
        side_panel = QVBoxLayout(left_container)
        side_panel.setContentsMargins(10, 10, 10, 10)
        title = QLabel("VESSEL-EYE v2.0")
        title.setStyleSheet("color: #38bdf8; font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        side_panel.addWidget(title)

        self.lbl_orig = QLabel("ẢNH GỐC")
        self.lbl_orig.setMinimumSize(260, 260)
        self.lbl_orig.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_orig.setStyleSheet(
            "border: 2px dashed #334155; border-radius: 12px; background: #1e293b; color: #94a3b8;")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_panel.addWidget(self.lbl_orig)

        lbl_clinical_t = QLabel("BẢN ĐỒ LÂM SÀNG")
        lbl_clinical_t.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold; margin-top: 6px;")
        lbl_clinical_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_panel.addWidget(lbl_clinical_t)

        self.lbl_clinical = QLabel("CHƯA CÓ ẢNH")
        self.lbl_clinical.setMinimumSize(260, 200)
        self.lbl_clinical.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_clinical.setStyleSheet(
            "border: 1px solid #334155; border-radius: 12px; background: #020617; color: #94a3b8;"
        )
        self.lbl_clinical.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_panel.addWidget(self.lbl_clinical)

        btn_upload = QPushButton("📥 TẢI ẢNH & PHÂN TÍCH")
        btn_upload.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; border-radius: 8px; padding: 18px; font-weight: bold; }")
        btn_upload.clicked.connect(self.upload_and_process)
        side_panel.addWidget(btn_upload)
        side_panel.addStretch()
        splitter.addWidget(left_container)

        # --- CỘT 2: 4 ẢNH XỬ LÝ ẢNH (2x2) ---
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setFrameShape(QFrame.Shape.NoFrame)
        center_content = QWidget()
        center_scroll.setWidget(center_content)
        center_panel = QVBoxLayout(center_content)
        center_panel.setContentsMargins(0, 0, 0, 0)
        grid_imgs = QGridLayout()
        grid_imgs.setHorizontalSpacing(10)
        grid_imgs.setVerticalSpacing(10)

        self.lbl_od, self.cont_od = self.img_box("PHÁT HIỆN ĐĨA THỊ (OPTIC DISC)")
        self.lbl_mask, self.cont_mask = self.img_box("PHÂN ĐOẠN MẠCH MÁU (SẮC NÉT)")
        self.lbl_skel, self.cont_skel = self.img_box("KHUNG XƯƠNG MẠCH (SKELETON)")
        self.lbl_closing, self.cont_closing = self.img_box("MORPHOLOGICAL CLOSING & DISCONTINUITY")

        grid_imgs.addWidget(self.cont_od, 0, 0)
        grid_imgs.addWidget(self.cont_mask, 0, 1)
        grid_imgs.addWidget(self.cont_skel, 1, 0)
        grid_imgs.addWidget(self.cont_closing, 1, 1)
        grid_imgs.setColumnStretch(0, 1)
        grid_imgs.setColumnStretch(1, 1)
        center_panel.addLayout(grid_imgs)
        splitter.addWidget(center_scroll)

        # --- CỘT 3: CHỈ SỐ & KẾT LUẬN ---
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setFrameShape(QFrame.Shape.NoFrame)
        result_content = QWidget()
        result_scroll.setWidget(result_content)
        result_panel = QVBoxLayout(result_content)
        result_panel.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(9, 3)
        self.table.setHorizontalHeaderLabels(["Chỉ số", "Giá trị", "Tham chiếu"])
        self.table.setStyleSheet("QTableWidget { background: #1e293b; color: white; border-radius: 10px; }")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(260)
        result_panel.addWidget(QLabel("<b style='color: white;'>THÔNG SỐ ĐỊNH LƯỢNG:</b>"))
        result_panel.addWidget(self.table)

        self.lbl_diagnosis = QLabel("Sẵn sàng phân tích...")
        self.lbl_diagnosis.setWordWrap(True)
        self.lbl_diagnosis.setMinimumHeight(220)
        self.lbl_diagnosis.setStyleSheet(
            "background: #1e293b; color: white; border-radius: 10px; padding: 15px; border-left: 8px solid #334155;")
        result_panel.addWidget(QLabel("<b style='color: white;'>KẾT LUẬN LÂM SÀNG AI:</b>"))
        result_panel.addWidget(self.lbl_diagnosis)

        # Chú giải
        legend_box = QGroupBox("CHÚ GIẢI")
        legend_box.setStyleSheet(
            "QGroupBox { color: #38bdf8; font-weight: bold; border: 1px solid #334155; margin-top: 10px; border-radius: 10px; padding-top: 10px; }")
        leg_lay = QVBoxLayout()

        def add_l(c, t):
            r = QHBoxLayout(); dot = QLabel(); dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background: {c}; border-radius: 5px;")
            r.addWidget(dot); r.addWidget(QLabel(f"<span style='color: #cbd5e1; font-size: 10px;'>{t}</span>"))
            leg_lay.addLayout(r)

        add_l("#ff0000", "Đỏ: Động mạch (Artery)")
        add_l("#0000ff", "Xanh dương: Tĩnh mạch (Vein)")
        add_l("#00ff00", "Vòng Lục: Xoắn vặn (Tortuosity cao)")
        add_l("#00ffff", "Vòng Cyan: Hẹp động mạch (AVR thấp)")
        add_l("#f59e0b", "Cam: Đứt đoạn mạch (Discontinuity)")
        add_l("#00a5ff", "Cam nhạt: Đĩa thị (Optic Disc)")
        legend_box.setLayout(leg_lay)
        result_panel.addWidget(legend_box)

        result_panel.addStretch()
        splitter.addWidget(result_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([320, 920, 360])

    def img_box(self, title):
        container = QFrame()
        container.setStyleSheet("background: #1e293b; border-radius: 10px; border: 1px solid #334155;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl = QLabel()
        img_lbl.setMinimumSize(300, 250)
        img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        img_lbl.setStyleSheet("background:#020617; border:1px solid #334155; border-radius:6px;")
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_t)
        layout.addWidget(img_lbl)
        return img_lbl, container

    def set_label_image(self, label, cv_img, is_gray=False):
        self._last_image_refs[id(label)] = (label, cv_img.copy(), is_gray)
        h, w = cv_img.shape[:2]
        fmt = QImage.Format.Format_Grayscale8 if is_gray else QImage.Format.Format_RGB888
        if not is_gray:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
        label.setPixmap(
            QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for _, (label, img, is_gray) in self._last_image_refs.items():
            self.set_label_image(label, img, is_gray=is_gray)

    def upload_and_process(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        try:
            img_raw = cv2.imread(file_path)
            img_disp = input_data.standardize_fundus_image(img_raw, IMG_SIZE)

            # ===== Pipeline xử lý ảnh =====
            en, mask, skel, img_no_bg, fov_mask = riched_image.get_enhanced_vessels(img_disp)

            # ===== Hiển thị ảnh gốc =====
            self.set_label_image(self.lbl_orig, img_no_bg)

            # ===== Feature extraction =====
            feats, regions, feat_details = feature_extract.extract_features(
                mask, en, skeleton=skel,
                img_bgr=img_disp, av_model=self.av_model,
                fov_mask=fov_mask, return_details=True
            )
            av_ratio, crae, crve, tort, std_tort, density, fractal_dim, disc_score, endpoint_score, white_score = feats

            # ===== 1. Optic Disc Detection =====
            od_center = feat_details.get("od_center", (img_disp.shape[1] // 2, img_disp.shape[0] // 2))
            od_radius = feat_details.get("od_radius", 30)
            od_vis = draw.draw_optic_disc_vis(img_no_bg, od_center, od_radius)
            self.set_label_image(self.lbl_od, od_vis)

            # ===== 2. Vessel Segmentation (sharp) =====
            self.set_label_image(self.lbl_mask, mask, True)

            # ===== 3. Skeleton =====
            skel_vis = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
            self.set_label_image(self.lbl_skel, skel_vis, True)

            # ===== 4. Morphological Closing & Discontinuity =====
            closing_vis = draw.draw_closing_vis(skel, mask, fov_mask)
            self.set_label_image(self.lbl_closing, closing_vis)

            # ===== Clinical map =====
            f_map, debug_map = draw.draw_feature_map(
                img_disp, mask, en, regions,
                img_no_bg=img_no_bg, fov_mask=fov_mask,
                skeleton=skel, img_bgr=img_disp,
                av_model=self.av_model, return_debug=True,
                anatomy_details=feat_details,
            )
            self.set_label_image(self.lbl_clinical, f_map)

            # ===== Bảng chỉ số (9 hàng, bỏ whitening) =====
            ranges = {"AV": 0.65, "Tort": 1.50, "Density": 0.05, "Endpoint": 0.05}

            metrics = [
                ("Tỷ lệ A/V Ratio", f"{av_ratio:.4f}", f"> {ranges['AV']}"),
                ("CRAE (artery calib)", f"{crae:.2f}", "tham chiếu nội bộ"),
                ("CRVE (vein calib)", f"{crve:.2f}", "tham chiếu nội bộ"),
                ("Độ cong TB (Tort)", f"{tort:.4f}", f"< {ranges['Tort']}"),
                ("Biến thiên Std", f"{std_tort:.4f}", "< 0.20"),
                ("Mật độ vi mạch", f"{density * 100:.2f}%", f"> {ranges['Density'] * 100}%"),
                ("Fractal dimension", f"{fractal_dim:.4f}", "~1.3-1.7"),
                ("Discontinuity score", f"{disc_score:.4f}", "< 0.15"),
                ("Endpoint gap score", f"{endpoint_score:.4f}", f"< {ranges['Endpoint']:.2f}"),
            ]

            self.table.setRowCount(9)
            for i, (n, v, r) in enumerate(metrics):
                self.table.setItem(i, 0, QTableWidgetItem(n))
                it_v = QTableWidgetItem(v)
                if (
                    (i == 0 and av_ratio < ranges["AV"]) or
                    (i == 3 and tort > ranges["Tort"]) or
                    (i == 5 and density < ranges["Density"]) or
                    (i == 7 and disc_score > 0.15) or
                    (i == 8 and endpoint_score > ranges["Endpoint"])
                ):
                    it_v.setForeground(QColor("#ef4444"))
                    it_v.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                self.table.setItem(i, 1, it_v)
                self.table.setItem(i, 2, QTableWidgetItem(r))

            # ===== AI model (phần phụ trợ) =====
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                prob = model.predict_proba([feats])[0][1]

                if prob >= 0.70:
                    status = "NGUY CƠ CAO"
                    color = "#ef4444"
                elif prob >= RISK_DECISION_THRESHOLD:
                    status = "NGHI NGỜ NGUY CƠ"
                    color = "#f59e0b"
                else:
                    status = "NGUY CƠ THẤP"
                    color = "#22c55e"

                msg = f"<h2 style='color:{color};'>{status} ({prob * 100:.1f}%)</h2>"
                msg += "<p><b>Phát hiện bệnh lý:</b></p>"

                pathos = []
                if av_ratio < ranges["AV"]:
                    pathos.append("• Hẹp tiểu động mạch lan tỏa (Retinal Narrowing).")
                if tort > ranges["Tort"]:
                    pathos.append(f"• Mạch máu xoắn vặn (Tortuosity: {tort:.2f}).")
                if density < ranges["Density"]:
                    pathos.append("• Suy giảm mật độ vi mạch (Vessel Dropout).")
                if disc_score > 0.15:
                    pathos.append("• Tăng đứt đoạn mạng mạch (discontinuity cao).")
                if endpoint_score > ranges["Endpoint"]:
                    pathos.append("• Nhiều cặp endpoint gap nghi đứt đoạn dài.")
                if not pathos:
                    pathos.append("• Các chỉ số hình thái mạch máu trong giới hạn bình thường.")

                self.lbl_diagnosis.setText(msg + "<br>".join(pathos))
                self.lbl_diagnosis.setStyleSheet(
                    f"background:#1e293b;color:white;border-radius:10px;padding:15px;border-left:10px solid {color};"
                )
            else:
                self.lbl_diagnosis.setText(
                    "<b style='color:#fbd38d;'>Chưa có model ML.</b><br>"
                    "Chạy training_model.py để huấn luyện."
                )

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrokeApp()
    window.show()
    sys.exit(app.exec())