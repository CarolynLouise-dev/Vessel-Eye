# main.py
import sys, cv2, joblib, os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from constant import MODEL_PATH
import riched_image, feature_extract, draw


class StrokeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RETINA-SCAN: CHẨN ĐOÁN ĐỘT QUỴ QUA ẢNH CHỤP VÕNG MẠC")
        self.setGeometry(30, 30, 1580, 950)
        self.setStyleSheet("QMainWindow { background-color: #0f172a; }")
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)

        # --- CỘT 1: NHẬP LIỆU ---
        side_panel = QVBoxLayout()
        title = QLabel("RETINA-SCAN v1.0")
        title.setStyleSheet("color: #38bdf8; font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        side_panel.addWidget(title)

        self.lbl_orig = QLabel("ẢNH GỐC")
        self.lbl_orig.setFixedSize(360, 360)
        self.lbl_orig.setStyleSheet(
            "border: 2px dashed #334155; border-radius: 12px; background: #1e293b; color: #94a3b8;")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_panel.addWidget(self.lbl_orig)

        btn_upload = QPushButton("📥 TẢI ẢNH & CHẨN ĐOÁN")
        btn_upload.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; border-radius: 8px; padding: 18px; font-weight: bold; }")
        btn_upload.clicked.connect(self.upload_and_process)
        side_panel.addWidget(btn_upload)
        side_panel.addStretch()
        main_layout.addLayout(side_panel, 1)

        # --- CỘT 2: 4 ẢNH PHÂN TÍCH ---
        center_panel = QVBoxLayout()
        grid_imgs = QGridLayout()
        self.lbl_en, self.cont_en = self.img_box("TĂNG CƯỜNG TƯƠNG PHẢN")
        self.lbl_mask, self.cont_mask = self.img_box("PHÂN ĐOẠN MẠCH MÁU")
        self.lbl_skel, self.cont_skel = self.img_box("KHUNG XƯƠNG MẠCH")
        self.lbl_final, self.cont_final = self.img_box("BẢN ĐỒ LÂM SÀNG (FOV)")

        grid_imgs.addWidget(self.cont_en, 0, 0);
        grid_imgs.addWidget(self.cont_mask, 0, 1)
        grid_imgs.addWidget(self.cont_skel, 1, 0);
        grid_imgs.addWidget(self.cont_final, 1, 1)
        center_panel.addLayout(grid_imgs)
        main_layout.addLayout(center_panel, 3)

        # --- CỘT 3: 6 CHỈ SỐ & KẾT LUẬN ---
        result_panel = QVBoxLayout()

        # KHÔI PHỤC BẢNG 6 CHỈ SỐ
        self.table = QTableWidget(6, 3)
        self.table.setHorizontalHeaderLabels(["Chỉ số", "Giá trị", "Tham chiếu"])
        self.table.setStyleSheet("QTableWidget { background: #1e293b; color: white; border-radius: 10px; }")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setFixedHeight(240)
        result_panel.addWidget(QLabel("<b style='color: white;'>THÔNG SỐ ĐỊNH LƯỢNG:</b>"))
        result_panel.addWidget(self.table)

        # KHÔI PHỤC KẾT LUẬN Y KHOA CHI TIẾT
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
            r = QHBoxLayout();
            dot = QLabel();
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background: {c}; border-radius: 5px;")
            r.addWidget(dot);
            r.addWidget(QLabel(f"<span style='color: #cbd5e1; font-size: 10px;'>{t}</span>"));
            leg_lay.addLayout(r)

        add_l("#00ff00", "Vòng Lục: Xoắn vặn (Tortuosity > 1.4)");
        add_l("#00ffff", "Vòng Cyan: Hẹp động mạch (AVR < 0.5)")
        legend_box.setLayout(leg_lay);
        result_panel.addWidget(legend_box)

        result_panel.addStretch()
        main_layout.addLayout(result_panel, 1)

    def img_box(self, title):
        container = QFrame();
        container.setStyleSheet("background: #1e293b; border-radius: 10px; border: 1px solid #334155;")
        layout = QVBoxLayout(container);
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;");
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl = QLabel();
        img_lbl.setFixedSize(380, 310);
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_t);
        layout.addWidget(img_lbl);
        return img_lbl, container

    def set_label_image(self, label, cv_img, is_gray=False):
        h, w = cv_img.shape[:2];
        fmt = QImage.Format.Format_Grayscale8 if is_gray else QImage.Format.Format_RGB888
        if not is_gray: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
        label.setPixmap(
            QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def upload_and_process(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.jpg *.png)")
        if not file_path:
            return

        try:
            img_raw = cv2.imread(file_path)
            img_disp = cv2.resize(img_raw, (800, 800))

            # ===== Pipeline xử lý =====
            en, mask, skel, img_no_bg, fov_mask = riched_image.get_enhanced_vessels(img_disp)

            # ===== Hiển thị ảnh gốc đã bỏ nền =====
            self.set_label_image(self.lbl_orig, img_no_bg)

            # ===== Hiển thị các bước =====
            self.set_label_image(self.lbl_en, en, True)
            self.set_label_image(self.lbl_mask, mask, True)
            self.set_label_image(self.lbl_skel, skel, True)

            # ===== Feature extraction =====
            feats, regions = feature_extract.extract_features(mask, en)
            av_ratio, tort, std_tort, density = feats

            # ===== Draw clinical map =====
            f_map = draw.draw_feature_map(
                img_disp,
                mask,
                en,
                regions,
                img_no_bg=img_no_bg,
                fov_mask=fov_mask
            )

            self.set_label_image(self.lbl_final, f_map)

            # ===== Đếm artery / vein =====
            v_pix = en[mask > 0]
            m_br = np.median(v_pix) if len(v_pix) > 0 else 128

            a_count = sum(
                1 for r in regions
                if r.area >= 50 and np.mean(en[r.coords[:, 0], r.coords[:, 1]]) > m_br
            )

            v_count = sum(
                1 for r in regions
                if r.area >= 50 and np.mean(en[r.coords[:, 0], r.coords[:, 1]]) <= m_br
            )

            # ===== Bảng chỉ số =====
            ranges = {"AV": 0.65, "Tort": 1.40, "Std": 0.20, "Density": 0.05, "Count": 15}

            metrics = [
                ("Tỷ lệ A/V Ratio", f"{av_ratio:.4f}", f"> {ranges['AV']}"),
                ("Độ cong TB (Tort)", f"{tort:.4f}", f"< {ranges['Tort']}"),
                ("Biến thiên Std", f"{std_tort:.4f}", f"< {ranges['Std']}"),
                ("Mật độ vi mạch", f"{density * 100:.2f}%", f"> {ranges['Density'] * 100}%"),
                ("Số đoạn Động mạch", str(a_count), f"{ranges['Count']}-40"),
                ("Số đoạn Tĩnh mạch", str(v_count), f"{ranges['Count']}-40")
            ]

            self.table.setRowCount(6)

            for i, (n, v, r) in enumerate(metrics):

                self.table.setItem(i, 0, QTableWidgetItem(n))

                it_v = QTableWidgetItem(v)

                if (
                        (i == 0 and av_ratio < ranges["AV"]) or
                        (i == 1 and tort > ranges["Tort"]) or
                        (i == 3 and density < ranges["Density"]) or
                        ((i == 4 or i == 5) and int(v) < ranges["Count"])
                ):
                    it_v.setForeground(QColor("#ef4444"))
                    it_v.setFont(QFont("Arial", 10, QFont.Weight.Bold))

                self.table.setItem(i, 1, it_v)
                self.table.setItem(i, 2, QTableWidgetItem(r))

            # ===== AI model =====
            if os.path.exists(MODEL_PATH):

                model = joblib.load(MODEL_PATH)

                prob = model.predict_proba([feats])[0][1]

                status = "NGUY CƠ CAO" if prob > 0.6 else "NGUY CƠ THẤP"
                color = "#ef4444" if prob > 0.6 else "#22c55e"

                msg = f"<h2 style='color:{color};'>{status} ({prob * 100:.1f}%)</h2>"
                msg += "<p><b>Phát hiện bệnh lý:</b></p>"

                pathos = []

                if av_ratio < ranges["AV"]:
                    pathos.append("• Hẹp tiểu động mạch lan tỏa (Retinal Narrowing).")

                if tort > ranges["Tort"]:
                    pathos.append(f"• Mạch máu xoắn vặn (Tortuosity: {tort:.2f}).")

                if density < ranges["Density"]:
                    pathos.append("• Suy giảm mật độ vi mạch (Vessel Dropout).")

                if a_count < ranges["Count"]:
                    pathos.append("• Thiếu hụt số lượng nhánh động mạch.")

                if not pathos:
                    pathos.append("• Các chỉ số hình thái mạch máu trong giới hạn bình thường.")

                self.lbl_diagnosis.setText(msg + "<br>".join(pathos))

                self.lbl_diagnosis.setStyleSheet(
                    f"background:#1e293b;color:white;border-radius:10px;padding:15px;border-left:10px solid {color};"
                )

            else:
                self.lbl_diagnosis.setText(
                    "<b style='color:#fbd38d;'>Hệ thống chưa được huấn luyện.</b><br>"
                    "Vui lòng chạy training_model.py trước."
                )

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv);
    window = StrokeApp();
    window.show();
    sys.exit(app.exec())