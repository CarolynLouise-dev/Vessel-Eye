import sys, cv2, joblib, os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from constant import MODEL_PATH
import riched_image, feature_extract, draw
from riched_image import resize_with_pad


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

        # Main Layout chia làm 3 cột
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # =======================================================
        # CỘT 1: ẢNH GỐC, NÚT TẢI VÀ CHÚ GIẢI
        # =======================================================
        col1_layout = QVBoxLayout()
        col1_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("RETINA-SCAN v2.0")
        title.setStyleSheet("color: #38bdf8; font-size: 26px; font-weight: bold; margin-bottom: 10px;")
        col1_layout.addWidget(title)

        # 1. Ảnh gốc
        lbl_orig_title = QLabel("[1] ẢNH GỐC")
        lbl_orig_title.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 13px;")
        lbl_orig_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orig = QLabel()
        self.lbl_orig.setFixedSize(320, 320)
        self.lbl_orig.setStyleSheet("border: 1px solid #475569; background-color: #1e293b; border-radius: 5px;")
        self.lbl_orig.setScaledContents(True)
        col1_layout.addWidget(lbl_orig_title)
        col1_layout.addWidget(self.lbl_orig)

        # 2. Nút tải ảnh
        self.btn_upload = QPushButton("TẢI ẢNH CHẨN ĐOÁN")
        self.btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upload.setMinimumHeight(50)
        self.btn_upload.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; color: white; font-weight: bold;
                border-radius: 8px; font-size: 15px; margin-top: 15px;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        self.btn_upload.clicked.connect(self.upload_and_process)
        col1_layout.addWidget(self.btn_upload)

        # 3. Chú giải lâm sàng
        legend_group = QGroupBox("📌 CHÚ GIẢI LÂM SÀNG")
        legend_group.setStyleSheet("""
            QGroupBox {
                color: #cbd5e1; font-weight: bold; font-size: 14px;
                border: 2px dashed #475569; border-radius: 8px; 
                margin-top: 20px; padding-top: 20px;
            }
        """)
        legend_layout = QVBoxLayout()
        legends = [
            ("<span style='color:#ef4444; font-size:18px;'>■</span> Động mạch"),
            ("<span style='color:#3b82f6; font-size:18px;'>■</span> Tĩnh mạch"),
            ("<span style='color:#22c55e; font-size:18px;'>⭕</span> Xoắn vặn"),
            ("<span style='color:#eab308; font-size:18px;'>✚</span> Ngã ba bất thường"),
            ("<span style='color:#f97316; font-size:18px;'>✖</span> Đứt/vỡ mạch")
        ]
        for text in legends:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #f8fafc; font-size: 14px;")
            legend_layout.addWidget(lbl)
        legend_group.setLayout(legend_layout)
        col1_layout.addWidget(legend_group)
        col1_layout.addStretch()

        # =======================================================
        # CỘT 2: KHUNG 4 ẢNH (LƯỚI 2x2)
        # =======================================================
        col2_layout = QGridLayout()
        col2_layout.setSpacing(15)

        def create_img_box(title_text):
            box = QWidget()
            vbox = QVBoxLayout(box)
            vbox.setContentsMargins(0, 0, 0, 0)
            lbl_title = QLabel(title_text)
            lbl_title.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 13px;")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            lbl_img = QLabel()
            lbl_img.setFixedSize(320, 320)
            lbl_img.setStyleSheet("border: 1px solid #475569; background-color: #1e293b; border-radius: 5px;")
            lbl_img.setScaledContents(True)

            vbox.addWidget(lbl_title)
            vbox.addWidget(lbl_img)
            return box, lbl_img

        box_en, self.lbl_en = create_img_box("[2] BỘ LỌC XANH")
        box_mask, self.lbl_mask = create_img_box("[3] MASK MẠCH MÁU")
        box_skel, self.lbl_skel = create_img_box("[4] KHUNG XƯƠNG")
        box_final, self.lbl_final = create_img_box("[5] BẢN ĐỒ LÂM SÀNG")

        col2_layout.addWidget(box_en, 0, 0)
        col2_layout.addWidget(box_mask, 0, 1)
        col2_layout.addWidget(box_skel, 1, 0)
        col2_layout.addWidget(box_final, 1, 1)

        col2_widget = QWidget()
        col2_widget.setLayout(col2_layout)

        # =======================================================
        # CỘT 3: BẢNG CHỈ SỐ Y KHOA & BẢNG CHẨN ĐOÁN
        # =======================================================
        col3_layout = QVBoxLayout()
        col3_layout.setSpacing(15)

        # 1. Bảng Chỉ số Y Khoa (Khôi phục lại `self.table` bị thiếu)
        lbl_table_title = QLabel("📊 BẢNG CHỈ SỐ Y KHOA")
        lbl_table_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 16px;")

        self.table = QTableWidget(0, 3)  # Sửa thành 3 cột
        self.table.setHorizontalHeaderLabels(["Chỉ số đặc trưng", "Giá trị đo lường", "Giá trị tiêu chuẩn"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1e293b; color: #f8fafc; gridline-color: #475569; border-radius: 5px; }
            QHeaderView::section { background-color: #334155; color: white; font-weight: bold; padding: 4px; }
        """)

        # 2. Bảng Chẩn đoán
        lbl_diag_title = QLabel("📝 KẾT LUẬN & CHẨN ĐOÁN")
        lbl_diag_title.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 16px; margin-top: 10px;")

        self.lbl_diagnosis = QLabel("Vui lòng tải ảnh lên để hệ thống AI phân tích...")
        self.lbl_diagnosis.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_diagnosis.setWordWrap(True)
        self.lbl_diagnosis.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_diagnosis.setStyleSheet("""
            background-color: #1e293b; color: #f8fafc;
            border-radius: 8px; padding: 15px; font-size: 14px;
            border: 1px solid #475569;
        """)

        col3_layout.addWidget(lbl_table_title)
        col3_layout.addWidget(self.table, 1)  # Cho bảng chiếm 1 phần không gian dọc
        col3_layout.addWidget(lbl_diag_title)
        col3_layout.addWidget(self.lbl_diagnosis, 2)  # Cho kết luận chiếm 2 phần không gian dọc

        col3_widget = QWidget()
        col3_widget.setLayout(col3_layout)

        # Ráp 3 cột vào Main Layout với tỷ lệ độ rộng (VD: 1 - 2 - 1.5)
        main_layout.addLayout(col1_layout, 1)
        main_layout.addWidget(col2_widget, 2)
        main_layout.addWidget(col3_widget, 2)

    def img_box(self, title):
        container = QFrame()
        container.setStyleSheet("background: #1e293b; border-radius: 10px; border: 1px solid #334155;")
        layout = QVBoxLayout(container)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl = QLabel()
        img_lbl.setFixedSize(380, 310)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_t)
        layout.addWidget(img_lbl)
        return img_lbl, container

    def set_label_image(self, label, cv_img, is_gray=False):
        h, w = cv_img.shape[:2]
        fmt = QImage.Format.Format_Grayscale8 if is_gray else QImage.Format.Format_RGB888
        if not is_gray: cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QImage(cv_img.data, w, h, cv_img.strides[0], fmt)
        label.setPixmap(
            QPixmap.fromImage(q_img).scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    def upload_and_process(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.jpg *.png)")
        if not file_path: return

        try:
            img_raw = cv2.imread(file_path)

            # Sử dụng kỹ thuật giữ nguyên khung hình đã tạo
            img_disp = resize_with_pad(img_raw, (800, 800))

            # Trích xuất các tầng ảnh
            en, mask, skel, img_no_bg, fov_mask = riched_image.get_enhanced_vessels(img_disp)

            # ================================================================
            # SỬA LẠI Ở ĐÂY: GẮN ĐÚNG BIẾN ẢNH VÀO 5 VỊ TRÍ TRÊN UI
            # ================================================================
            # [1] ẢNH GỐC: Dùng img_disp (Ảnh được chọn đưa vào, chưa xóa nền)
            self.set_label_image(self.lbl_orig, img_disp)

            # [2] ẢNH TƯƠNG PHẢN: Dùng en (Ảnh đã qua bộ lọc CLAHE xám)
            self.set_label_image(self.lbl_en, en, True)

            # [3] MASK MẠCH MÁU: Dùng mask (Ảnh nhị phân trắng đen)
            self.set_label_image(self.lbl_mask, mask, True)

            # [4] KHUNG XƯƠNG: Dùng skel (Ảnh skeletonize)
            self.set_label_image(self.lbl_skel, skel, True)
            # ================================================================

            # Truyền ảnh gốc màu vào để lấy không gian màu R/G
            ai_feats, regions, ab_junctions, non_ai_stats, break_pts = feature_extract.extract_features(
                img_disp, mask, en, skel, fov_mask
            )

            av_ratio, tort, std_tort, density, avg_angle, total_branches, dut_mach = ai_feats

            # [5] ẢNH LÂM SÀNG: Vẽ các tổn thương đè lên ảnh gốc (img_disp)
            f_map = draw.draw_feature_map(
                img_disp, mask, en, regions, img_no_bg=img_no_bg,
                fov_mask=fov_mask, abnormal_junctions=ab_junctions, breakages=break_pts
            )
            self.set_label_image(self.lbl_final, f_map)

            # Áp dụng chuẩn y khoa
            ranges = {"AV": 0.66, "Tort": 1.20, "Std": 0.20, "Density": 0.05, "AngleMin": 45, "AngleMax": 105,
                      "Count": 15, "Break": 10}

            metrics = [
                ("Tỷ lệ A/V Ratio", f"{av_ratio:.4f}", f"> {ranges['AV']}"),
                ("Độ xoắn vặn (Tort)", f"{tort:.4f}", f"< {ranges['Tort']}"),
                ("Phân tán xoắn vặn", f"{std_tort:.4f}", f"< {ranges['Std']}"),
                ("Mật độ vi mạch", f"{density * 100:.2f}%", f"> {ranges['Density'] * 100}%"),
                ("Góc phân nhánh", f"{avg_angle:.1f}°", f"{ranges['AngleMin']}° - {ranges['AngleMax']}°"),
                ("Tổng số ngã ba", f"{int(total_branches)}", f"> {ranges['Count']}"),
                ("Số điểm đứt gãy", f"{int(dut_mach)}", f"< {ranges['Break']}")
            ]

            self.table.setRowCount(7)
            for i, (n, v, r) in enumerate(metrics):
                self.table.setItem(i, 0, QTableWidgetItem(n))
                it_v = QTableWidgetItem(v)
                if ((i == 0 and av_ratio < ranges["AV"]) or
                        (i == 1 and tort > ranges["Tort"]) or
                        (i == 3 and density < ranges["Density"]) or
                        (i == 4 and (avg_angle < ranges["AngleMin"] or avg_angle > ranges["AngleMax"])) or
                        (i == 5 and int(total_branches) < ranges["Count"]) or
                        (i == 6 and int(dut_mach) > ranges["Break"])):
                    it_v.setForeground(QColor("#ef4444"))
                    it_v.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                self.table.setItem(i, 1, it_v)
                self.table.setItem(i, 2, QTableWidgetItem(r))

            # ====================================================
            # 3. CHẨN ĐOÁN (HIỂN THỊ 3 HƯỚNG RÕ RÀNG)
            # ====================================================
            diagnosis_html = ""

            # [HƯỚNG 3.1] NON-AI: KẾT QUẢ XỬ LÝ ẢNH TRỰC TIẾP
            diagnosis_html += "<h3 style='color:#38bdf8; margin-bottom:2px; font-size:14px;'>[1] CHẨN ĐOÁN HÌNH THÁI LÂM SÀNG (NON-AI)</h3>"
            diagnosis_html += f"<span style='font-size:13px;'>• <b>Mạng lưới:</b> {non_ai_stats['a_count']} đoạn động mạch, {non_ai_stats['v_count']} đoạn tĩnh mạch, {int(total_branches)} ngã ba.</span><br>"

            non_ai_warns = []
            # Đồng bộ với ngưỡng đứt gãy lâm sàng (ranges["Break"])
            if non_ai_stats['dut_mach'] > ranges["Break"]:
                non_ai_warns.append(
                    f"<span style='color:#f97316'>Đứt đoạn ({non_ai_stats['dut_mach']} điểm)</span>")
            if non_ai_stats['phong_mach'] > 0:
                non_ai_warns.append(
                    f"<span style='color:#eab308'>Phình mạch ({non_ai_stats['phong_mach']} vị trí)</span>")
            if non_ai_stats['vo_mach'] > 0:
                non_ai_warns.append(
                    f"<span style='color:#ef4444'>Nghi vỡ/xuất huyết ({non_ai_stats['vo_mach']} vị trí)</span>")

            if non_ai_warns:
                diagnosis_html += "<span style='font-size:13px;'>• <b>Cảnh báo:</b> " + " | ".join(
                    non_ai_warns) + "</span><br>"
                # BỔ SUNG KẾT LUẬN NON-AI KHI CÓ BỆNH LÝ
                diagnosis_html += "<span style='font-size:13px; color:#ef4444;'><b>=> KẾT LUẬN: PHÁT HIỆN TỔN THƯƠNG THỰC THỂ TẠI VI MẠCH (NGUY CƠ CAO)</b></span><br>"
            else:
                diagnosis_html += "<span style='font-size:13px;'>• <b>Tổn thương:</b> Không phát hiện vỡ mạch, phình mạch hay đứt gãy nghiêm trọng.</span><br>"
                # BỔ SUNG KẾT LUẬN NON-AI KHI KHỎE MẠNH
                diagnosis_html += "<span style='font-size:13px; color:#22c55e;'><b>=> KẾT LUẬN: CẤU TRÚC HÌNH THÁI MẠCH MÁU LÀNH LẶN</b></span><br>"

            diagnosis_html += "<h3 style='color:#38bdf8; margin-top:10px; margin-bottom:2px; font-size:14px;'>[2] ĐÁNH GIÁ THEO NGƯỠNG Y KHOA</h3>"
            med_warns = []
            if av_ratio < ranges["AV"]: med_warns.append("Hẹp tiểu động mạch (AVR thấp).")
            if tort > ranges["Tort"]: med_warns.append("Mạch máu xoắn vặn.")
            if density < ranges["Density"]: med_warns.append("Thiếu máu cục bộ (Mật độ thấp).")
            if avg_angle < ranges["AngleMin"] or avg_angle > ranges["AngleMax"]: med_warns.append(
                "Thay đổi áp lực mạch (Góc bất thường).")

            if med_warns:
                diagnosis_html += "<span style='font-size:13px;'>• " + "<br>• ".join(med_warns) + "</span><br>"
            else:
                diagnosis_html += "<span style='font-size:13px;'>• Các chỉ số y khoa nằm trong giới hạn an toàn.</span><br>"

            diagnosis_html += "<h3 style='color:#38bdf8; margin-top:10px; margin-bottom:2px; font-size:14px;'>[3] DỰ ĐOÁN TỪ MÔ HÌNH AI</h3>"
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                prob = model.predict_proba([ai_feats])[0][1]
                status = "NGUY CƠ ĐỘT QUỴ CAO" if prob > 0.6 else "NGUY CƠ THẤP"
                color = "#ef4444" if prob > 0.6 else "#22c55e"

                diagnosis_html += f"<div style='background:#0f172a; padding:10px; border-radius:5px; border-left:5px solid {color}; margin-top:5px;'>"
                diagnosis_html += f"<b style='color:{color}; font-size:16px;'>{status} ({prob * 100:.1f}%)</b>"
                diagnosis_html += f"<br><i style='font-size:12px;'>(Đánh giá tổng hợp dựa trên 7 đặc trưng sinh tồn)</i></div>"
            else:
                diagnosis_html += "<b style='color:#fbd38d;'>Chưa có AI Model. Vui lòng train mô hình.</b>"

            self.lbl_diagnosis.setText(diagnosis_html)

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StrokeApp()
    window.show()
    sys.exit(app.exec())