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


# ══════════════════════════════════════════════════════════════════════════════
# ZoomableImageLabel — Hỗ trợ zoom/pan bản đồ lâm sàng
# ══════════════════════════════════════════════════════════════════════════════

class ZoomableImageLabel(QLabel):
    """
    QLabel mở rộng với khả năng:
    - Ctrl + Scroll wheel: Zoom in/out
    - Click & drag: Pan ảnh
    - Double-click: Mở popup full-screen
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cv_img = None
        self._zoom = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 8.0
        self._pan_offset = QPoint(0, 0)
        self._drag_start = None
        self._pan_start = QPoint(0, 0)

        self._rendering = False  # re-entry guard
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def sizeHint(self):
        """Trả về kích thước gợi ý hợp lý để layout cấp không gian đúng (quan trọng cho fullscreen)."""
        mw = max(self.minimumWidth(), 260)
        mh = max(self.minimumHeight(), 200)
        return QSize(mw, mh)

    def minimumSizeHint(self):
        return QSize(max(self.minimumWidth(), 80), max(self.minimumHeight(), 60))

    def set_cv_image(self, cv_img):
        """Nhận ảnh BGR từ OpenCV."""
        self._cv_img = cv_img.copy()
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._render()

    def _render(self):
        if self._cv_img is None or self._rendering:
            return
        lw, lh = self.width(), self.height()
        if lw < 10 or lh < 10:
            return

        self._rendering = True
        try:
            h, w = self._cv_img.shape[:2]
            rgb = cv2.cvtColor(self._cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(q_img)

            # Scale theo zoom, giới hạn theo kích thước label
            zoom_w = int(w * self._zoom)
            zoom_h = int(h * self._zoom)
            scaled = pix.scaled(zoom_w, zoom_h,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)

            # Vẽ lên canvas cố định kích thước label hiện tại
            canvas = QPixmap(lw, lh)
            canvas.fill(QColor("#020617"))
            painter = QPainter(canvas)
            ox = (lw - scaled.width()) // 2 + self._pan_offset.x()
            oy = (lh - scaled.height()) // 2 + self._pan_offset.y()
            painter.drawPixmap(ox, oy, scaled)

            # Hint text khi zoom = 1.0
            if self._zoom <= 1.0:
                painter.setPen(QColor(100, 150, 200, 160))
                f = painter.font()
                f.setPointSize(8)
                painter.setFont(f)
                painter.drawText(6, lh - 8, "Ctrl+Scroll: Zoom | Drag: Pan | Dbl-click: Phong to")

            painter.end()
            # Dùng QLabel.setPixmap trực tiếp (không qua self.setPixmap để tránh sizeHint)
            QLabel.setPixmap(self, canvas)
        finally:
            self._rendering = False

    def wheelEvent(self, event: QWheelEvent):
        """Ctrl + Scroll → zoom."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else (1 / 1.15)
            new_zoom = float(np.clip(self._zoom * factor, self._zoom_min, self._zoom_max))
            self._zoom = new_zoom
            self._render()
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._pan_start = QPoint(self._pan_offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None:
            delta = event.position().toPoint() - self._drag_start
            self._pan_offset = self._pan_start + delta
            self._render()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_start = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def open_fullscreen(self):
        """Mở dialog phóng to."""
        if self._cv_img is None:
            return
        dlg = _FullscreenImageDialog(self._cv_img, self)
        dlg.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._rendering:
            self._render()


class _FullscreenImageDialog(QDialog):
    """Dialog hiển thị ảnh toàn màn hình với scroll zoom."""

    def __init__(self, cv_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bản đồ lâm sàng — Phóng to")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet("background: #020617;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = ZoomableImageLabel()
        lbl.setStyleSheet("background: #020617;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.set_cv_image(cv_img)
        layout.addWidget(lbl, stretch=1)  # stretch=1 để label chiếm toàn bộ không gian

        hint = QLabel("ESC / đóng cửa sổ để quay lại  |  Ctrl+Scroll: Zoom  |  Kéo: Pan")
        hint.setStyleSheet("color: #64748b; font-size: 11px; padding: 4px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


# ══════════════════════════════════════════════════════════════════════════════
# StrokeApp — Main Application Window
# ══════════════════════════════════════════════════════════════════════════════

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
        central_widget.setStyleSheet("background-color: #0f172a;")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("QSplitter::handle { background: #1f2937; width: 4px; }")
        main_layout.addWidget(splitter)

        # ─────────────────────────────────────────────────────────────────────
        # CỘT 1: NHẬP LIỆU + BẢN ĐỒ LÂM SÀNG
        # ─────────────────────────────────────────────────────────────────────
        left_container = QFrame()
        left_container.setStyleSheet("QFrame { background:#111827; border:1px solid #1f2937; border-radius:10px; }")
        side_panel = QVBoxLayout(left_container)
        side_panel.setContentsMargins(10, 10, 10, 10)
        side_panel.setSpacing(8)

        title = QLabel("VESSEL-EYE v2.0")
        title.setStyleSheet("color: #38bdf8; font-size: 24px; font-weight: bold; margin-bottom: 4px; border: none;")
        side_panel.addWidget(title)

        self.lbl_orig = QLabel("ẢNH GỐC")
        self.lbl_orig.setMinimumSize(260, 240)
        self.lbl_orig.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_orig.setStyleSheet(
            "border: 2px dashed #334155; border-radius: 12px; background: #1e293b; color: #94a3b8;")
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_panel.addWidget(self.lbl_orig)

        lbl_clinical_t = QLabel("🗺  BẢN ĐỒ LÂM SÀNG")
        lbl_clinical_t.setStyleSheet(
            "color: #38bdf8; font-size: 11px; font-weight: bold; margin-top: 4px; border: none;")
        lbl_clinical_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_panel.addWidget(lbl_clinical_t)

        # Hàng nút: zoom hint + nút phóng to
        clinical_ctrl = QHBoxLayout()
        hint_zoom = QLabel("Ctrl+Scroll: Zoom  |  Kéo: Pan")
        hint_zoom.setStyleSheet("color: #475569; font-size: 9px; border: none;")
        clinical_ctrl.addWidget(hint_zoom, stretch=1)

        self.btn_fullscreen = QPushButton("🔍 Phóng to")
        self.btn_fullscreen.setFixedSize(72, 22)
        self.btn_fullscreen.setStyleSheet(
            "QPushButton { background: #1e3a5f; color: #38bdf8; border-radius: 4px; "
            "font-size: 9px; font-weight: bold; border: 1px solid #334155; padding: 2px; }"
            "QPushButton:hover { background: #1d4ed8; color: white; }"
        )
        self.btn_fullscreen.clicked.connect(lambda: self.lbl_clinical.open_fullscreen())
        clinical_ctrl.addWidget(self.btn_fullscreen)
        side_panel.addLayout(clinical_ctrl)

        # ZoomableImageLabel thay thế QLabel cũ
        self.lbl_clinical = ZoomableImageLabel()
        self.lbl_clinical.setMinimumSize(260, 200)
        self.lbl_clinical.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_clinical.setStyleSheet(
            "border: 1px solid #334155; border-radius: 12px; background: #020617; color: #94a3b8;"
        )
        self.lbl_clinical.setText("CHƯA CÓ ẢNH")
        side_panel.addWidget(self.lbl_clinical)

        btn_upload = QPushButton("📥 TẢI ẢNH & PHÂN TÍCH")
        btn_upload.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; border-radius: 8px; "
            "padding: 16px; font-weight: bold; font-size: 13px; border: none; }"
            "QPushButton:hover { background-color: #0369a1; }"
            "QPushButton:pressed { background-color: #075985; }")
        btn_upload.clicked.connect(self.upload_and_process)
        side_panel.addWidget(btn_upload)
        side_panel.addStretch()
        splitter.addWidget(left_container)

        # ─────────────────────────────────────────────────────────────────────
        # CỘT 2: 4 ẢNH XỬ LÝ (2×2 grid)
        # ─────────────────────────────────────────────────────────────────────
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setFrameShape(QFrame.Shape.NoFrame)
        center_scroll.setStyleSheet("QScrollArea { background: #0f172a; border: none; }")
        center_content = QWidget()
        center_content.setStyleSheet("background: #0f172a;")
        center_scroll.setWidget(center_content)
        center_panel = QVBoxLayout(center_content)
        center_panel.setContentsMargins(0, 0, 0, 0)
        grid_imgs = QGridLayout()
        grid_imgs.setHorizontalSpacing(10)
        grid_imgs.setVerticalSpacing(10)

        self.lbl_od, self.cont_od = self.img_box(
            "🎯  OPTIC DISC — Vùng Zone B phân tích")
        self.lbl_mask, self.cont_mask = self.img_box(
            "🩸  A/V CALIBRE — Đường kính động/tĩnh mạch (nhiệt độ màu)")
        self.lbl_skel, self.cont_skel = self.img_box(
            "🔬  PHÂN ĐOẠN MẠCH — Trích xuất mạch máu rõ nét (B&W)")
        self.lbl_closing, self.cont_closing = self.img_box(
            "🌡  HEAT-MAP HẸP/PHỒNG — Biến đổi đường kính dọc mạch")

        grid_imgs.addWidget(self.cont_od, 0, 0)
        grid_imgs.addWidget(self.cont_mask, 0, 1)
        grid_imgs.addWidget(self.cont_skel, 1, 0)
        grid_imgs.addWidget(self.cont_closing, 1, 1)
        grid_imgs.setColumnStretch(0, 1)
        grid_imgs.setColumnStretch(1, 1)
        center_panel.addLayout(grid_imgs)
        splitter.addWidget(center_scroll)

        # ─────────────────────────────────────────────────────────────────────
        # CỘT 3: CHỈ SỐ & KẾT LUẬN
        # ─────────────────────────────────────────────────────────────────────
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setFrameShape(QFrame.Shape.NoFrame)
        result_scroll.setStyleSheet("QScrollArea { background: #0f172a; border: none; }")

        result_content = QWidget()
        # ── FIX: đặt màu nền đồng bộ với toàn bộ app ──
        result_content.setStyleSheet("QWidget { background: #0f172a; }")
        result_scroll.setWidget(result_content)
        result_panel = QVBoxLayout(result_content)
        result_panel.setContentsMargins(4, 4, 4, 4)
        result_panel.setSpacing(8)

        header_metrics = QLabel("<b style='color: #94a3b8; font-size: 11px;'>THÔNG SỐ ĐỊNH LƯỢNG:</b>")
        header_metrics.setStyleSheet("background: transparent; border: none;")
        result_panel.addWidget(header_metrics)

        self.table = QTableWidget(9, 3)
        self.table.setHorizontalHeaderLabels(["Chỉ số", "Giá trị", "Tham chiếu"])
        self.table.setStyleSheet(
            "QTableWidget { background: #1e293b; color: #e2e8f0; border-radius: 10px; "
            "border: 1px solid #334155; gridline-color: #334155; }"
            "QHeaderView::section { background: #0f172a; color: #38bdf8; "
            "border: none; font-weight: bold; padding: 4px; }"
            "QTableWidget::item { padding: 3px; }"
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(260)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        result_panel.addWidget(self.table)

        header_diag = QLabel("<b style='color: #94a3b8; font-size: 11px;'>KẾT LUẬN LÂM SÀNG AI:</b>")
        header_diag.setStyleSheet("background: transparent; border: none;")
        result_panel.addWidget(header_diag)

        self.lbl_diagnosis = QLabel("Sẵn sàng phân tích...")
        self.lbl_diagnosis.setWordWrap(True)
        self.lbl_diagnosis.setMinimumHeight(200)
        self.lbl_diagnosis.setStyleSheet(
            "background: #1e293b; color: white; border-radius: 10px; padding: 15px; "
            "border-left: 8px solid #334155; border-top: 1px solid #334155; "
            "border-right: 1px solid #334155; border-bottom: 1px solid #334155;")
        result_panel.addWidget(self.lbl_diagnosis)

        # Chú giải với giao diện đẹp hơn
        legend_box = QGroupBox("CHÚ GIẢI")
        legend_box.setStyleSheet(
            "QGroupBox { color: #38bdf8; font-weight: bold; font-size: 11px; "
            "border: 1px solid #334155; margin-top: 10px; border-radius: 10px; "
            "padding-top: 10px; background: #111827; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 4px 0 4px; }")
        leg_lay = QVBoxLayout()
        leg_lay.setSpacing(5)

        def add_l(c, t):
            r = QHBoxLayout()
            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background: {c}; border-radius: 6px; border: none;")
            txt = QLabel(f"<span style='color: #cbd5e1; font-size: 10px;'>{t}</span>")
            txt.setStyleSheet("background: transparent; border: none;")
            r.addWidget(dot)
            r.addWidget(txt)
            r.addStretch()
            leg_lay.addLayout(r)

        add_l("#ff4444", "🔴 Đỏ/Cam: Động mạch (Artery)")
        add_l("#3b82f6", "🔵 Xanh: Tĩnh mạch (Vein)")
        add_l("#22c55e", "🟢 Xanh lá: Xoắn vặn (Tortuosity)")
        add_l("#f97316", "🟠 Cam: Đứt đoạn mạch (Discontinuity)")
        add_l("#ef4444", "⭕ Vòng đỏ kép: Hẹp động mạch (Narrowing)")
        add_l("#facc15", "🟡 Vàng: Giãn mạch bất thường (Dilation)")
        add_l("#00a5ff", "🔵 Cam sáng: Đĩa thị (Optic Disc)")
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
        container.setStyleSheet(
            "QFrame { background: #1e293b; border-radius: 10px; border: 1px solid #334155; }")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: bold; border: none; background: transparent;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl = QLabel()
        img_lbl.setMinimumSize(300, 220)
        img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        img_lbl.setStyleSheet("background:#020617; border:1px solid #1e3a5f; border-radius:6px;")
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
            QPixmap.fromImage(q_img).scaled(label.width(), label.height(),
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))

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

            # ── Pipeline xử lý ảnh ──
            en, mask, skel, img_no_bg, fov_mask = riched_image.get_enhanced_vessels(img_disp)

            # ── Hiển thị ảnh gốc (đã loại nền) ──
            self.set_label_image(self.lbl_orig, img_no_bg)

            # ── Feature extraction ──
            feats, regions, feat_details = feature_extract.extract_features(
                mask, en, skeleton=skel,
                img_bgr=img_disp, av_model=self.av_model,
                fov_mask=fov_mask, return_details=True
            )
            av_ratio, crae, crve, tort, std_tort, density, fractal_dim, disc_score, endpoint_score, white_score = feats

            od_center = feat_details.get("od_center", (img_disp.shape[1] // 2, img_disp.shape[0] // 2))
            od_radius = feat_details.get("od_radius", 30)

            # ── Panel 1: Optic Disc + Zone B ──
            od_vis = draw.draw_optic_disc_vis(img_no_bg, od_center, od_radius)
            self.set_label_image(self.lbl_od, od_vis)

            # ── Panel 2: A/V Calibre Heat-map ──
            av_calibre = draw.draw_av_calibre_map(
                skel, mask, en, fov_mask=fov_mask,
                img_bgr=img_disp, av_model=self.av_model
            )
            self.set_label_image(self.lbl_mask, av_calibre)

            # ── Panel 3: Phân đoạn mạch máu B&W rõ nét ──
            vessel_bw = draw.draw_vessel_segmentation(mask, en_green=en, fov_mask=fov_mask)
            self.set_label_image(self.lbl_skel, vessel_bw, is_gray=True)

            # ── Panel 4: Heat-map hẹp/phồng ──
            diam_heat = draw.draw_diameter_heatmap(
                skel, mask, en, fov_mask=fov_mask,
                img_bgr=img_disp, av_model=self.av_model
            )
            self.set_label_image(self.lbl_closing, diam_heat)

            # ── Bản đồ lâm sàng (Zoomable) ──
            f_map, debug_map = draw.draw_feature_map(
                img_disp, mask, en, regions,
                img_no_bg=img_no_bg, fov_mask=fov_mask,
                skeleton=skel, img_bgr=img_disp,
                av_model=self.av_model, return_debug=True,
                anatomy_details=feat_details,
            )
            self.lbl_clinical.set_cv_image(f_map)

            # ── Bảng chỉ số ──
            ranges = {"AV": 0.65, "Tort": 1.50, "Density": 0.05, "Endpoint": 0.05}
            metrics = [
                ("Tỷ lệ A/V Ratio",      f"{av_ratio:.4f}",          f"> {ranges['AV']}"),
                ("CRAE (artery calib)",   f"{crae:.2f}",              "tham chiếu nội bộ"),
                ("CRVE (vein calib)",     f"{crve:.2f}",              "tham chiếu nội bộ"),
                ("Độ cong TB (Tort)",     f"{tort:.4f}",              f"< {ranges['Tort']}"),
                ("Biến thiên Std",        f"{std_tort:.4f}",          "< 0.20"),
                ("Mật độ vi mạch",        f"{density * 100:.2f}%",    f"> {ranges['Density'] * 100}%"),
                ("Fractal dimension",     f"{fractal_dim:.4f}",       "~1.3-1.7"),
                ("Discontinuity score",   f"{disc_score:.4f}",        "< 0.15"),
                ("Endpoint gap score",    f"{endpoint_score:.4f}",    f"< {ranges['Endpoint']:.2f}"),
            ]

            self.table.setRowCount(9)
            for i, (n, v, r) in enumerate(metrics):
                self.table.setItem(i, 0, QTableWidgetItem(n))
                it_v = QTableWidgetItem(v)
                is_abnormal = (
                    (i == 0 and av_ratio < ranges["AV"]) or          # AV ratio
                    (i == 1 and crae < 2.0) or                        # CRAE cực thấp → RAO
                    (i == 3 and tort > ranges["Tort"]) or             # Tortuosity
                    (i == 5 and density < ranges["Density"]) or       # Mật độ
                    (i == 6 and (fractal_dim < 1.3 or fractal_dim > 1.7)) or  # Fractal ngoài ngưỡng
                    (i == 7 and disc_score > 0.15) or                 # Discontinuity
                    (i == 8 and endpoint_score > ranges["Endpoint"])  # Endpoint gap
                )
                if is_abnormal:
                    it_v.setForeground(QColor("#ef4444"))
                    it_v.setFont(QFont("Arial", 10, QFont.Weight.Bold))
                else:
                    it_v.setForeground(QColor("#22c55e"))
                self.table.setItem(i, 1, it_v)
                self.table.setItem(i, 2, QTableWidgetItem(r))

            # ── Kết luận AI ──
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                prob = model.predict_proba([feats])[0][1]

                # ══ Override cứng cho trường hợp chỉ số cực kỳ bất thường ══
                # ML model có thể bỏ sót RAO/CRVO nặng vì training data không đủ
                # Các quy tắc dựa trên tiêu chuẩn lâm sàng được chứng minh:
                rao_suspected = (
                    av_ratio < 0.30 or                           # AV ratio cực thấp: tắc động mạch
                    (crae < 2.0 and crve > 3.0) or               # Artery gần biến mất, vein còn thấy
                    (fractal_dim < 1.20 and av_ratio < 0.50) or  # Cây mạch rất thưa + hẹp nặng
                    (av_ratio < 0.40 and density < 0.08)         # Hẹp + mật độ thấp kép
                )
                if rao_suspected:
                    prob = max(prob, 0.88)   # Ép ít nhất 88%

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

                # ── Dấu hiệu nặng (ưu tiên hiển thị trước) ──
                if av_ratio < 0.30:
                    pathos.append(f"⚠ AV ratio cực thấp ({av_ratio:.3f}) — nghi ngờ tắc động mạch võng mạc (RAO).")
                elif av_ratio < ranges["AV"]:
                    pathos.append("• Hẹp tiểu động mạch lan tỏa (Retinal Narrowing).")

                if crae < 2.0 and crve > 3.0:
                    pathos.append(f"⚠ Động mạch gần như không phát hiện được (CRAE={crae:.1f}px) — RAO nặng.")

                if fractal_dim < 1.20:
                    pathos.append(f"⚠ Cây mạch cực kỳ thưa (Fractal={fractal_dim:.3f} < 1.2) — mất mạch diện rộng.")
                elif fractal_dim < 1.30:
                    pathos.append(f"• Cây mạch thưa hơn bình thường (Fractal={fractal_dim:.3f}).")

                # ── Dấu hiệu thông thường ──
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
                    f"background:#1e293b; color:white; border-radius:10px; padding:15px; "
                    f"border-left:10px solid {color}; border-top:1px solid #334155; "
                    f"border-right:1px solid #334155; border-bottom:1px solid #334155;"
                )
            else:
                self.lbl_diagnosis.setText(
                    "<b style='color:#fbd38d;'>Chưa có model ML.</b><br>"
                    "Chạy training_model.py để huấn luyện."
                )

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Lỗi xử lý ảnh", str(e) + "\n\n" + traceback.format_exc())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = StrokeApp()
    window.show()
    sys.exit(app.exec())