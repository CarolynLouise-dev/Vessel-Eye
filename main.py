# main.py — Vessel-Eye: Retinal Vessel Image Processing
import sys, cv2, joblib, os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from constant import MODEL_PATH, IMG_SIZE, RISK_DECISION_THRESHOLD
import riched_image, feature_extract, draw
import anatomy
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


class PopupImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cv_img = None
        self._is_gray = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_source_image(self, cv_img, is_gray=False):
        self._cv_img = None if cv_img is None else cv_img.copy()
        self._is_gray = bool(is_gray)

    def open_fullscreen(self):
        if self._cv_img is None:
            return
        img = self._cv_img
        if self._is_gray and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dlg = _FullscreenImageDialog(img, self)
        dlg.exec()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_fullscreen()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


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
        self.deep_models = None
        self.od_models = None
        self.av_ensemble = None
        self.deep_device = "cpu"
        self._deep_backend_error = None
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

        backend_row = QHBoxLayout()
        backend_lbl = QLabel("Backend phân đoạn")
        backend_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; border: none;")
        self.cbo_backend = QComboBox()
        self.cbo_backend.addItem("Classical Frangi", "classical")
        self.cbo_backend.addItem("AutoMorph DL", "automorph")
        self.cbo_backend.setStyleSheet(
            "QComboBox { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; "
            "border-radius: 6px; padding: 6px 8px; }"
            "QComboBox::drop-down { border: none; }"
        )
        backend_row.addWidget(backend_lbl)
        backend_row.addWidget(self.cbo_backend, stretch=1)
        side_panel.addLayout(backend_row)

        self.lbl_backend_hint = QLabel(
            "Classical nhanh hơn; AutoMorph cho phân đoạn deep để đối chiếu lâm sàng."
        )
        self.lbl_backend_hint.setWordWrap(True)
        self.lbl_backend_hint.setStyleSheet("color: #64748b; font-size: 10px; border: none;")
        side_panel.addWidget(self.lbl_backend_hint)

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
            "🩸  PHÂN LOẠI A/V — Mạng Động/Tĩnh mạch")
        self.lbl_skel, self.cont_skel = self.img_box(
            "🔬  BẢN ĐỒ CẤU TRÚC — Đứt đoạn & Xoắn vặn")
        self.lbl_seg, self.cont_seg = self.img_box(
            "🧭  MẠCH MÁU — Phân đoạn rõ nét")
        self.lbl_closing, self.cont_closing = self.img_box(
            "🌡  HEAT-MAP HẸP/PHỒNG — Đường kính tuyệt đối")

        grid_imgs.addWidget(self.cont_od, 0, 0)
        grid_imgs.addWidget(self.cont_mask, 0, 1)
        grid_imgs.addWidget(self.cont_skel, 1, 0)
        grid_imgs.addWidget(self.cont_seg, 1, 1)
        grid_imgs.addWidget(self.cont_closing, 2, 0, 1, 2)
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

        self.table = QTableWidget(12, 3)
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

        add_l("#ef4444", "Đỏ: Động mạch (Artery)")
        add_l("#3b82f6", "Xanh dương: Tĩnh mạch (Vein)")
        add_l("#22c55e", "Xanh lá: Overlap A/V hoặc xoắn vặn")
        add_l("#f97316", "Cam: Đứt đoạn mạch (Gap / Discontinuity)")
        add_l("#ef4444", "Vòng đỏ kép: Hẹp động mạch")
        add_l("#facc15", "Vàng: Phồng hoặc giãn bất thường")
        add_l("#fb923c", "Cam nhạt: Đĩa thị và vùng phân tích quanh OD")
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
        title_row = QHBoxLayout()
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: bold; border: none; background: transparent;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_zoom = QPushButton("Phong to")
        btn_zoom.setFixedHeight(22)
        btn_zoom.setStyleSheet(
            "QPushButton { background: #1e3a5f; color: #38bdf8; border-radius: 5px; "
            "font-size: 9px; font-weight: bold; border: 1px solid #334155; padding: 2px 8px; }"
            "QPushButton:hover { background: #1d4ed8; color: white; }"
        )
        title_row.addWidget(lbl_t, stretch=1)
        title_row.addWidget(btn_zoom)
        img_lbl = PopupImageLabel()
        img_lbl.setMinimumSize(300, 220)
        img_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        img_lbl.setStyleSheet("background:#020617; border:1px solid #1e3a5f; border-radius:6px;")
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_zoom.clicked.connect(img_lbl.open_fullscreen)
        layout.addLayout(title_row)
        layout.addWidget(img_lbl)
        return img_lbl, container

    def set_label_image(self, label, cv_img, is_gray=False):
        self._last_image_refs[id(label)] = (label, cv_img.copy(), is_gray)
        if hasattr(label, "set_source_image"):
            label.set_source_image(cv_img, is_gray=is_gray)
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

    def _load_deep_models(self):
        if self.deep_models is not None:
            return self.deep_models

        try:
            import deep_backend
            models = deep_backend.load_vessel_seg_ensemble(n_models=3, device=self.deep_device)
            if not models:
                self._deep_backend_error = (
                    "Không tải được AutoMorph DL. Ứng dụng sẽ fallback về Classical Frangi."
                )
                self.deep_models = []
            else:
                self.deep_models = models
                self._deep_backend_error = None
        except Exception as exc:
            self.deep_models = []
            self._deep_backend_error = f"AutoMorph DL lỗi tải model: {exc}"

        return self.deep_models

    def _load_od_models(self):
        """Lazy-load OD wnet ensemble (cached). Returns list or None."""
        if self.od_models is not None:
            return self.od_models if self.od_models else None
        try:
            import od_backend
            models = od_backend.load_od_ensemble(n_models=3, device=self.deep_device)
            self.od_models = models if models else []
        except Exception as exc:
            import warnings
            warnings.warn(f"[main] OD backend load failed: {exc}")
            self.od_models = []
        return self.od_models if self.od_models else None

    def _load_av_ensemble(self):
        """Lazy-load AV Generator ensemble (cached). Returns list or None."""
        if self.av_ensemble is not None:
            return self.av_ensemble if self.av_ensemble else None
        try:
            import av_backend
            ensemble = av_backend.load_av_ensemble(n_models=2, device=self.deep_device)
            self.av_ensemble = ensemble if ensemble else []
        except Exception as exc:
            import warnings
            warnings.warn(f"[main] AV backend load failed: {exc}")
            self.av_ensemble = []
        return self.av_ensemble if self.av_ensemble else None

    def _run_selected_backend(self, img_disp):
        backend_requested = self.cbo_backend.currentData()
        if backend_requested == "automorph":
            models = self._load_deep_models()
            if models:
                import deep_backend
                en, mask, skel, img_no_bg, fov_mask, pipe_details = deep_backend.get_enhanced_vessels_deep(
                    img_disp,
                    models=models,
                    device=self.deep_device,
                    return_details=True,
                )
                return {
                    "backend_requested": "automorph",
                    "backend_used": "automorph",
                    "backend_label": "AutoMorph DL",
                    "backend_note": "Đang dùng vessel segmentation deep từ AutoMorph.",
                    "backend_warning": None,
                    "en": en,
                    "mask": mask,
                    "skel": skel,
                    "img_no_bg": img_no_bg,
                    "fov_mask": fov_mask,
                    "raw_vessel_mask": pipe_details.get("raw_vessel_mask"),
                }

            return {
                "backend_requested": "automorph",
                "backend_used": "classical",
                "backend_label": "Classical Frangi (fallback)",
                "backend_note": "AutoMorph không sẵn sàng nên đã tự động fallback về Classical.",
                "backend_warning": self._deep_backend_error,
                "en": None,
                "mask": None,
                "skel": None,
                "img_no_bg": None,
                "fov_mask": None,
                "raw_vessel_mask": None,
            }

        en, mask, skel, img_no_bg, fov_mask, pipe_details = riched_image.get_enhanced_vessels(img_disp, return_details=True)
        return {
            "backend_requested": "classical",
            "backend_used": "classical",
            "backend_label": "Classical Frangi",
            "backend_note": "Đang dùng pipeline segmentation cổ điển hiện tại.",
            "backend_warning": None,
            "en": en,
            "mask": mask,
            "skel": skel,
            "img_no_bg": img_no_bg,
            "fov_mask": fov_mask,
            "raw_vessel_mask": pipe_details.get("raw_vessel_mask"),
        }

    def upload_and_process(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.jpg *.png *.jpeg)")
        if not file_path:
            return

        try:
            img_raw = cv2.imread(file_path)
            img_disp = input_data.standardize_fundus_image(img_raw, IMG_SIZE)

            # ── Pipeline xử lý ảnh ──
            backend_result = self._run_selected_backend(img_disp)
            en = backend_result["en"]
            mask = backend_result["mask"]
            skel = backend_result["skel"]
            img_no_bg = backend_result["img_no_bg"]
            fov_mask = backend_result["fov_mask"]
            raw_vessel_mask = backend_result.get("raw_vessel_mask")
            if en is None:
                en, mask, skel, img_no_bg, fov_mask, pipe_details = riched_image.get_enhanced_vessels(img_disp, return_details=True)
                raw_vessel_mask = pipe_details.get("raw_vessel_mask")

            # ── Hiển thị ảnh gốc (đã loại nền) ──
            self.set_label_image(self.lbl_orig, img_no_bg)

            # ── Deep OD detection + Deep A/V classification (AutoMorph mode only) ──
            od_center_override = od_radius_override = None
            od_conf_deep = None
            od_mask_override = None
            artery_mask = vein_mask = None
            heuristic_od_center = None
            heuristic_od_radius = None
            heuristic_od_details = {}

            try:
                heuristic_od_center, heuristic_od_radius, heuristic_od_details = anatomy.detect_optic_disc(
                    img_disp,
                    fov_mask=fov_mask,
                    vessel_mask=mask,
                    return_details=True,
                )
            except Exception:
                heuristic_od_center = None
                heuristic_od_radius = None
                heuristic_od_details = {}

            if backend_result["backend_used"] == "automorph":
                od_mods = self._load_od_models()
                if od_mods:
                    try:
                        import od_backend
                        od_center_override, od_radius_override, od_conf_deep, od_mask_override = \
                            od_backend.detect_optic_disc_deep(img_disp, od_mods, self.deep_device)
                        od_area_ratio = 0.0
                        if od_mask_override is not None:
                            od_area_ratio = float(np.count_nonzero(od_mask_override)) / float(od_mask_override.size)
                        od_radius_ratio = float(od_radius_override) / max(1.0, float(min(img_disp.shape[:2])))
                        od_dist_ok = True
                        vessel_support_ok = True
                        if heuristic_od_center is not None and heuristic_od_radius is not None:
                            deep_shift = float(np.hypot(
                                float(od_center_override[0] - heuristic_od_center[0]),
                                float(od_center_override[1] - heuristic_od_center[1]),
                            ))
                            od_dist_ok = deep_shift <= max(28.0, 2.6 * float(heuristic_od_radius))
                            vessel_support_ok = float(heuristic_od_details.get("vessel_support", 0.0)) >= 0.12
                        od_valid = (
                            od_mask_override is not None
                            and 0.002 <= od_area_ratio <= 0.08
                            and 0.025 <= od_radius_ratio <= 0.14
                            and float(od_conf_deep) >= 0.35
                            and (od_dist_ok or (float(od_conf_deep) >= 0.72 and vessel_support_ok))
                        )
                        if not od_valid:
                            od_center_override = None
                            od_radius_override = None
                            od_conf_deep = None
                            od_mask_override = None
                    except Exception as _e:
                        pass  # Fall back to heuristic silently

                av_ens = self._load_av_ensemble()
                if av_ens:
                    try:
                        import av_backend
                        artery_mask, vein_mask = av_backend.segment_av_deep(
                            img_disp, av_ens, self.deep_device
                        )
                        art_px = int(np.count_nonzero(artery_mask)) if artery_mask is not None else 0
                        vein_px = int(np.count_nonzero(vein_mask)) if vein_mask is not None else 0
                        overlap_px = int(np.count_nonzero((artery_mask > 0) & (vein_mask > 0))) if artery_mask is not None and vein_mask is not None else 0
                        av_valid = (
                            artery_mask is not None and vein_mask is not None
                            and art_px > 120 and vein_px > 120
                            and overlap_px < 0.85 * min(art_px, vein_px)
                        )
                        if not av_valid:
                            artery_mask = None
                            vein_mask = None
                    except Exception as _e:
                        pass  # Fall back to SVM silently

            # ── Feature extraction ──
            feats, regions, feat_details = feature_extract.extract_features(
                mask, en, skeleton=skel,
                img_bgr=img_disp, av_model=self.av_model,
                fov_mask=fov_mask, return_details=True,
                od_center_override=od_center_override,
                od_radius_override=od_radius_override,
                od_mask_override=od_mask_override,
                artery_mask=artery_mask,
                vein_mask=vein_mask,
                raw_vessel_mask=raw_vessel_mask,
            )
            av_ratio, crae, crve, tort, std_tort, density, fractal_dim, disc_score, endpoint_score, white_score = feats

            od_center = feat_details.get("od_center", (img_disp.shape[1] // 2, img_disp.shape[0] // 2))
            od_radius = feat_details.get("od_radius", 30)
            od_details = feat_details.get("od_details", {})
            od_mask = feat_details.get("od_mask")
            zone_b_mask = feat_details.get("zone_b_mask")
            quality = feat_details.get("quality", {})
            quality_score = float(quality.get("quality_score", 0.0))
            quality_level = quality.get("quality_level", "unknown")
            quality_action = quality.get("quality_action", "review")
            quality_reasons = quality.get("reasons", [])
            low_visibility_may_be_pathology = bool(
                quality.get("low_vessel_visibility_may_be_pathology", False)
            )
            # Deep OD confidence overrides heuristic confidence
            if od_conf_deep is not None:
                od_confidence = float(od_conf_deep)
            else:
                od_confidence = float(od_details.get("confidence", 0.0))

            # ── Panel 1: Optic Disc + Zone B ──
            od_vis = draw.draw_optic_disc_vis(
                img_no_bg,
                od_center,
                od_radius,
                zone_b_mask=zone_b_mask,
                disc_mask=od_mask,
                confidence=od_confidence,
            )
            self.set_label_image(self.lbl_od, od_vis)

            # ── Panel 2: A/V Calibre Heat-map ──
            av_calibre = draw.draw_av_calibre_map(
                skel, mask, en, fov_mask=fov_mask,
                img_bgr=img_disp, av_model=self.av_model,
                artery_mask=artery_mask, vein_mask=vein_mask
            )
            self.set_label_image(self.lbl_mask, av_calibre)

            # ── Panel 3: Bản đồ Cấu trúc (Đứt đoạn & Xoắn vặn) ──
            structural_map = draw.draw_structural_map(
                skel, mask, en, fov_mask=fov_mask,
                img_bgr=img_disp, av_model=self.av_model,
                artery_mask=artery_mask, vein_mask=vein_mask,
                raw_vessel_mask=raw_vessel_mask,
            )
            self.set_label_image(self.lbl_skel, structural_map)

            # ── Panel 4: Phân đoạn mạch máu B&W rõ nét ──
            vessel_bw = draw.draw_vessel_segmentation(mask, en_green=en, fov_mask=fov_mask)
            self.set_label_image(self.lbl_seg, vessel_bw, is_gray=True)

            # ── Panel 5: Heat-map hẹp/phồng ──
            diam_heat = draw.draw_diameter_heatmap(
                skel, mask, en, fov_mask=fov_mask,
                img_bgr=img_disp, av_model=self.av_model,
                artery_mask=artery_mask, vein_mask=vein_mask,
                anatomy_details=feat_details,
                raw_vessel_mask=raw_vessel_mask,
            )
            self.set_label_image(self.lbl_closing, diam_heat)

            # ── Bản đồ lâm sàng (Zoomable) ──
            f_map, debug_map = draw.draw_feature_map(
                img_disp, mask, en, regions,
                img_no_bg=img_no_bg, fov_mask=fov_mask,
                skeleton=skel, img_bgr=img_disp,
                av_model=self.av_model, return_debug=True,
                anatomy_details=feat_details,
                artery_mask=artery_mask, vein_mask=vein_mask,
                raw_vessel_mask=raw_vessel_mask,
            )
            self.lbl_clinical.set_cv_image(f_map)

            findings = draw.analyze_pathology_findings(
                skel, mask, en,
                fov_mask=fov_mask,
                img_bgr=img_disp,
                av_model=self.av_model,
                artery_mask=artery_mask,
                vein_mask=vein_mask,
                anatomy_details=feat_details,
                raw_vessel_mask=raw_vessel_mask,
            )

            # ── Bảng chỉ số ──
            ranges = {"AV": 0.65, "Tort": 1.50, "Density": 0.05, "Endpoint": 0.05}
            metrics = [
                ("Backend phân đoạn",      backend_result["backend_label"], "Classical / AutoMorph"),
                ("Độ tin cậy ảnh",         f"{quality_score * 100:.1f}%",   quality_level),
                ("Độ tin cậy đĩa thị",     f"{od_confidence * 100:.1f}%",    "> 40%"),
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

            self.table.setRowCount(len(metrics))
            for i, (n, v, r) in enumerate(metrics):
                self.table.setItem(i, 0, QTableWidgetItem(n))
                it_v = QTableWidgetItem(v)
                is_abnormal = (
                    (i == 1 and quality_score < 0.68) or               # Image quality confidence
                    (i == 2 and od_confidence < 0.40) or               # Optic disc confidence
                    (i == 3 and av_ratio < ranges["AV"]) or           # AV ratio
                    (i == 4 and crae < 2.0) or                         # CRAE cực thấp → RAO
                    (i == 6 and tort > ranges["Tort"]) or             # Tortuosity
                    (i == 8 and density < ranges["Density"]) or       # Mật độ
                    (i == 9 and (fractal_dim < 1.3 or fractal_dim > 1.7)) or  # Fractal ngoài ngưỡng
                    (i == 10 and disc_score > 0.15) or                 # Discontinuity
                    (i == 11 and endpoint_score > ranges["Endpoint"]) # Endpoint gap
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
                top_findings = findings.get("summary", [])[:5]
                visible_marker_count = sum(len(findings.get(name, [])) for name in ("narrowing", "dilation", "tortuosity", "gaps"))

                evidence_items = []
                if av_ratio < ranges["AV"]:
                    evidence_items.append(("AVR thấp", max(0.0, (ranges["AV"] - av_ratio) / max(ranges["AV"], 1e-6))))
                if crae < 2.0:
                    evidence_items.append(("CRAE rất thấp", max(0.0, (2.0 - crae) / 2.0)))
                if density < ranges["Density"]:
                    evidence_items.append(("Mật độ mạch thấp", max(0.0, (ranges["Density"] - density) / max(ranges["Density"], 1e-6))))
                if fractal_dim < 1.30:
                    evidence_items.append(("Fractal thấp", max(0.0, (1.30 - fractal_dim) / 1.30)))
                if disc_score > 0.15:
                    evidence_items.append(("Discontinuity cao", max(0.0, (disc_score - 0.15) / 0.15)))
                if endpoint_score > ranges["Endpoint"]:
                    evidence_items.append(("Endpoint gap cao", max(0.0, (endpoint_score - ranges["Endpoint"]) / max(ranges["Endpoint"], 1e-6))))
                if white_score > 0.03:
                    evidence_items.append(("Whitening cao", max(0.0, (white_score - 0.03) / 0.03)))

                evidence_items.sort(key=lambda item: -float(item[1]))
                dominant_evidence = [name for name, _ in evidence_items[:3]]
                narrowing_count = len(findings.get("narrowing", []))

                # Kiểm tra dấu hiệu lâm sàng RAO (chỉ để cảnh báo, không ép xác suất ML)
                rao_suspected = (
                    av_ratio < 0.30 or                           # AV ratio cực thấp
                    (crae < 2.0 and crve > 3.0) or               # Artery gần biến mất, vein còn thấy
                    (fractal_dim < 1.20 and av_ratio < 0.50) or  # Cây mạch rất thưa + hẹp nặng
                    (av_ratio < 0.40 and density < 0.08)         # Hẹp + mật độ thấp kép
                )

                morphology_high_concern = (
                    rao_suspected or
                    ((av_ratio < 0.55) and (fractal_dim < 1.30)) or
                    ((av_ratio < 0.55) and (density < 0.06)) or
                    ((crae < 2.4) and (av_ratio < 0.60)) or
                    ((narrowing_count >= 2) and (fractal_dim < 1.30))
                )

                if prob >= 0.70:
                    status = "NGUY CƠ CAO"
                    color = "#ef4444"
                elif prob >= RISK_DECISION_THRESHOLD:
                    status = "NGHI NGỜ NGUY CƠ"
                    color = "#f59e0b"
                else:
                    status = "NGUY CƠ THẤP"
                    color = "#22c55e"

                escalated_by_rules = False
                if prob < RISK_DECISION_THRESHOLD and morphology_high_concern:
                    status = "NGHI NGỜ NGUY CƠ"
                    color = "#f59e0b"
                    escalated_by_rules = True

                review_needed = (
                    quality_action != "proceed" or
                    od_confidence < 0.40 or
                    backend_result["backend_used"] != backend_result["backend_requested"]
                )

                msg = f"<h2 style='color:{color};'>{status} ({prob * 100:.1f}%)</h2>"
                if escalated_by_rules:
                    msg += (
                        "<div style='margin:4px 0 8px 0; padding:8px 10px; border-radius:8px; "
                        "background:#3b2f12; border:1px solid #d97706; color:#fbbf24; font-weight:bold;'>"
                        "⚠ Xác suất ML còn thấp nhưng hình thái mạch máu đang đáng ngờ, nên không xếp vào nhóm an toàn."
                        "</div>"
                    )
                if rao_suspected:
                    msg += (
                        "<div style='margin:4px 0 8px 0; padding:8px 10px; border-radius:8px; "
                        "background:#431407; border:1px solid #ea580c; color:#fb923c; font-weight:bold;'>"
                        "⚠ Nghi ngờ tắc động mạch võng mạc (RAO) — cần bác sĩ đánh giá trực tiếp."
                        "</div>"
                    )
                if review_needed:
                    msg += (
                        "<div style='margin:6px 0 10px 0; padding:8px 10px; border-radius:8px; "
                        "background:#1e293b; border:1px solid #334155; color:#fbbf24; font-weight:bold;'>"
                        "CẦN BÁC SĨ XEM LẠI VÌ ĐỘ TIN CẬY CHƯA CAO"
                        "</div>"
                    )
                msg += "<p><b>Phát hiện bệnh lý:</b></p>"

                pathos = []
                pathos.append(f"• Backend phân đoạn đang dùng: {backend_result['backend_label']}.")
                if escalated_by_rules:
                    if dominant_evidence:
                        pathos.append(
                            "• Case này được nâng mức cảnh báo theo rule-based morphology: "
                            + ", ".join(dominant_evidence)
                            + "."
                        )
                    else:
                        pathos.append("• Case này được nâng mức cảnh báo theo rule-based morphology dù xác suất ML hiện còn thấp.")
                if prob >= RISK_DECISION_THRESHOLD:
                    if visible_marker_count <= 2:
                        if dominant_evidence:
                            pathos.append(
                                "• Mức nguy cơ hiện tại đến chủ yếu từ model ML trên đặc trưng toàn cục: "
                                + ", ".join(dominant_evidence)
                                + "."
                            )
                        else:
                            pathos.append("• Mức nguy cơ hiện tại đến chủ yếu từ model ML toàn cục, không phải từ số lượng marker cục bộ đang hiển thị.")
                    else:
                        pathos.append("• Mức nguy cơ hiện tại là kết hợp giữa marker cục bộ trên map và model ML toàn cục.")
                if backend_result["backend_warning"]:
                    pathos.append(f"• {backend_result['backend_warning']}")
                elif backend_result["backend_note"]:
                    pathos.append(f"• {backend_result['backend_note']}")

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
                if top_findings:
                    pathos.append("<b>• Vị trí nổi bật:</b>")
                    for marker in top_findings:
                        label = marker.get("type", "Finding")
                        cx = int(marker.get("cx", 0))
                        cy = int(marker.get("cy", 0))
                        score = float(marker.get("score", 0.0))
                        value = marker.get("value", None)
                        if value is None:
                            pathos.append(f"• {label} tại ({cx}, {cy}) | score={score:.2f}.")
                        elif "narrow" in label.lower() or "dilation" in label.lower():
                            pathos.append(f"• {label} tại ({cx}, {cy}) | diam={float(value):.1f}px, score={score:.2f}.")
                        elif "tortuosity" in label.lower():
                            pathos.append(f"• {label} tại ({cx}, {cy}) | tort={float(value):.2f}, score={score:.2f}.")
                        else:
                            pathos.append(f"• {label} tại ({cx}, {cy}) | area={float(value):.0f}, score={score:.2f}.")
                if quality_action != "proceed":
                    pathos.append(f"• Độ tin cậy ảnh ở mức {quality_level}; nên đọc kết quả cùng đánh giá lâm sàng trực tiếp.")
                if od_confidence < 0.40:
                    pathos.append("• Tâm đĩa thị được phát hiện với độ tin cậy thấp; các chỉ số quanh Zone B có thể dao động.")
                if low_visibility_may_be_pathology:
                    pathos.append("• Mạch máu hiện thưa nhưng ảnh không quá mờ; không nên tự động coi đây chỉ là ảnh kém chất lượng.")
                for reason in quality_reasons[:2]:
                    if reason not in pathos:
                        pathos.append(f"• {reason}")
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