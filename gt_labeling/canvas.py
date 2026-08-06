"""自繪的影像/標註畫布。

為何自繪 QWidget 而非 QGraphicsView:本工具的正確性關鍵是「normalized 座標不漂移」。
自繪只有一組 ``ViewTransform``(zoom + offset),model 的 normalized bbox 是唯一真值,
滑鼠與繪製都走同一條換算;QGraphicsView 會多出 item local / scene / view 三層座標
與 item 狀態對 JSON 的雙向同步,正是座標漂移的溫床。代價是 zoom/pan/hit-test 要自己
實作(約 150 行),換來的好處是控制點固定像素大小、undo 就是 dets 快照。

繪製效能:縮小時(zoom<1,一般檢視狀態)用預先縮好的 pixmap,平移只是 blit;放大時
(zoom>=1)用 source rect 只畫可見區域。兩者都不重新解碼 JPEG。
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

from .model import Det, FrameLabel, canonical_bbox
from .transform import ViewTransform

HANDLES: tuple[tuple[int, int], ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)
HANDLE_HALF = 4.0
HANDLE_HIT = 7.0
SMALL_BOX_PX = 12.0
SMALL_BOX_PAD = 4.0
NEW_MIN_PX = 4.0

COLOR_BG = QColor("#1e1e1e")
COLOR_OK = QColor("#2ecc40")
COLOR_NG = QColor("#ff9500")
COLOR_UNSET = QColor("#b0b0b0")
COLOR_DRONE = QColor("#ff3b30")
COLOR_SELECT = QColor("#ffffff")
COLOR_BAND = QColor("#4fa3ff")
COLOR_OUTSIDE = QColor(0, 0, 0, 70)
COLOR_IMAGE_EDGE = QColor("#3c3c3c")

HANDLE_CURSORS = {
    (-1, -1): Qt.CursorShape.SizeFDiagCursor,
    (1, 1): Qt.CursorShape.SizeFDiagCursor,
    (1, -1): Qt.CursorShape.SizeBDiagCursor,
    (-1, 1): Qt.CursorShape.SizeBDiagCursor,
    (0, -1): Qt.CursorShape.SizeVerCursor,
    (0, 1): Qt.CursorShape.SizeVerCursor,
    (-1, 0): Qt.CursorShape.SizeHorCursor,
    (1, 0): Qt.CursorShape.SizeHorCursor,
}


class Mode(Enum):
    IDLE = auto()
    PAN = auto()
    MOVE = auto()
    RESIZE = auto()
    NEW = auto()


def det_color(det: Det) -> QColor:
    if not det.is_person:
        return COLOR_DRONE
    if det.ppe == "ok":
        return COLOR_OK
    if det.ppe == "ng":
        return COLOR_NG
    return COLOR_UNSET


def _clamp01(p: QPointF) -> QPointF:
    return QPointF(min(max(p.x(), 0.0), 1.0), min(max(p.y(), 0.0), 1.0))


class ImageCanvas(QWidget):
    selectionChanged = pyqtSignal(int)
    detsEdited = pyqtSignal()
    viewChanged = pyqtSignal()
    navigateRequested = pyqtSignal(int)
    hoverMoved = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # 新框的預設屬性由外部決定(右側「新框預設」+ 全域取號),canvas 只負責畫。
        self.new_det_factory: Callable[[], Det] = Det

        self.tf = ViewTransform()
        self._frame: FrameLabel | None = None
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._scaled_key: tuple | None = None
        self._needs_fit = True

        self._sel = -1
        self._mode = Mode.IDLE
        self._handle = 0
        self._drag_n0 = QPointF()
        self._orig_bbox: list[float] = []
        self._new_det: Det | None = None
        self._pan_last = QPointF()
        self._space_held = False

        self._band: tuple[float, float] | None = None
        self._label_font = QFont()
        self._label_font.setPointSize(10)

    # ------------------------------------------------------------------ 對外狀態

    @property
    def frame(self) -> FrameLabel | None:
        return self._frame

    @property
    def selected_index(self) -> int:
        return self._sel

    @property
    def selected_det(self) -> Det | None:
        if self._frame is None or not (0 <= self._sel < len(self._frame.dets)):
            return None
        return self._frame.dets[self._sel]

    def set_frame(self, frame: FrameLabel | None, pixmap: QPixmap | None) -> None:
        prev_size = None
        if self._frame is not None:
            try:
                prev_size = self._frame.size
            except ValueError:
                prev_size = None

        self._frame = frame
        self._pixmap = pixmap
        self._scaled = None
        self._scaled_key = None
        self._mode = Mode.IDLE
        self._new_det = None

        if frame is not None:
            width, height = frame.size
            self.tf.set_image_size(width, height)
            # 影像尺寸相同就保留 zoom/pan:逐幀比對同一區域是這工具的主要用法。
            if prev_size != (width, height):
                self._needs_fit = True
                self._apply_fit()
            else:
                self.tf.clamp_offset(self.size())

        self._set_selection(-1)
        self.viewChanged.emit()
        self.update()

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        """影像延後解碼完成後補上,不動座標狀態。"""
        self._pixmap = pixmap
        self._scaled = None
        self._scaled_key = None
        self.update()

    def set_band(self, lo: float | None, hi: float | None) -> None:
        self._band = None if lo is None or hi is None else (min(lo, hi), max(lo, hi))
        self.update()

    def reload_dets(self) -> None:
        """外部改了 dets(undo/redo、面板編輯)後同步選取範圍與畫面。"""
        if self._frame is None:
            self._set_selection(-1)
        elif self._sel >= len(self._frame.dets):
            self._set_selection(len(self._frame.dets) - 1)
        self.update()

    def select(self, index: int) -> None:
        self._set_selection(index)
        self.update()

    def fit_view(self) -> None:
        self._needs_fit = True
        self._apply_fit()
        self.viewChanged.emit()
        self.update()

    def delete_selected(self) -> bool:
        det = self.selected_det
        if det is None or self._frame is None:
            return False
        del self._frame.dets[self._sel]
        remaining = len(self._frame.dets)
        self._set_selection(min(self._sel, remaining - 1) if remaining else -1)
        self.detsEdited.emit()
        self.update()
        return True

    # ------------------------------------------------------------------- 內部工具

    def _set_selection(self, index: int) -> None:
        index = index if index >= 0 else -1
        if index == self._sel:
            return
        self._sel = index
        self.selectionChanged.emit(index)

    def _apply_fit(self) -> None:
        if self._frame is None or self.width() <= 0 or self.height() <= 0:
            return
        self.tf.fit(self.size())
        self._needs_fit = False

    def _min_zoom(self) -> float:
        return self.tf.fit_zoom(self.size()) * 0.25

    def _handle_center(self, rect: QRectF, hx: int, hy: int) -> QPointF:
        x = rect.left() if hx < 0 else (rect.right() if hx > 0 else rect.center().x())
        y = rect.top() if hy < 0 else (rect.bottom() if hy > 0 else rect.center().y())
        return QPointF(x, y)

    def _hit_handle(self, pos: QPointF) -> int | None:
        det = self.selected_det
        if det is None:
            return None
        rect = self.tf.n2v_rect(det.bbox)
        for i, (hx, hy) in enumerate(HANDLES):
            c = self._handle_center(rect, hx, hy)
            if abs(pos.x() - c.x()) <= HANDLE_HIT and abs(pos.y() - c.y()) <= HANDLE_HIT:
                return i
        return None

    def _hit_rect(self, bbox) -> QRectF:
        rect = self.tf.n2v_rect(bbox)
        if rect.width() < SMALL_BOX_PX or rect.height() < SMALL_BOX_PX:
            return rect.adjusted(-SMALL_BOX_PAD, -SMALL_BOX_PAD, SMALL_BOX_PAD, SMALL_BOX_PAD)
        return rect

    def _hit_box(self, pos: QPointF) -> int | None:
        if self._frame is None:
            return None
        dets = self._frame.dets
        # 已選取的優先(拖曳穩定),其餘由上層(繪製順序在後)往下找。
        order = [self._sel] if 0 <= self._sel < len(dets) else []
        order += range(len(dets) - 1, -1, -1)
        for i in order:
            if self._hit_rect(dets[i].bbox).contains(pos):
                return i
        return None

    def _begin_box_drag(self, mode: Mode, pos: QPointF) -> None:
        det = self.selected_det
        if det is None:
            return
        self._mode = mode
        self._drag_n0 = self.tf.v2n_point(pos)
        self._orig_bbox = list(det.bbox)

    def _apply_move(self, pos: QPointF) -> None:
        det = self.selected_det
        if det is None:
            return
        now = self.tf.v2n_point(pos)
        dx = now.x() - self._drag_n0.x()
        dy = now.y() - self._drag_n0.y()
        x1, y1, x2, y2 = self._orig_bbox
        # 整體平移不變形:位移量夾到框仍在 [0,1] 內。
        dx = min(max(dx, -x1), 1.0 - x2)
        dy = min(max(dy, -y1), 1.0 - y2)
        det.bbox = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        self.update()

    def _apply_resize(self, pos: QPointF) -> None:
        det = self.selected_det
        if det is None:
            return
        now = self.tf.v2n_point(pos)
        dx = now.x() - self._drag_n0.x()
        dy = now.y() - self._drag_n0.y()
        hx, hy = HANDLES[self._handle]
        x1, y1, x2, y2 = self._orig_bbox
        # 每次都從「拖曳起點的原始 bbox」重算,不累加,避免捨入誤差堆積。
        if hx < 0:
            x1 = min(max(x1 + dx, 0.0), 1.0)
        elif hx > 0:
            x2 = min(max(x2 + dx, 0.0), 1.0)
        if hy < 0:
            y1 = min(max(y1 + dy, 0.0), 1.0)
        elif hy > 0:
            y2 = min(max(y2 + dy, 0.0), 1.0)
        det.bbox = [x1, y1, x2, y2]
        self.update()

    def _update_cursor(self, pos: QPointF) -> None:
        if self._space_held:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        handle = self._hit_handle(pos)
        if handle is not None:
            self.setCursor(HANDLE_CURSORS[HANDLES[handle]])
        elif self._hit_box(pos) is not None:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    # --------------------------------------------------------------------- 事件

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._needs_fit:
            self._apply_fit()
        else:
            self.tf.clamp_offset(self.size())
        self.viewChanged.emit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._frame is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # 1.0015**120 ~= 1.20:一個滑鼠刻度約 20%,觸控板小刻度也能平滑縮放。
        if self.tf.zoom_by(1.0015**delta, event.position(), self._min_zoom()):
            self.tf.clamp_offset(self.size())
            self.viewChanged.emit()
            self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame is None:
            return
        pos = event.position()
        button = event.button()
        pan_button = button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton)
        if pan_button or (button == Qt.MouseButton.LeftButton and self._space_held):
            self._mode = Mode.PAN
            self._pan_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if button != Qt.MouseButton.LeftButton:
            return

        handle = self._hit_handle(pos)
        if handle is not None:
            self._handle = handle
            self._begin_box_drag(Mode.RESIZE, pos)
            return

        index = self._hit_box(pos)
        if index is not None:
            self._set_selection(index)
            self._begin_box_drag(Mode.MOVE, pos)
            self.update()
            return

        self._set_selection(-1)
        start = _clamp01(self.tf.v2n_point(pos))
        # 起手就決定屬性,拖曳中的虛線框才能顯示最終顏色(drone = 紅)。
        det = self.new_det_factory()
        det.bbox = [start.x(), start.y(), start.x(), start.y()]
        self._new_det = det
        self._mode = Mode.NEW
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._frame is None:
            return
        pos = event.position()
        n = self.tf.v2n_point(pos)
        self.hoverMoved.emit(n.x(), n.y())

        if self._mode is Mode.PAN:
            self.tf.pan_by(pos.x() - self._pan_last.x(), pos.y() - self._pan_last.y())
            self._pan_last = pos
            self.tf.clamp_offset(self.size())
            self.viewChanged.emit()
            self.update()
        elif self._mode is Mode.MOVE:
            self._apply_move(pos)
        elif self._mode is Mode.RESIZE:
            self._apply_resize(pos)
        elif self._mode is Mode.NEW and self._new_det is not None:
            end = _clamp01(n)
            self._new_det.bbox[2] = end.x()
            self._new_det.bbox[3] = end.y()
            self.update()
        else:
            self._update_cursor(pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        mode, self._mode = self._mode, Mode.IDLE
        pos = event.position()

        if mode is Mode.PAN:
            self._update_cursor(pos)
            return

        if mode is Mode.NEW:
            pending, self._new_det = self._new_det, None
            if pending is not None and self._frame is not None:
                rect = self.tf.n2v_rect(pending.bbox)
                if rect.width() >= NEW_MIN_PX and rect.height() >= NEW_MIN_PX:
                    pending.bbox = canonical_bbox(pending.bbox)
                    self._frame.dets.append(pending)
                    self._set_selection(len(self._frame.dets) - 1)
                    self.detsEdited.emit()
            self.update()
            return

        if mode in (Mode.MOVE, Mode.RESIZE):
            det = self.selected_det
            if det is not None:
                det.bbox = canonical_bbox(det.bbox)
                if det.bbox != canonical_bbox(self._orig_bbox):
                    self.detsEdited.emit()
            self._update_cursor(pos)
            self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return
        if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.navigateRequested.emit(-1)
            return
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.navigateRequested.emit(1)
            return
        # 只吃單獨的 F:Ctrl+F 是視窗層的「找 track」,萬一沒被攔下來(動作停用等)
        # 而落到這裡,不該悄悄變成還原檢視——那會把使用者的縮放位置洗掉。
        if key == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.fit_view()
            return
        if key == Qt.Key.Key_Escape:
            if self._mode is Mode.NEW:
                self._new_det = None
            self._mode = Mode.IDLE
            self._set_selection(-1)
            self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        super().keyReleaseEvent(event)

    # -------------------------------------------------------------------- 繪製

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), COLOR_BG)
        if self._frame is None:
            self._draw_hint(painter, "開啟含 frames/ 與 labels/ 的資料夾(Ctrl+O)")
            painter.end()
            return

        image_rect = self.tf.image_rect()
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QPen(COLOR_IMAGE_EDGE, 1))
            painter.drawRect(image_rect)
            self._draw_hint(painter, "此幀找不到對應影像(frames/ 下無同 stem 檔案)")
        else:
            self._draw_image(painter, image_rect)

        self._draw_band(painter, image_rect)
        painter.setFont(self._label_font)
        self._draw_dets(painter)
        self._draw_new_box(painter)
        painter.end()

    def _draw_hint(self, painter: QPainter, text: str) -> None:
        painter.setPen(QPen(COLOR_UNSET))
        painter.setFont(self._label_font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)

    def _draw_image(self, painter: QPainter, image_rect: QRectF) -> None:
        assert self._pixmap is not None
        if self.tf.zoom < 1.0:
            painter.drawPixmap(image_rect.topLeft(), self._scaled_pixmap())
        else:
            visible = image_rect.intersected(QRectF(self.rect()))
            if visible.isEmpty():
                return
            z = self.tf.zoom
            source = QRectF(
                (visible.left() - image_rect.left()) / z,
                (visible.top() - image_rect.top()) / z,
                visible.width() / z,
                visible.height() / z,
            )
            painter.drawPixmap(visible, self._pixmap, source)
        painter.setPen(QPen(COLOR_IMAGE_EDGE, 1))
        painter.drawRect(image_rect)

    def _scaled_pixmap(self) -> QPixmap:
        assert self._pixmap is not None
        size = QSize(
            max(1, round(self.tf.span_x)),
            max(1, round(self.tf.span_y)),
        )
        key = (self._pixmap.cacheKey(), size.width(), size.height())
        if self._scaled_key != key or self._scaled is None:
            self._scaled = self._pixmap.scaled(
                size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scaled_key = key
        return self._scaled

    def _draw_band(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._band is None:
            return
        lo, hi = self._band
        y_lo = self.tf.n2v(0.0, lo).y()
        y_hi = self.tf.n2v(0.0, hi).y()

        above = QRectF(image_rect.left(), image_rect.top(), image_rect.width(),
                       max(0.0, y_lo - image_rect.top()))
        below = QRectF(image_rect.left(), y_hi, image_rect.width(),
                       max(0.0, image_rect.bottom() - y_hi))
        for band in (above, below):
            clipped = band.intersected(image_rect)
            if not clipped.isEmpty():
                painter.fillRect(clipped, COLOR_OUTSIDE)

        pen = QPen(COLOR_BAND, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for value, y in ((lo, y_lo), (hi, y_hi)):
            painter.drawLine(QPointF(image_rect.left(), y), QPointF(image_rect.right(), y))
            painter.drawText(QPointF(image_rect.left() + 6.0, y - 4.0), f"y={value:.5f}")

    def _draw_dets(self, painter: QPainter) -> None:
        assert self._frame is not None
        dets = self._frame.dets
        metrics = QFontMetricsF(self._label_font)
        for index, det in enumerate(dets):
            rect = self.tf.n2v_rect(det.bbox)
            color = det_color(det)
            selected = index == self._sel
            painter.setPen(QPen(color, 3 if selected else 2))
            painter.drawRect(rect)
            self._draw_tag(painter, metrics, rect, det.display_text(), color)
        if 0 <= self._sel < len(dets):
            self._draw_handles(painter, self.tf.n2v_rect(dets[self._sel].bbox))

    def _draw_tag(
        self, painter: QPainter, metrics: QFontMetricsF, rect: QRectF, text: str, color: QColor
    ) -> None:
        width = metrics.horizontalAdvance(text) + 8.0
        height = metrics.height() + 2.0
        top = rect.top() - height
        if top < 0.0:
            top = rect.top()
        tag = QRectF(rect.left(), top, width, height)
        painter.fillRect(tag, QColor(0, 0, 0, 170))
        painter.setPen(QPen(color))
        painter.drawText(tag.adjusted(4.0, 0.0, 0.0, 0.0),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)

    def _draw_handles(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(COLOR_SELECT, 1))
        painter.setBrush(QColor("#101010"))
        for hx, hy in HANDLES:
            c = self._handle_center(rect, hx, hy)
            painter.drawRect(
                QRectF(c.x() - HANDLE_HALF, c.y() - HANDLE_HALF, HANDLE_HALF * 2, HANDLE_HALF * 2)
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_new_box(self, painter: QPainter) -> None:
        if self._new_det is None:
            return
        painter.setPen(QPen(det_color(self._new_det), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.tf.n2v_rect(self._new_det.bbox))
