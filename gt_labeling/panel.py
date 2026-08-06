"""側邊面板:幀清單、選取框屬性、偵測帶設定。"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIntValidator, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .model import LABELS, FrameLabel

PPE_ITEMS: tuple[tuple[str, str | None], ...] = (("未定", None), ("ok", "ok"), ("ng", "ng"))
DEFAULT_MAX_GAP = 20
COLOR_PENDING = QColor("#ffb340")
COLOR_DUPLICATE = QColor("#ff5f56")
COLOR_NORMAL = QColor("#dcdcdc")


def _mono_font() -> QFont:
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(9)
    return font


class FrameListPanel(QWidget):
    """每幀一列:seq、框數、待補標記(ID / PPE)、id 重疊標記(DUP)、未存標記(*)。"""

    rowSelected = pyqtSignal(int)
    deleteRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._syncing = False
        self.list = QListWidget(self)
        self.list.setFont(_mono_font())
        self.list.setUniformItemSizes(True)
        # 可多選:Shift 拉一段連續的幀、Ctrl 加點零散的幀,用來圈出「這條軌跡從哪
        # 到哪是錯的」。currentRow 仍然只有一列,所以「目前顯示哪一幀」的語意不變,
        # 選取範圍是**額外**的資訊,只有 deleteRequested 會去讀它。
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.currentRowChanged.connect(self._on_row_changed)

        # Delete 綁在清單上(WidgetShortcut),畫布上的 Delete 仍是「刪這一個框」:
        # 同一顆鍵在兩處各做各的事,靠焦點在誰身上區分——人在清單裡拉範圍時,腦中
        # 想的就是「刪掉這一段」,跑去畫布刪單框反而是意外。
        delete = QAction("刪除選取幀內的整條軌跡", self.list)
        delete.setShortcuts(
            [QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace)]
        )
        delete.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        delete.triggered.connect(lambda: self.deleteRequested.emit())
        self.list.addAction(delete)

        self.summary = QLabel("—", self)
        self.summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("幀清單  seq  框數  待補  重疊  未存"))
        layout.addWidget(self.list, 1)
        layout.addWidget(self.summary)

    def set_frames(self, frames: list[FrameLabel]) -> None:
        self._syncing = True
        self.list.clear()
        for _ in frames:
            self.list.addItem(QListWidgetItem(""))
        self._syncing = False
        for index in range(len(frames)):
            self.refresh_row(index, frames[index])
        self.refresh_summary(frames)

    def refresh_row(self, index: int, frame: FrameLabel) -> None:
        item = self.list.item(index)
        if item is None:
            return
        flag_id = "ID " if frame.has_null_track else "   "
        flag_ppe = "PPE" if frame.has_null_ppe else "   "
        duplicate = frame.has_duplicate_track
        flag_dup = "DUP" if duplicate else "   "
        dirty = "*" if frame.dirty else " "
        item.setText(
            f"{frame.seq:>7d}  n={len(frame.dets):<3d} {flag_id}{flag_ppe} {flag_dup}  {dirty}"
        )

        # 重疊比待補嚴重:待補是還沒填,重疊是已經填錯,顏色要壓過橘色。
        pending = frame.has_null_track or frame.has_null_ppe
        if duplicate:
            item.setForeground(COLOR_DUPLICATE)
        else:
            item.setForeground(COLOR_PENDING if pending else COLOR_NORMAL)
        font = _mono_font()
        font.setBold(frame.dirty)
        item.setFont(font)

    def refresh_summary(self, frames: list[FrameLabel]) -> None:
        boxes = sum(len(f.dets) for f in frames)
        pending_boxes = sum(f.pending_count for f in frames)
        pending_frames = sum(1 for f in frames if f.pending_count)
        dirty_frames = sum(1 for f in frames if f.dirty)
        duplicate_frames = sum(1 for f in frames if f.has_duplicate_track)
        self.summary.setText(
            f"{len(frames)} 幀 / {boxes} 框\n"
            f"待補 {pending_boxes} 框(分布 {pending_frames} 幀)\n"
            f"id 重疊 {duplicate_frames} 幀\n"
            f"未存 {dirty_frames} 幀"
        )

    def selected_rows(self) -> list[int]:
        """目前反白的所有列(升冪)。只有一列時就是當前幀本身。"""
        return sorted(self.list.row(item) for item in self.list.selectedItems())

    def set_current(self, index: int) -> None:
        # currentRow 已經對就直接返回,不只是省事:setCurrentRow 預設會清掉其他選取,
        # 而使用者用 Shift 拉範圍時每一步都會走到這裡(rowSelected -> 切幀 -> 回頭
        # 同步),真的設下去會把剛拉好的範圍當場清空。
        if self.list.currentRow() == index:
            return
        self._syncing = True
        self.list.setCurrentRow(index)
        self.list.scrollToItem(self.list.item(index))
        self._syncing = False

    def _on_row_changed(self, row: int) -> None:
        if not self._syncing and row >= 0:
            self.rowSelected.emit(row)


class DeleteTrackDialog(QDialog):
    """刪掉一條軌跡在選取幀範圍內的所有框。

    做成對話框而不是常駐面板:這是偶爾才做一次的破壞性操作,常駐在幀清單下方
    只會一直佔著左欄的高度與寬度,而右欄那疊面板已經證明過「面板越疊越高」會
    把視窗頂大到超出螢幕。

    「刪哪一條」與「影響多大」擺在同一個視窗裡即時連動:改一次號碼就重算一次
    命中數,按鈕上直接寫出要刪幾個框。這樣不必先猜再看確認視窗,也不會出現
    「畫面寫 #7、實際刪 #3」——送出去的就是畫面上這一組值。
    """

    def __init__(
        self,
        scope: str,
        preview: Callable[[str, int], tuple[int, str]],
        initial: tuple[str, int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("刪除軌跡片段")
        self._preview = preview

        self.kind_box = QComboBox(self)
        self.kind_box.addItems(LABELS)
        self.kind_box.currentIndexChanged.connect(lambda _: self._refresh())

        self.track_edit = QLineEdit(self)
        self.track_edit.setValidator(QIntValidator(0, 2_000_000_000, self))
        self.track_edit.setPlaceholderText("track_id")
        self.track_edit.textChanged.connect(lambda _: self._refresh())

        self.detail = QLabel("", self)
        self.detail.setWordWrap(True)
        self.detail.setMinimumWidth(320)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow(QLabel(scope, self))
        form.addRow("label", self.kind_box)
        form.addRow("track_id", self.track_edit)
        form.addRow(self.detail)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

        if initial is not None:
            label, track_id = initial
            index = self.kind_box.findText(label)
            if index >= 0:
                self.kind_box.setCurrentIndex(index)
            self.track_edit.setText(str(track_id))
        self._refresh()
        self.track_edit.setFocus()
        self.track_edit.selectAll()

    def _refresh(self) -> None:
        """號碼一改就重算影響範圍,沒有命中就不讓按下去。"""
        target = self.target()
        if target is None:
            self.detail.setText("填一個 track_id。")
            count, ok_text = 0, "刪除"
        else:
            count, summary = self._preview(*target)
            self.detail.setText(summary)
            ok_text = f"刪除 {count} 個框" if count else "刪除"
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText(ok_text)
        ok.setEnabled(count > 0)

    def target(self) -> tuple[str, int] | None:
        text = self.track_edit.text().strip()
        return (self.kind_box.currentText(), int(text)) if text else None


class NewDetPanel(QWidget):
    """畫新框時要套用的預設屬性。

    只影響「接下來畫的框」,不動已存在的框(那是 DetPanel 的事)。放 radio 而非
    下拉:標註時最常做的動作就是在 person/drone 之間切換,一次點擊就到位。
    """

    changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, QRadioButton] = {}
        self._follow_id: int | None = None

        # 同一個 parent 下的 radio 預設全部互斥,label 與 track_id 是兩組獨立選擇,
        # 必須各自 QButtonGroup,否則選 drone 會把 track_id 的選擇彈掉。
        label_group = QButtonGroup(self)
        track_group = QButtonGroup(self)

        box = QVBoxLayout()
        for label, caption in (("person", "person(ppe=ng)"), ("drone", "drone(紅框)")):
            button = QRadioButton(caption, self)
            button.toggled.connect(
                lambda checked, name=label: checked and self.changed.emit(name)
            )
            self._buttons[label] = button
            label_group.addButton(button)
            box.addWidget(button)
        self._buttons["person"].setChecked(True)

        caption = QLabel("track_id", self)
        caption.setStyleSheet("color: #909090; margin-top: 6px;")
        box.addWidget(caption)

        self.auto_radio = QRadioButton("自動(本幀最大 +1)", self)
        self.auto_radio.setChecked(True)
        self.follow_radio = QRadioButton("沿用(先點一個框)", self)
        self.follow_radio.setEnabled(False)
        for button in (self.auto_radio, self.follow_radio):
            track_group.addButton(button)
            box.addWidget(button)

        group = QGroupBox("新框預設", self)
        group.setLayout(box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)

    def label(self) -> str:
        return next(
            (name for name, button in self._buttons.items() if button.isChecked()), LABELS[0]
        )

    def set_label(self, label: str) -> None:
        button = self._buttons.get(label)
        if button is not None:
            button.setChecked(True)

    def defaults(self) -> tuple[str, str | None]:
        """(label, ppe)。drone 沒有 ppe;person 直接判 ng(要修的幾乎都是 ng)。"""
        label = self.label()
        return label, ("ng" if label == "person" else None)

    # ------------------------------------------------------------ track_id 沿用

    def set_follow_candidate(self, track_id: int | None) -> None:
        """記住剛點到的 track_id。

        做成黏著值而非即時讀取選取框:canvas 在空白處按下時會先清掉選取才建新框,
        那時已經問不到「剛剛選的是誰」。點過一次就一路沿用,也正好符合補錨點的
        用法——連畫好幾幀都屬於同一條軌跡。
        """
        if track_id is None:
            return
        self._follow_id = track_id
        self.follow_radio.setText(f"沿用 #{track_id}")
        self.follow_radio.setEnabled(True)

    def follow_id(self) -> int | None:
        """選了沿用且有記住的號碼才回傳,否則 None 代表走自動取號。"""
        if self.follow_radio.isChecked():
            return self._follow_id
        return None


class DetPanel(QWidget):
    """選取框的屬性編輯。空白的 track_id 代表 null。"""

    fieldChanged = pyqtSignal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False

        self.label_box = QComboBox(self)
        self.label_box.addItems(LABELS)
        self.label_box.currentTextChanged.connect(
            lambda text: self._emit("label", text)
        )

        self.track_edit = QLineEdit(self)
        self.track_edit.setValidator(QIntValidator(0, 2_000_000_000, self))
        self.track_edit.setPlaceholderText("空白 = null(待指定)")
        self.track_edit.editingFinished.connect(self._emit_track)

        self.ppe_box = QComboBox(self)
        for text, value in PPE_ITEMS:
            self.ppe_box.addItem(text, value)
        self.ppe_box.currentIndexChanged.connect(
            lambda _: self._emit("ppe", self.ppe_box.currentData())
        )

        self.bbox_label = QLabel("—", self)
        self.bbox_label.setFont(_mono_font())
        self.bbox_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.addRow("label", self.label_box)
        form.addRow("track_id", self.track_edit)
        form.addRow("ppe", self.ppe_box)
        form.addRow("bbox", self.bbox_label)

        group = QGroupBox("選取的框", self)
        group.setLayout(form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)
        self.set_det(None)

    def set_det(self, det, image_size: tuple[int, int] | None = None) -> None:
        self._loading = True
        try:
            enabled = det is not None
            self.label_box.setEnabled(enabled)
            self.track_edit.setEnabled(enabled)
            if det is None:
                self.ppe_box.setEnabled(False)
                self.track_edit.clear()
                self.bbox_label.setText("—")
                return

            self.label_box.setCurrentText(det.label)
            self.track_edit.setText("" if det.track_id is None else str(det.track_id))
            # ppe 只對 person 有意義,選 drone 時鎖住並固定為 null。
            self.ppe_box.setEnabled(det.is_person)
            target = det.ppe if det.is_person else None
            index = self.ppe_box.findData(target)
            self.ppe_box.setCurrentIndex(max(index, 0))
            self.bbox_label.setText(self._bbox_text(det.bbox, image_size))
        finally:
            self._loading = False

    @staticmethod
    def _bbox_text(bbox, image_size: tuple[int, int] | None) -> str:
        x1, y1, x2, y2 = bbox
        text = f"{x1:.5f} {y1:.5f}\n{x2:.5f} {y2:.5f}"
        if image_size:
            w, h = image_size
            text += f"\n{(x2 - x1) * w:.0f} x {(y2 - y1) * h:.0f} px"
        return text

    def _emit(self, field: str, value) -> None:
        if not self._loading:
            self.fieldChanged.emit(field, value)

    def _emit_track(self) -> None:
        text = self.track_edit.text().strip()
        self._emit("track_id", int(text) if text else None)


class InterpolatePanel(QWidget):
    """補框設定:是否啟用、錨點最大間距。"""

    toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.enabled_box = QCheckBox("啟用內插補框", self)
        self.enabled_box.toggled.connect(self._on_toggled)

        self.gap_spin = QSpinBox(self)
        self.gap_spin.setRange(2, 300)
        self.gap_spin.setValue(DEFAULT_MAX_GAP)
        self.gap_spin.setSuffix(" 幀")

        hint = QLabel(
            "選一個框後按「補框」,把同 track_id 的相鄰錨點之間補滿。\n"
            "間距超過門檻的洞會跳過(目標可能被遮擋,不該憑空補)。\n"
            "實測建議門檻:drone 20 幀、person 30 幀。",
            self,
        )
        hint.setStyleSheet("color: #909090;")
        hint.setWordWrap(True)

        form = QFormLayout()
        form.addRow(self.enabled_box)
        form.addRow("錨點最大間距", self.gap_spin)
        form.addRow(hint)

        group = QGroupBox("內插補框", self)
        group.setLayout(form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)
        self._sync_enabled()

    def _on_toggled(self, checked: bool) -> None:
        self._sync_enabled()
        self.toggled.emit(checked)

    def _sync_enabled(self) -> None:
        self.gap_spin.setEnabled(self.enabled_box.isChecked())

    def is_enabled(self) -> bool:
        return self.enabled_box.isChecked()

    def max_gap(self) -> int:
        return self.gap_spin.value()

    def set_state(self, enabled: bool, max_gap: int) -> None:
        self.gap_spin.setValue(max_gap)
        self.enabled_box.setChecked(enabled)
        self._sync_enabled()


class BandPanel(QWidget):
    """偵測帶上下界(歸一化 y),提醒帶外不要標。"""

    bandChanged = pyqtSignal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False

        self.enabled_box = QCheckBox("顯示偵測帶", self)
        self.enabled_box.setChecked(True)
        self.enabled_box.toggled.connect(lambda _: self._emit())

        self.lo_spin = self._make_spin(0.0)
        self.hi_spin = self._make_spin(1.0)

        form = QFormLayout()
        form.addRow(self.enabled_box)
        form.addRow("上界 y1", self.lo_spin)
        form.addRow("下界 y2", self.hi_spin)

        group = QGroupBox("偵測帶", self)
        group.setLayout(form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)

    def _make_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(0.0, 1.0)
        spin.setDecimals(5)
        spin.setSingleStep(0.005)
        spin.setValue(value)
        spin.valueChanged.connect(lambda _: self._emit())
        return spin

    def set_band(self, lo: float, hi: float, enabled: bool = True) -> None:
        self._loading = True
        try:
            self.lo_spin.setValue(lo)
            self.hi_spin.setValue(hi)
            self.enabled_box.setChecked(enabled)
        finally:
            self._loading = False
        self._emit()

    def band(self) -> tuple[float | None, float | None]:
        if not self.enabled_box.isChecked():
            return None, None
        return self.lo_spin.value(), self.hi_spin.value()

    def _emit(self) -> None:
        if not self._loading:
            self.bandChanged.emit(*self.band())
